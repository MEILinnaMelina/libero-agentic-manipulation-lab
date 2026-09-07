"""Verified single-arm grasp skill."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND
from roboeval.agentic_v2.motion.candidate_generator import EDGE_GRASP_DEEP_INSET, EDGE_GRASP_PAD_INSET
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state, vertical_half_extent
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    FailureCode,
    Pose,
    SkillName,
    SkillRequest,
    SkillResult,
)
from roboeval.agentic_v2.skills.base import BaseSkill


class GraspSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.GRASP:
            raise ValueError("GraspSkill received a non-grasp request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"object {object_name!r} is unavailable", [],
            )
        if state.objects[object_name].fixed:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"{object_name} is a fixed scene part and cannot be grasped", [],
            )
        side = self._select_side(request, state)
        if side in state.objects[object_name].held_by:
            return SkillResult(request, True, f"{side} already holds {object_name}", state)
        candidates = self.context.candidates.grasp_candidates(object_name, side, state)
        if not candidates:
            return self.failure(
                request, state, FailureCode.NO_VALID_GRASP,
                f"no aperture-compatible grasp for {object_name}", [],
            )

        reports = []
        attempts: list[dict[str, Any]] = []
        open_report = self.context.executor.command_gripper(
            side,
            OPEN_COMMAND,
            constraints=ConstraintSet(allowed_contacts=candidates[0].contact_policy),
        )
        reports.append(open_report)
        if not open_report.success:
            return self.failure(
                request, open_report.final_state,
                open_report.failure_code or FailureCode.EXECUTION_DIVERGED,
                "failed to open before approach", reports,
            )
        # The object is still untouched and the scene has had the open
        # command's steps to settle: record the surface it rests on so a
        # later staged handover can place it back at this height.
        settled = open_report.final_state.objects.get(object_name)
        if settled is not None and not settled.held_by:
            self.context.resting_surfaces[object_name] = float(
                settled.pose.position[2] - vertical_half_extent(settled)
            )
            self.context.resting_orientations[object_name] = tuple(settled.pose.quaternion_wxyz)
        # Re-derive the candidates from the settled pose: some tasks spawn
        # the object above its support (the standing rod starts 5 cm in the
        # air), and a grasp planned against the pre-settle pose would close
        # above the object.
        candidates = self.context.candidates.grasp_candidates(
            object_name, side, open_report.final_state
        ) or candidates

        for candidate in candidates:
            result = self._attempt(request, candidate, reports)
            attempts.append(result[1])
            if result[0] is not None:
                return result[0]
        state = collect_scene_state(self.env)
        return self.failure(
            request,
            state,
            FailureCode.NO_VALID_GRASP,
            f"all {len(candidates)} grasp candidates failed",
            reports,
            {"attempts": attempts},
        )

    def _attempt(self, request, candidate, reports):
        object_name = candidate.object_name
        side = candidate.side
        initial = collect_scene_state(self.env)
        protected = {object_name: initial.objects[object_name].pose}
        pre_execution, pre_ik, pre_path = self.move(
            name=f"{candidate.name}_pregrasp",
            targets={side: candidate.pregrasp_pose},
            protected_objects=protected,
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "pregrasp", pre_execution, pre_ik, pre_path)
        if pre_execution is None or not pre_execution.success:
            if pre_execution is not None:
                reports.append(pre_execution)
                self._record_recovery(reports, side)
            return None, evidence
        reports.append(pre_execution)

        approach_constraints = ConstraintSet(allowed_contacts=candidate.contact_policy)
        # No stop_condition: is_gripper_holding_object() is a pure
        # pad-contact check with no requirement that the gripper be closed
        # or gripping firmly (roboeval/robots/gripper.py:184), so an early
        # stop on "first contact" can leave the wrist short of the
        # precisely-computed grasp pose - most visible for objects resting
        # close to a support surface, where the shortfall leaves fingers
        # nearer the table on close. Travel the full planned path instead.
        approach, approach_ik, approach_path = self.move(
            name=f"{candidate.name}_approach",
            targets={side: candidate.grasp_pose},
            constraints=approach_constraints,
            protected_objects=protected,
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "approach", approach, approach_ik, approach_path)
        if approach is None or not approach.success:
            if approach is not None:
                reports.append(approach)
            self._record_recovery(reports, side)
            return None, evidence
        reports.append(approach)

        close_report = self.context.executor.command_gripper(
            side,
            CLOSE_COMMAND,
            steps=16,
            constraints=approach_constraints,
        )
        reports.append(close_report)
        if not close_report.success:
            self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "close", "failure": close_report.failure_code.value})
            return None, evidence
        grasped = collect_scene_state(self.env)
        if side not in grasped.objects[object_name].held_by:
            self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "verify_contact", "failure": FailureCode.GRASP_FAILED.value})
            return None, evidence

        attachment = capture_attachment(grasped, object_name, side)
        if candidate.edge_grasp and self.context.consolidate_edge_grasps:
            # Deepen the pinch while the object still rests on its support
            # (ends with its own verification lift).
            consolidated = self._consolidate_edge_grasp(candidate, attachment, reports)
            if consolidated is None:
                self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
                evidence.update({"stage": "consolidate", "failure": FailureCode.SLIP_DETECTED.value})
                return None, evidence
            attachment = capture_attachment(collect_scene_state(self.env), object_name, side)
        else:
            verified = self._verify_lift(candidate, attachment, reports)
            if verified is None:
                self._record_recovery(reports, side, open_first=True, constraints=approach_constraints)
                evidence.update({"stage": "verify_lift", "failure": FailureCode.SLIP_DETECTED.value})
                return None, evidence
        self.context.attachments[(object_name, side)] = attachment
        final_state = collect_scene_state(self.env)
        return SkillResult(
            request,
            True,
            f"verified {side} grasp of {object_name}",
            final_state,
            execution_reports=tuple(reports),
            diagnostics={"candidate": candidate.name, "attachment": attachment},
        ), evidence

    def _verify_lift(self, candidate, attachment, reports):
        state = collect_scene_state(self.env)
        side = candidate.side
        current = state.robot.arms[side].ee_pose
        target = Pose(
            (current.position[0], current.position[1], current.position[2] + 0.025),
            current.quaternion_wxyz,
        )
        if candidate.edge_grasp:
            # An edge grasp holds a thin object overhanging its support; a
            # pure vertical lift is fine there too, but pull back a little
            # along the approach as well so the lower finger clears the
            # support's front edge rather than scraping up along it.
            approach = np.asarray(candidate.approach_axis, dtype=float)
            target = Pose(
                tuple(np.asarray(target.position) - 0.01 * approach),
                target.quaternion_wxyz,
            )
        departure_policy = AllowedContactPolicy(
            rules=candidate.contact_policy.rules
            + (
                AllowedContactRule(f"object:{candidate.object_name}", "scene:*table*"),
                AllowedContactRule(f"object:{candidate.object_name}", "scene:cabinet_*"),
            )
            + tuple(self.context.carry_contact_rules.get(candidate.object_name, ())),
            penetration_tolerance=0.01,
        )
        constraints = ConstraintSet(
            allowed_contacts=departure_policy,
            held_objects=(attachment,),
            position_tolerance=0.035,
            orientation_tolerance=0.25,
        )
        execution, _, _ = self.move(
            name=f"{candidate.name}_verification_lift",
            targets={side: target},
            constraints=constraints,
            require_holds=True,
            candidate_count=5,
        )
        if execution is not None:
            reports.append(execution)
        if execution is None or not execution.success:
            return None
        final_state = collect_scene_state(self.env)
        if side not in final_state.objects[candidate.object_name].held_by:
            return None
        return execution

    def _consolidate_edge_grasp(self, candidate, attachment, reports):
        """Deepen an edge pinch by dragging the object further off its
        support and re-gripping closer to its center.

        The book is 4 kg. Pinched flat at one short edge (1.3 cm pad inset,
        6.9 cm lever to its center) the gravity moment is ~2.7 Nm, far past
        what two 2 cm pads resist by friction, so it creeps inside the grip
        during any lift (measured: 2.8 cm rise for a 9 cm hand rise). Both
        ways of re-orienting it were measured to saturate the 12 Nm
        wrist-pitch joint (standing it over the pinch: 0.29 rad steady
        error; raising it flat: hand sinking 10 cm). What does work is a
        shorter lever: with the object still resting on its support (which
        carries the weight), drag it back so ~8 cm overhangs, let go, slide
        the open fingers 6 cm under the overhang and re-grip; the moment is
        then ~0.9 Nm and the flat carry keeps the load within the wrist's
        limits. The object's center stays over the support the whole time."""

        side = candidate.side
        object_name = candidate.object_name
        approach = np.asarray(candidate.approach_axis, dtype=float)
        approach[2] = 0.0
        approach /= max(np.linalg.norm(approach), 1e-9)
        policy = AllowedContactPolicy(
            rules=candidate.contact_policy.rules
            + (
                AllowedContactRule(f"object:{object_name}", "scene:*table*", penetration_tolerance=0.02),
                AllowedContactRule(f"object:{object_name}", "scene:cabinet_*", penetration_tolerance=0.02),
            ),
            penetration_tolerance=0.012,
        )
        drag_constraints = ConstraintSet(
            allowed_contacts=policy,
            held_objects=(attachment,),
            position_tolerance=0.05,
            orientation_tolerance=0.5,
        )
        state = collect_scene_state(self.env)
        obj = state.objects[object_name]
        # Target pad depth: as deep as the palm allows (the object's edge
        # then butts against the palm, which is what makes the pinch rigid;
        # measured: a 4.2 cm pinch on the book just grazes the palm), never
        # past the object's center.
        length = float(np.abs(np.asarray(obj.pose.as_matrix())[:3, :3].T @ approach) @ np.asarray(obj.canonical_size))
        margin = 0.012
        depth = min(EDGE_GRASP_DEEP_INSET, length / 2.0 - margin - 0.012)
        needed = depth + 0.012 + margin
        # The pads slip along the object during a drag (the support's
        # friction on a 4 kg object beats the pad shear), so drag on the
        # live overhang, up to twice, without treating slip as failure.
        for attempt in range(2):
            overhang = self.context.candidates.edge_overhang(object_name, collect_scene_state(self.env))
            drag = max(0.0, needed - overhang) if overhang is not None else 0.0
            if drag <= 0.005:
                break
            current = collect_scene_state(self.env).robot.arms[side].ee_pose
            drag_target = Pose(
                tuple(np.asarray(current.position) - drag * approach),
                current.quaternion_wxyz,
            )
            dragged, _, _ = self.move(
                name=f"{candidate.name}_deepen_drag_{attempt}",
                targets={side: drag_target},
                constraints=drag_constraints,
                require_holds=False,
                candidate_count=7,
            )
            if dragged is not None:
                reports.append(dragged)
            if dragged is None or not dragged.success:
                return None
        free = ConstraintSet(allowed_contacts=AllowedContactPolicy(policy.rules, 0.02))
        opened = self.context.executor.command_gripper(side, OPEN_COMMAND, constraints=free)
        reports.append(opened)
        if not opened.success:
            return None
        state = collect_scene_state(self.env)
        overhang = self.context.candidates.edge_overhang(object_name, state)
        if overhang is not None:
            depth = min(depth, overhang - 0.012 - margin)
        if depth < EDGE_GRASP_PAD_INSET:
            return None
        current = state.robot.arms[side].ee_pose
        pad = np.asarray(current.position) + 0.1034 * np.asarray(current.as_matrix())[:3, 2]
        near_edge = np.asarray(state.objects[object_name].aabb_center) - 0.5 * np.asarray(state.objects[object_name].size) * np.sign(approach)
        current_depth = float(np.dot(pad - near_edge, approach))
        advance = depth - current_depth
        deeper_target = Pose(
            tuple(np.asarray(current.position) + advance * approach),
            current.quaternion_wxyz,
        )
        protected = {object_name: state.objects[object_name].pose}
        advanced, _, _ = self.move(
            name=f"{candidate.name}_deepen_advance",
            targets={side: deeper_target},
            constraints=ConstraintSet(allowed_contacts=candidate.contact_policy),
            protected_objects=protected,
            require_holds=False,
            candidate_count=7,
        )
        if advanced is not None:
            reports.append(advanced)
        if advanced is None or not advanced.success:
            return None
        closed = self.context.executor.command_gripper(
            side, CLOSE_COMMAND, steps=16,
            constraints=ConstraintSet(allowed_contacts=candidate.contact_policy),
        )
        reports.append(closed)
        grasped = collect_scene_state(self.env)
        if not closed.success or side not in grasped.objects[object_name].held_by:
            return None
        new_attachment = capture_attachment(grasped, object_name, side)
        # The object's edge now sits against (or a millimetre from) the
        # palm; that contact is part of the hold from here on.
        palm_rule = AllowedContactRule(f"robot:{side}:link", f"object:{object_name}", penetration_tolerance=0.012)
        self.context.carry_contact_rules[object_name] = (palm_rule,)
        verified = self._verify_lift(candidate, new_attachment, reports)
        if verified is None:
            self.context.carry_contact_rules.pop(object_name, None)
            return None
        self.context.carry_modes[object_name] = f"edge:{depth:.3f}"
        self.context.edge_pinch_depths[object_name] = float(depth)
        return verified

    def _record_recovery(
        self,
        reports,
        side: str,
        *,
        open_first: bool = False,
        constraints: ConstraintSet | None = None,
    ) -> None:
        if open_first:
            lenient = (
                ConstraintSet(
                    allowed_contacts=AllowedContactPolicy(
                        rules=constraints.allowed_contacts.rules,
                        penetration_tolerance=0.02,
                    )
                )
                if constraints is not None
                else None
            )
            opened = self.context.executor.command_gripper(
                side,
                OPEN_COMMAND,
                constraints=lenient,
            )
            reports.append(opened)
            constraints = lenient
        for key in tuple(self.context.attachments):
            if key[1] == side:
                self.context.attachments.pop(key, None)
        # Fingers that just opened are still touching the object; the
        # retreat must tolerate that or it is vetoed before it starts.
        retreat = self.retreat(side, constraints=constraints)
        if retreat is not None:
            reports.append(retreat)

    @staticmethod
    def _select_side(request: SkillRequest, state) -> str:
        direct = [side for side in ("left", "right") if side in request.roles]
        if len(direct) == 1:
            return direct[0]
        for role in ("donor", "receiver"):
            value = request.roles.get(role)
            if value in ("left", "right"):
                return value
        if request.strategy in ("left", "right"):
            return request.strategy
        obj = state.objects[request.object_name]
        return min(
            ("left", "right"),
            key=lambda side: sum(
                (state.robot.arms[side].ee_pose.position[index] - obj.pose.position[index]) ** 2
                for index in range(3)
            ),
        )

    @staticmethod
    def _evidence(name, stage, execution, ik_result, path_result):
        if execution is not None:
            code = execution.failure_code
        elif path_result is not None:
            code = path_result.report.failure_code
        else:
            code = ik_result.report.failure_code
        return {
            "candidate": name,
            "stage": stage,
            "failure": code.value if code else None,
            "ik_accepted": len(ik_result.accepted),
            "ik_rejected": len(ik_result.rejected),
            "path_attempted": path_result.attempted_paths if path_result else 0,
        }
