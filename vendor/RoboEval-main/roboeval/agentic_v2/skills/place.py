"""Geometry-derived placement with release and settle verification."""

from __future__ import annotations

import numpy as np

from roboeval.agentic_v2.constraints.bimanual import ee_targets_for_object_pose
from roboeval.agentic_v2.evaluator import benchmark_success
from roboeval.agentic_v2.executor import OPEN_COMMAND
from roboeval.agentic_v2.skills.base import BaseSkill
from roboeval.agentic_v2.skills.transport import held_constraints
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    FailureCode,
    PlacementCandidate,
    Pose,
    SkillName,
    SkillRequest,
    SkillResult,
)


class PlaceSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.PLACE:
            raise ValueError("PlaceSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects or not state.objects[object_name].held_by:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"{object_name!r} must be held before placement", [],
            )
        support = self._support_name(request, state)
        if support is None:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"placement goal must name a support object: {request.goal!r}", [],
            )
        reports = []
        attempts = []
        if state.objects[support].fixed:
            candidates = self.shelf_candidates(object_name, support, state)
            attempt = self._attempt
        else:
            candidates = self.placement_candidates(object_name, support, state)
            attempt = self._attempt
        for candidate in candidates:
            result = attempt(request, candidate, reports, attempts)
            if result is not None:
                return result
        return self.failure(
            request, collect_scene_state(self.env), FailureCode.PLACEMENT_UNREACHABLE,
            "no stable reachable placement candidate", reports,
            {"attempts": attempts},
        )

    def _attempt(self, request, candidate, reports, attempts):
        object_name = candidate.object_name
        state = collect_scene_state(self.env)
        holders = state.objects[object_name].held_by
        attachments = tuple(
            self.context.attachments.get((object_name, side))
            or capture_attachment(state, object_name, side)
            for side in holders
        )
        carry_rules = tuple(self.context.carry_contact_rules.get(object_name, ()))
        # The object may still be grazing the surface it was picked from
        # (a heavy object sags in the grip); leaving that surface is part of
        # the move, not a collision.
        departure_rules = (
            AllowedContactRule(f"object:{object_name}", "scene:*table*"),
            AllowedContactRule(f"object:{object_name}", "scene:cabinet_*"),
        )
        constraints = held_constraints(
            object_name, attachments,
            extra_rules=candidate.contact_policy.rules + carry_rules + departure_rules,
        )
        pre, pre_ik, pre_path = self.move(
            name=f"{candidate.name}_preplace",
            targets=ee_targets_for_object_pose(candidate.preplace_object_pose, attachments),
            constraints=constraints,
            require_holds=True,
            candidate_count=9,
        )
        if pre is not None:
            reports.append(pre)
        if pre is None or not pre.success:
            attempts.append(self._evidence(candidate.name, "preplace", pre, pre_ik, pre_path))
            return None
        terminal = ConstraintSet(
            allowed_contacts=AllowedContactPolicy(candidate.contact_policy.rules + carry_rules, 0.015),
            held_objects=attachments,
            position_tolerance=0.06,
            orientation_tolerance=0.35,
        )
        lower, lower_ik, lower_path = self.move(
            name=f"{candidate.name}_lower",
            targets=ee_targets_for_object_pose(candidate.placed_object_pose, attachments),
            constraints=constraints,
            require_holds=True,
            candidate_count=9,
            stop_condition=lambda observed: f"object:{candidate.support_name}" in observed.objects[object_name].contacts,
            terminal_constraints=terminal,
        )
        if lower is not None:
            reports.append(lower)
        if lower is None or not lower.success:
            attempts.append(self._evidence(candidate.name, "lower", lower, lower_ik, lower_path))
            return None
        released = self.context.executor.command_grippers(
            {side: OPEN_COMMAND for side in holders},
            steps=16,
            constraints=terminal,
        )
        reports.append(released)
        if not released.success:
            return self.failure(
                request, released.final_state, FailureCode.RELEASE_FAILED,
                "gripper opening failed during placement", reports,
            )
        for side in holders:
            self.context.attachments.pop((object_name, side), None)
        self.context.carry_modes.pop(object_name, None)
        self.context.edge_pinch_depths.pop(object_name, None)
        self.context.carry_contact_rules.pop(object_name, None)

        free_constraints = ConstraintSet(
            allowed_contacts=AllowedContactPolicy(candidate.contact_policy.rules + carry_rules, 0.012),
        )
        for side in holders:
            current = collect_scene_state(self.env).robot.arms[side].ee_pose
            approach = np.asarray(current.as_matrix())[:3, 2]
            if abs(approach[2]) < 0.5:
                # Horizontal hand (edge grasp): one finger is still under the
                # object, so back straight out along the approach first;
                # lifting immediately would flip the object off its support.
                back_target = Pose(
                    tuple(np.asarray(current.position) - 0.08 * approach),
                    current.quaternion_wxyz,
                )
                back, _, _ = self.move(
                    name=f"place_release_back_{side}", targets={side: back_target},
                    constraints=free_constraints, require_holds=False,
                )
                if back is not None:
                    reports.append(back)
                    current = collect_scene_state(self.env).robot.arms[side].ee_pose
            retreat_target = Pose(
                (current.position[0], current.position[1], current.position[2] + 0.08),
                current.quaternion_wxyz,
            )
            retreat, _, _ = self.move(
                name=f"place_release_retreat_{side}", targets={side: retreat_target},
                constraints=free_constraints, require_holds=False,
            )
            if retreat is not None:
                reports.append(retreat)
        settle = self.context.executor.execute(
            self.context.executor.hold_plan("placement_settle", steps=20, constraints=free_constraints),
            require_holds=False,
        )
        reports.append(settle)
        final = collect_scene_state(self.env)
        held = final.objects[object_name].held_by
        on_support = f"object:{candidate.support_name}" in final.objects[object_name].contacts
        on_table = any("table" in contact for contact in final.objects[object_name].contacts)
        valid = not held and on_support and not on_table
        if state.task_key == "stack_two_blocks":
            valid = valid and benchmark_success(final) >= 1.0
        if not valid:
            return self.failure(
                request, final,
                FailureCode.RELEASE_FAILED if held else FailureCode.POSTCONDITION_FAILED,
                "placement did not settle into the required support-only contact", reports,
                {"held_by": held, "contacts": final.objects[object_name].contacts},
            )
        return SkillResult(
            request, True, f"placed {object_name} on {candidate.support_name}", final,
            execution_reports=tuple(reports),
            diagnostics={"candidate": candidate.name, "contacts": final.objects[object_name].contacts},
        )

    @staticmethod
    def placement_candidates(object_name, support_name, state):
        obj = state.objects[object_name]
        support = state.objects[support_name]
        base_z = support.pose.position[2] + 0.5 * (support.size[2] + obj.size[2])
        offsets = ((0.0, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006))
        maximum_com_offset = (
            max(0.0, 0.5 * support.size[0] - 0.25 * obj.size[0]),
            max(0.0, 0.5 * support.size[1] - 0.25 * obj.size[1]),
        )
        stable_offsets = tuple(
            (dx, dy)
            for dx, dy in offsets
            if abs(dx) <= maximum_com_offset[0]
            and abs(dy) <= maximum_com_offset[1]
        )
        policy = AllowedContactPolicy(
            (
                *(AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}") for side in ("left", "right")),
                AllowedContactRule(f"object:{object_name}", f"object:{support_name}"),
            ),
            penetration_tolerance=0.012,
        )
        return tuple(
            PlacementCandidate(
                f"{object_name}_on_{support_name}_{index}", object_name, support_name,
                Pose((support.pose.position[0] + dx, support.pose.position[1] + dy, base_z + 0.08), obj.pose.quaternion_wxyz),
                Pose((support.pose.position[0] + dx, support.pose.position[1] + dy, base_z + 0.001), obj.pose.quaternion_wxyz),
                policy, float(np.hypot(dx, dy)),
            )
            for index, (dx, dy) in enumerate(stable_offsets)
        )

    def shelf_candidates(self, object_name, support_name, state):
        """Place a held flat object onto a fixed plank (shelf). The object
        keeps its current orientation; it is set down so that its near edge
        overhangs the plank's front (robot-facing) edge by a few cm - the
        hand gripping that edge has a finger *under* the object, which would
        otherwise collide with the plank - while its center stays well over
        the plank."""

        obj = state.objects[object_name]
        support = state.objects[support_name]
        support_center = np.asarray(support.aabb_center)
        support_size = np.asarray(support.size)
        # Set the object down the way it lay at rest (an edge-grasped book
        # is carried standing up); size it in that orientation.
        resting = self.context.resting_orientations.get(object_name, obj.pose.quaternion_wxyz)
        rest_rotation = np.asarray(Pose((0.0, 0.0, 0.0), resting).as_matrix())[:3, :3]
        obj_size = np.abs(rest_rotation) @ np.asarray(obj.canonical_size)
        front_x = support_center[0] - support_size[0] / 2.0
        back_x = support_center[0] + support_size[0] / 2.0
        top_z = support_center[2] + support_size[2] / 2.0
        base_z = top_z + obj_size[2] / 2.0
        current_y = obj.pose.position[1]
        policy = AllowedContactPolicy(
            (
                *(AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}") for side in ("left", "right")),
                AllowedContactRule(f"object:{object_name}", f"object:{support_name}"),
            ),
            penetration_tolerance=0.012,
        )
        result = []
        index = 0
        # The lower finger of an edge grasp sits under the object about
        # (pinch depth + 1.2 cm) in from its near edge; that much of the
        # object must overhang the support's front edge or the finger lands
        # on the plank. Fall back to a shallow default for other grasps.
        depth = self.context.edge_pinch_depths.get(object_name)
        base_overhang = (depth + 0.012) if depth is not None else 0.035
        for overhang in (base_overhang, base_overhang + 0.01, base_overhang - 0.01):
            center_x = front_x - overhang + obj_size[0] / 2.0
            if center_x < front_x + 0.01 or center_x + obj_size[0] / 2.0 > back_x - 0.005:
                continue
            for dy in (0.0, 0.06, -0.06):
                y = current_y + dy
                if abs(y - support_center[1]) > support_size[1] / 2.0 - obj_size[1] / 2.0:
                    continue
                result.append(
                    PlacementCandidate(
                        f"{object_name}_on_{support_name}_{index}", object_name, support_name,
                        Pose((center_x, y, base_z + 0.06), resting),
                        Pose((center_x, y, base_z + 0.002), resting),
                        policy, float(abs(dy) + abs(overhang - 0.035)),
                    )
                )
                index += 1
        return tuple(sorted(result, key=lambda candidate: candidate.score))

    @staticmethod
    def _support_name(request, state):
        goal = request.goal.strip().lower()
        if goal.startswith("on:") and goal[3:] in state.objects:
            return goal[3:]
        others = [name for name in state.objects if name != request.object_name and not state.objects[name].fixed]
        return others[0] if request.strategy == "stack" and len(others) == 1 else None

    @staticmethod
    def _evidence(name, stage, execution, ik_result, path_result):
        code = execution.failure_code if execution is not None else (
            path_result.report.failure_code if path_result is not None else ik_result.report.failure_code
        )
        return {"candidate": name, "stage": stage, "failure": code.value if code else None}
