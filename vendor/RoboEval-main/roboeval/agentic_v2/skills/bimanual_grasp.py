"""Synchronized, verified two-arm grasp skill."""

from __future__ import annotations

from itertools import product

from roboeval.agentic_v2.constraints.bimanual import ee_targets_for_object_pose
from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND
from roboeval.agentic_v2.skills.base import BaseSkill
from roboeval.agentic_v2.state import capture_attachment, collect_scene_state
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


class BimanualGraspSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.BIMANUAL_GRASP:
            raise ValueError("BimanualGraspSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"object {object_name!r} is unavailable", [],
            )
        left = self.context.candidates.grasp_candidates(object_name, "left", state)
        right = self.context.candidates.grasp_candidates(object_name, "right", state)
        pairs = sorted(product(left, right), key=lambda pair: pair[0].score + pair[1].score)
        if not pairs:
            return self.failure(
                request, state, FailureCode.NO_VALID_GRASP,
                "no paired aperture-compatible candidates", [],
            )
        reports = []
        attempts = []
        policy = self._combined_policy(object_name, pairs[0])
        opened = self.context.executor.command_grippers(
            {"left": OPEN_COMMAND, "right": OPEN_COMMAND},
            constraints=ConstraintSet(allowed_contacts=policy),
        )
        reports.append(opened)
        if not opened.success:
            return self.failure(
                request, opened.final_state,
                opened.failure_code or FailureCode.EXECUTION_DIVERGED,
                "failed to open both grippers", reports,
            )

        for left_candidate, right_candidate in pairs:
            policy = self._combined_policy(object_name, (left_candidate, right_candidate))
            constraints = ConstraintSet(allowed_contacts=policy)
            terminal_constraints = ConstraintSet(
                allowed_contacts=AllowedContactPolicy(
                    rules=policy.rules,
                    penetration_tolerance=0.02,
                )
            )
            protected = {object_name: collect_scene_state(self.env).objects[object_name].pose}
            pre, pre_ik, pre_path = self.move(
                name=f"{object_name}_paired_pregrasp",
                targets={
                    "left": left_candidate.pregrasp_pose,
                    "right": right_candidate.pregrasp_pose,
                },
                constraints=constraints,
                protected_objects=protected,
                require_holds=False,
            )
            if pre is not None:
                reports.append(pre)
            if pre is None or not pre.success:
                attempts.append(self._failure_evidence("pregrasp", pre, pre_ik, pre_path))
                continue
            approach, approach_ik, approach_path = self.move(
                name=f"{object_name}_paired_approach",
                targets={
                    "left": left_candidate.grasp_pose,
                    "right": right_candidate.grasp_pose,
                },
                constraints=constraints,
                protected_objects=protected,
                require_holds=False,
                stop_condition=lambda state: set(
                    state.objects[object_name].held_by
                ) == {"left", "right"},
                terminal_constraints=terminal_constraints,
            )
            if approach is not None:
                reports.append(approach)
            if approach is None or not approach.success:
                attempts.append(self._failure_evidence("approach", approach, approach_ik, approach_path))
                self._recover(reports, policy)
                continue
            closed = self.context.executor.command_grippers(
                {"left": CLOSE_COMMAND, "right": CLOSE_COMMAND},
                steps=16,
                constraints=ConstraintSet(allowed_contacts=policy),
            )
            reports.append(closed)
            if not closed.success:
                attempts.append({
                    "stage": "close",
                    "failure": (closed.failure_code or FailureCode.EXECUTION_DIVERGED).value,
                })
                self._recover(reports, policy)
                continue
            grasped = collect_scene_state(self.env)
            if set(grasped.objects[object_name].held_by) != {"left", "right"}:
                attempts.append({
                    "stage": "close",
                    "failure": FailureCode.GRASP_FAILED.value,
                    "observed_holders": grasped.objects[object_name].held_by,
                })
                self._recover(reports, policy)
                continue
            attachments = (
                capture_attachment(grasped, object_name, "left"),
                capture_attachment(grasped, object_name, "right"),
            )
            verified = self._verify_lift(object_name, attachments, policy, reports)
            if not verified:
                attempts.append({"stage": "verify_lift", "failure": FailureCode.SLIP_DETECTED.value})
                self._recover(reports, policy)
                continue
            final = collect_scene_state(self.env)
            for side in ("left", "right"):
                self.context.attachments[(object_name, side)] = capture_attachment(final, object_name, side)
            return SkillResult(
                request, True, f"verified bimanual grasp of {object_name}", final,
                execution_reports=tuple(reports),
                diagnostics={"attempts_before_success": len(attempts)},
            )
        return self.failure(
            request, collect_scene_state(self.env), FailureCode.NO_VALID_GRASP,
            "all paired grasp candidates failed", reports, {"attempts": attempts},
        )

    @staticmethod
    def _combined_policy(object_name, candidates) -> AllowedContactPolicy:
        rules = tuple(
            rule for candidate in candidates for rule in candidate.contact_policy.rules
        ) + (
            AllowedContactRule(f"object:{object_name}", "scene:*table*"),
            AllowedContactRule(f"object:{object_name}", "scene:cabinet_*"),
        )
        return AllowedContactPolicy(rules=rules, penetration_tolerance=0.01)

    def _verify_lift(self, object_name, attachments, policy, reports) -> bool:
        state = collect_scene_state(self.env)
        obj = state.objects[object_name].pose
        target_object = Pose(
            (obj.position[0], obj.position[1], obj.position[2] + 0.025),
            obj.quaternion_wxyz,
        )
        constraints = ConstraintSet(
            allowed_contacts=policy,
            held_objects=attachments,
            position_tolerance=0.05,
            orientation_tolerance=0.30,
            maximum_object_tilt=0.35,
        )
        execution, _, _ = self.move(
            name=f"{object_name}_bimanual_verification_lift",
            targets=ee_targets_for_object_pose(target_object, attachments),
            constraints=constraints,
            require_holds=True,
            candidate_count=7,
        )
        if execution is not None:
            reports.append(execution)
        if execution is None or not execution.success:
            return False
        final = collect_scene_state(self.env)
        return set(final.objects[object_name].held_by) == {"left", "right"}

    def _recover(self, reports, policy) -> None:
        constraints = ConstraintSet(allowed_contacts=policy)
        report = self.context.executor.command_grippers(
            {"left": OPEN_COMMAND, "right": OPEN_COMMAND},
            constraints=constraints,
        )
        reports.append(report)
        for side in ("left", "right"):
            for key in tuple(self.context.attachments):
                if key[1] == side:
                    self.context.attachments.pop(key, None)
        for side in ("left", "right"):
            retreat = self.retreat(side)
            if retreat is not None:
                reports.append(retreat)

    @staticmethod
    def _failure_evidence(stage, execution, ik_result, path_result):
        if execution is not None:
            code = execution.failure_code
        elif path_result is not None:
            code = path_result.report.failure_code
        else:
            code = ik_result.report.failure_code
        return {
            "stage": stage,
            "failure": code.value if code else None,
            "ik_accepted": len(ik_result.accepted),
            "path_attempted": path_result.attempted_paths if path_result else 0,
            "ik_rejections": [
                {
                    "seed": candidate.seed_name,
                    "failure": candidate.failure_code.value
                    if candidate.failure_code else None,
                    "contacts": candidate.diagnostics.get("contacts", ()),
                }
                for candidate in ik_result.rejected
            ],
        }
