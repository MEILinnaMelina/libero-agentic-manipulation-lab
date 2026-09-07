"""Verified one- or two-arm semantic lift skill."""

from __future__ import annotations

from roboeval.agentic_v2.constraints.bimanual import ee_targets_for_object_pose, object_tilt_change
from roboeval.agentic_v2.evaluator import benchmark_success
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


_HEIGHTS = {
    "small": 0.05,
    "clear_table": 0.10,
    "task_height": 0.14,
}


class LiftSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.LIFT:
            raise ValueError("LiftSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"object {object_name!r} is unavailable", [],
            )
        holders = state.objects[object_name].held_by
        if not holders:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"{object_name} is not held", [],
            )
        if state.objects[object_name].fixed:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"{object_name} is a fixed scene part and cannot be lifted", [],
            )
        attachments = tuple(
            self.context.attachments.get((object_name, side))
            or capture_attachment(state, object_name, side)
            for side in holders
        )
        strategy = request.strategy or (
            "task_height" if state.task_key == "lift_pot" else "clear_table"
        )
        if strategy not in _HEIGHTS:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"unknown lift strategy {strategy!r}", [],
            )
        height = _HEIGHTS[strategy]
        start_pose = state.objects[object_name].pose
        target_object = Pose(
            (
                start_pose.position[0],
                start_pose.position[1],
                start_pose.position[2] + height,
            ),
            start_pose.quaternion_wxyz,
        )
        policy = AllowedContactPolicy(
            rules=tuple(
                AllowedContactRule(f"robot:{side}:finger", f"object:{object_name}")
                for side in holders
            )
            + (AllowedContactRule(f"object:{object_name}", "scene:*table*"),)
            + tuple(self.context.carry_contact_rules.get(object_name, ())),
            penetration_tolerance=0.01,
        )
        constraints = ConstraintSet(
            allowed_contacts=policy,
            held_objects=attachments,
            position_tolerance=0.05,
            orientation_tolerance=0.30,
            maximum_object_tilt=0.35,
        )
        execution, ik_result, path_result = self.move(
            name=f"lift_{object_name}_{strategy}",
            targets=ee_targets_for_object_pose(target_object, attachments),
            constraints=constraints,
            require_holds=True,
            candidate_count=9,
        )
        reports = [execution] if execution is not None else []
        if execution is None or not execution.success:
            code = (
                execution.failure_code if execution is not None
                else path_result.report.failure_code if path_result is not None
                else ik_result.report.failure_code
            ) or FailureCode.EXECUTION_DIVERGED
            return self.failure(
                request, collect_scene_state(self.env), code,
                f"lift planning/execution failed: {code.value}", reports,
            )
        final = collect_scene_state(self.env)
        rise = final.objects[object_name].pose.position[2] - start_pose.position[2]
        # Raw benchmark success is authoritative whenever the task is a
        # lift; otherwise require most of the requested rise.
        task_height_reached = benchmark_success(final) >= 1.0 or rise >= 0.75 * height
        if not task_height_reached or any(
            side not in final.objects[object_name].held_by for side in holders
        ):
            return self.failure(
                request, final, FailureCode.POSTCONDITION_FAILED,
                "lift did not preserve height and verified holds", reports,
                {
                    "requested_height": height,
                    "actual_rise": rise,
                    "benchmark_success": benchmark_success(final),
                },
            )
        if object_tilt_change(start_pose, final.objects[object_name].pose) > constraints.maximum_object_tilt:
            return self.failure(
                request, final, FailureCode.CONSTRAINT_VIOLATION,
                "object tilt exceeded the lift bound", reports,
            )
        for side in holders:
            self.context.attachments[(object_name, side)] = capture_attachment(final, object_name, side)
        return SkillResult(
            request, True, f"lifted {object_name} using {holders}", final,
            execution_reports=tuple(reports),
            diagnostics={
                "requested_height": height,
                "actual_rise": rise,
                "benchmark_success": benchmark_success(final),
            },
        )
