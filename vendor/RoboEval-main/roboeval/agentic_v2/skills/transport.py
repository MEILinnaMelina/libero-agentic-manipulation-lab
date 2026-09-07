"""Collision-checked transport of an already held object."""

from __future__ import annotations

import numpy as np

from roboeval.agentic_v2.constraints.bimanual import ee_targets_for_object_pose
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


def held_constraints(object_name, attachments, *, extra_rules=()) -> ConstraintSet:
    rules = tuple(
        AllowedContactRule(f"robot:{attachment.side}:finger", f"object:{object_name}")
        for attachment in attachments
    ) + tuple(extra_rules)
    return ConstraintSet(
        allowed_contacts=AllowedContactPolicy(rules, penetration_tolerance=0.012),
        held_objects=tuple(attachments),
        position_tolerance=0.06,
        orientation_tolerance=0.35,
    )


class TransportSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.TRANSPORT:
            raise ValueError("TransportSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects or not state.objects[object_name].held_by:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"{object_name!r} must be held before transport", [],
            )
        target = self._semantic_target(request, state)
        if target is None:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"unknown symbolic transport goal {request.goal!r}", [],
            )
        return self.to_pose(request, target)

    def to_pose(self, request: SkillRequest, target: Pose) -> SkillResult:
        state = collect_scene_state(self.env)
        object_name = request.object_name
        holders = state.objects[object_name].held_by
        attachments = tuple(
            self.context.attachments.get((object_name, side))
            or capture_attachment(state, object_name, side)
            for side in holders
        )
        constraints = held_constraints(
            object_name, attachments,
            extra_rules=self.context.carry_contact_rules.get(object_name, ()),
        )
        execution, ik_result, path_result = self.move(
            name=f"transport_{object_name}",
            targets=ee_targets_for_object_pose(target, attachments),
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
            ) or FailureCode.PATH_BLOCKED
            return self.failure(
                request, collect_scene_state(self.env), code,
                f"transport failed: {code.value}", reports,
            )
        final = collect_scene_state(self.env)
        error = float(np.linalg.norm(
            np.asarray(final.objects[object_name].pose.position)
            - np.asarray(target.position)
        ))
        if error > 0.07 or any(side not in final.objects[object_name].held_by for side in holders):
            return self.failure(
                request, final, FailureCode.POSTCONDITION_FAILED,
                "transport target or hold postcondition failed", reports,
                {"position_error": error},
            )
        for side in holders:
            self.context.attachments[(object_name, side)] = capture_attachment(final, object_name, side)
        return SkillResult(
            request, True, f"transported {object_name}", final,
            execution_reports=tuple(reports), diagnostics={"position_error": error},
        )

    @staticmethod
    def _semantic_target(request: SkillRequest, state) -> Pose | None:
        obj = state.objects[request.object_name]
        goal = (request.goal or request.strategy or "").lower()
        if goal in {"handover_region", "shared_workspace", "center"}:
            return Pose((0.55, 0.0, max(obj.pose.position[2], 1.08)), obj.pose.quaternion_wxyz)
        if goal in {"clear_table", "lift_clearance"}:
            return Pose(
                (obj.pose.position[0], obj.pose.position[1], obj.pose.position[2] + 0.08),
                obj.pose.quaternion_wxyz,
            )
        if goal.startswith("above:"):
            support_name = goal.split(":", 1)[1]
            if support_name not in state.objects:
                return None
            support = state.objects[support_name]
            return Pose(
                (
                    support.pose.position[0],
                    support.pose.position[1],
                    support.pose.position[2]
                    + 0.5 * (support.size[2] + obj.size[2]) + 0.08,
                ),
                obj.pose.quaternion_wxyz,
            )
        return None
