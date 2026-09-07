"""Verified handwheel rotation: grip the wheel from above, twist the wrist."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.executor import CLOSE_COMMAND, OPEN_COMMAND
from roboeval.agentic_v2.skills.base import BaseSkill
from roboeval.agentic_v2.state import collect_scene_state
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


# RoboEval's RotateValve success is get_state() > 0.10 (normalized by the
# joint range, i.e. ~0.47 rad); aim past it with margin.
TARGET_NORMALIZED_STATE = 0.10
TARGET_MARGIN = 0.06
# Wrist yaw per twist step about world +Z (counterclockwise seen from
# above, the positive direction of the valve's vertical revolute joint).
TWIST_STEP = 0.35
MAXIMUM_TWISTS = 4
# If the wrist runs out of travel, let go, wind the hand back, re-grip and
# keep turning (a ratchet) - at most this many grips.
MAXIMUM_GRIPS = 3


class RotateSkill(BaseSkill):
    def execute(self, request: SkillRequest) -> SkillResult:
        if request.skill is not SkillName.ROTATE:
            raise ValueError("RotateSkill received the wrong request")
        state = collect_scene_state(self.env)
        object_name = request.object_name
        if object_name not in state.objects:
            return self.failure(
                request, state, FailureCode.PRECONDITION_FAILED,
                f"object {object_name!r} is unavailable", [],
            )
        valve = self._valve(object_name)
        if valve is None:
            return self.failure(
                request, state, FailureCode.INVALID_REQUEST,
                f"{object_name} is not a rotatable valve", [],
            )
        if self._normalized_state(valve) > TARGET_NORMALIZED_STATE:
            return SkillResult(request, True, f"{object_name} is already turned past the threshold", state)
        side = self._select_side(request, state)
        candidates = self.context.candidates.grasp_candidates(object_name, side, state)
        if not candidates:
            return self.failure(
                request, state, FailureCode.NO_VALID_GRASP,
                f"no aperture-compatible wheel grasp for {object_name}", [],
            )
        reports = []
        attempts: list[dict[str, Any]] = []
        # Both hands start with their fingertips a centimeter above the two
        # wheels and the idle one settles onto its wheel while the other
        # works (observed: left-finger/valve_1 contact appearing mid-approach
        # of the right arm), vetoing every plan. Park the idle arm higher.
        idle = "left" if side == "right" else "right"
        parked = self.park_idle_arm(idle)
        if parked is not None:
            reports.append(parked)
        opened = self.context.executor.command_gripper(
            side, OPEN_COMMAND,
            constraints=ConstraintSet(allowed_contacts=candidates[0].contact_policy),
        )
        reports.append(opened)
        if not opened.success:
            return self.failure(
                request, opened.final_state,
                opened.failure_code or FailureCode.EXECUTION_DIVERGED,
                "failed to open before approach", reports,
            )
        for candidate in candidates:
            result, evidence = self._attempt(request, candidate, valve, reports)
            attempts.append(evidence)
            if result is not None:
                return result
        return self.failure(
            request, collect_scene_state(self.env), FailureCode.NO_VALID_GRASP,
            f"all {len(candidates)} wheel grasp candidates failed", reports,
            {"attempts": attempts},
        )

    def _attempt(self, request, candidate, valve, reports):
        object_name = candidate.object_name
        side = candidate.side
        pre, pre_ik, pre_path = self.move(
            name=f"{candidate.name}_pregrasp",
            targets={side: candidate.pregrasp_pose},
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "pregrasp", pre, pre_ik, pre_path)
        if pre is None or not pre.success:
            if pre is not None:
                reports.append(pre)
                self._recover(reports, side)
            return None, evidence
        reports.append(pre)
        approach_constraints = ConstraintSet(allowed_contacts=candidate.contact_policy)
        approach, approach_ik, approach_path = self.move(
            name=f"{candidate.name}_approach",
            targets={side: candidate.grasp_pose},
            constraints=approach_constraints,
            require_holds=False,
        )
        evidence = self._evidence(candidate.name, "approach", approach, approach_ik, approach_path)
        if approach is None or not approach.success:
            if approach is not None:
                reports.append(approach)
            self._recover(reports, side)
            return None, evidence
        reports.append(approach)
        closed = self.context.executor.command_gripper(
            side, CLOSE_COMMAND, steps=16, constraints=approach_constraints,
        )
        reports.append(closed)
        grasped = collect_scene_state(self.env)
        if not closed.success or side not in grasped.objects[object_name].held_by:
            self._recover(reports, side, open_first=True, constraints=approach_constraints)
            evidence.update({"stage": "close", "failure": FailureCode.GRASP_FAILED.value})
            return None, evidence

        # Twist: rotate the wrist about world Z at its current position. The
        # wheel is a revolute part of a 250 kg base, so the object's root
        # pose never moves - hold monitoring would misread that as slip, so
        # the twist runs with require_holds off and verifies the joint state
        # directly instead.
        twist_policy = AllowedContactPolicy(
            rules=candidate.contact_policy.rules,
            penetration_tolerance=0.02,
        )
        twist_constraints = ConstraintSet(allowed_contacts=twist_policy)
        before = self._normalized_state(valve)
        twists = 0
        grips = 0
        goal = TARGET_NORMALIZED_STATE + TARGET_MARGIN
        grip_pose = collect_scene_state(self.env).robot.arms[side].ee_pose
        while grips < MAXIMUM_GRIPS and self._normalized_state(valve) < goal:
            grips += 1
            stalled = False
            for twist in range(MAXIMUM_TWISTS):
                if self._normalized_state(valve) >= goal:
                    break
                current = collect_scene_state(self.env).robot.arms[side].ee_pose
                rotation = Rotation.from_euler("z", TWIST_STEP) * Rotation.from_matrix(current.as_matrix()[:3, :3])
                xyzw = rotation.as_quat()
                target = Pose(current.position, (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))
                execution, twist_ik, twist_path = self.move(
                    name=f"{candidate.name}_grip{grips}_twist_{twist}",
                    targets={side: target},
                    constraints=twist_constraints,
                    require_holds=False,
                    candidate_count=5,
                    stop_on_success=False,
                )
                if execution is not None:
                    reports.append(execution)
                twists += 1
                if execution is None or not execution.success:
                    evidence.update(self._evidence(candidate.name, f"grip{grips}_twist_{twist}", execution, twist_ik, twist_path))
                    stalled = True
                    break
            if self._normalized_state(valve) >= goal or not stalled:
                break
            # Ratchet: let go, wind the hand back to the original grip
            # orientation (unloaded), come back down and grip again.
            if not self._regrip(side, grip_pose, approach_constraints, twist_constraints, reports):
                break
        after = self._normalized_state(valve)
        released = self.context.executor.command_gripper(side, OPEN_COMMAND, constraints=twist_constraints)
        reports.append(released)
        retreat = self.retreat(side, constraints=twist_constraints)
        if retreat is not None:
            reports.append(retreat)
        final = collect_scene_state(self.env)
        evidence.update({"state_before": before, "state_after": after, "twists": twists})
        if after <= TARGET_NORMALIZED_STATE:
            evidence.update({"stage": "twist", "failure": FailureCode.POSTCONDITION_FAILED.value})
            return None, evidence
        return SkillResult(
            request, True,
            f"turned {object_name} with {side} from {before:.3f} to {after:.3f}",
            final,
            execution_reports=tuple(reports),
            diagnostics={"candidate": candidate.name, "state_before": before, "state_after": after, "twists": twists},
        ), evidence

    def _regrip(self, side, grip_pose, approach_constraints, twist_constraints, reports) -> bool:
        opened = self.context.executor.command_gripper(side, OPEN_COMMAND, constraints=twist_constraints)
        reports.append(opened)
        if not opened.success:
            return False
        lifted = self.retreat(side, distance=0.04, constraints=twist_constraints)
        if lifted is not None:
            reports.append(lifted)
        if lifted is None or not lifted.success:
            return False
        above = Pose(
            (grip_pose.position[0], grip_pose.position[1], grip_pose.position[2] + 0.04),
            grip_pose.quaternion_wxyz,
        )
        rewound, _, _ = self.move(
            name=f"rotate_rewind_{side}", targets={side: above},
            constraints=approach_constraints, require_holds=False,
        )
        if rewound is not None:
            reports.append(rewound)
        if rewound is None or not rewound.success:
            return False
        descended, _, _ = self.move(
            name=f"rotate_regrip_{side}", targets={side: grip_pose},
            constraints=approach_constraints, require_holds=False,
        )
        if descended is not None:
            reports.append(descended)
        if descended is None or not descended.success:
            return False
        closed = self.context.executor.command_gripper(
            side, CLOSE_COMMAND, steps=16, constraints=approach_constraints,
        )
        reports.append(closed)
        return bool(closed.success)

    def _recover(self, reports, side, *, open_first=False, constraints=None) -> None:
        if open_first:
            opened = self.context.executor.command_gripper(side, OPEN_COMMAND, constraints=constraints)
            reports.append(opened)
            if not opened.success:
                return
        retreat = self.retreat(side)
        if retreat is not None:
            reports.append(retreat)

    def _valve(self, object_name: str):
        valves = getattr(self.env, "valves", None)
        if not valves or not object_name.startswith("valve_"):
            return None
        try:
            return valves[int(object_name.split("_", 1)[1])]
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _normalized_state(valve) -> float:
        state = np.asarray(valve.get_state(), dtype=float).reshape(-1)
        return float(state[0]) if state.size else 0.0

    @staticmethod
    def _select_side(request: SkillRequest, state) -> str:
        direct = [side for side in ("left", "right") if side in request.roles]
        if len(direct) == 1:
            return direct[0]
        if request.strategy in ("left", "right"):
            return request.strategy
        obj = state.objects[request.object_name]
        return min(
            ("left", "right"),
            key=lambda side: sum(
                (state.robot.arms[side].ee_pose.position[index] - obj.aabb_center[index]) ** 2
                for index in range(2)
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
