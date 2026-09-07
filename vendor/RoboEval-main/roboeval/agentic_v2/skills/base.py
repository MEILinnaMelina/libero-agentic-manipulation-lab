"""Shared deterministic machinery behind semantic Agentic v2 skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from roboeval.agentic_v2.executor import MonitoredExecutor
from roboeval.agentic_v2.motion.candidate_generator import CandidateGenerator
from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.motion.ik import IKSearchResult, MultiStartIK
from roboeval.agentic_v2.motion.path_planner import JointPathPlanner, PathSearchResult
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    AllowedContactRule,
    ConstraintSet,
    ExecutionReport,
    FailureCode,
    HeldObjectAttachment,
    Pose,
    SkillRequest,
    SkillResult,
    to_jsonable,
)


@dataclass
class SkillContext:
    env: Any
    checker: CollisionChecker
    ik: MultiStartIK
    planner: JointPathPlanner
    executor: MonitoredExecutor
    candidates: CandidateGenerator
    attachments: dict[tuple[str, str], HeldObjectAttachment] = field(default_factory=dict)
    planning_trace: list[dict[str, Any]] = field(default_factory=list)
    # Support-surface height (world z of the object's bottom face) observed
    # while each object was last at rest, keyed by object name. Lets a
    # staged handover put an object back down at a known-good height even
    # in scenes with no other resting object to reference.
    resting_surfaces: dict[str, float] = field(default_factory=dict)
    # Orientation (wxyz) each object had while last at rest, so a placement
    # can set it back down the way it lay even if the carry reoriented it.
    resting_orientations: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    # Whether edge grasps are settled into a hanging carry after
    # verification (see GraspSkill._consolidate_edge_grasp); off only for
    # experiments.
    consolidate_edge_grasps: bool = True
    # How each held object is currently carried ("edge:<depth>" for a flat
    # object pinched at one edge) and, for edge carries, how far in from
    # that edge the pads sit - placement uses it to leave exactly that much
    # of the object overhanging the support so the lower finger clears it.
    carry_modes: dict[str, str] = field(default_factory=dict)
    edge_pinch_depths: dict[str, float] = field(default_factory=dict)
    # Extra contact rules that hold for as long as an object is carried a
    # particular way (an edge-pinched book butts against the palm, which
    # is what makes the pinch rigid - that palm contact must stay allowed
    # through lifts, transports and placements).
    carry_contact_rules: dict[str, tuple[AllowedContactRule, ...]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        env: Any,
        *,
        render: bool = False,
        frame_callback: Any | None = None,
        feasibility_gate: str = "full",
    ) -> "SkillContext":
        if feasibility_gate not in {"full", "ik-only"}:
            raise ValueError(f"unknown feasibility gate {feasibility_gate!r}")
        checker = CollisionChecker(
            env,
            enforce_contacts=feasibility_gate == "full",
        )
        executor = MonitoredExecutor(
            env,
            collision_checker=checker,
            render=render,
            frame_callback=frame_callback,
        )
        ik = MultiStartIK(env, checker)
        ik.commanded_joints = lambda: executor.adapter.last_commanded
        return cls(
            env=env,
            checker=checker,
            ik=ik,
            planner=JointPathPlanner(env, checker),
            executor=executor,
            candidates=CandidateGenerator(env),
        )


class BaseSkill:
    def __init__(self, context: SkillContext) -> None:
        self.context = context

    @property
    def env(self) -> Any:
        return self.context.env

    def move(
        self,
        *,
        name: str,
        targets: Mapping[str, Pose],
        constraints: ConstraintSet | None = None,
        protected_objects: Mapping[str, Pose] | None = None,
        require_holds: bool = True,
        candidate_count: int = 7,
        stop_condition: Callable[[Any], bool] | None = None,
        terminal_constraints: ConstraintSet | None = None,
        stop_on_success: bool = True,
    ) -> tuple[ExecutionReport | None, IKSearchResult, PathSearchResult | None]:
        started = perf_counter()
        constraints = constraints or ConstraintSet()
        state = collect_scene_state(self.env)
        ik_result = self.context.ik.solve_candidates(
            targets,
            count=candidate_count,
            constraints=constraints,
        )
        if not ik_result.report.feasible:
            self._record_motion_trace(
                name, targets, constraints, ik_result, None, None, started
            )
            return None, ik_result, None
        path_result = self.context.planner.plan_to_candidates(
            state.robot.joint_positions,
            ik_result.accepted,
            constraints=constraints,
            name=name,
        )
        if path_result.plan is None:
            self._record_motion_trace(
                name, targets, constraints, ik_result, path_result, None, started
            )
            return None, ik_result, path_result
        execution = self.context.executor.execute(
            path_result.plan,
            protected_objects=protected_objects,
            require_holds=require_holds,
            stop_condition=stop_condition,
            terminal_constraints=terminal_constraints,
            stop_on_success=stop_on_success,
        )
        self._record_motion_trace(
            name, targets, constraints, ik_result, path_result, execution, started
        )
        return execution, ik_result, path_result

    def _record_motion_trace(
        self,
        name: str,
        targets: Mapping[str, Pose],
        constraints: ConstraintSet,
        ik_result: IKSearchResult,
        path_result: PathSearchResult | None,
        execution: ExecutionReport | None,
        started: float,
    ) -> None:
        plan = path_result.plan if path_result is not None else None
        self.context.planning_trace.append(
            {
                "name": name,
                "targets": to_jsonable(targets),
                "constraints": to_jsonable(constraints),
                "ik": {
                    "report": to_jsonable(ik_result.report),
                    "accepted": to_jsonable(ik_result.accepted),
                    "rejected": to_jsonable(ik_result.rejected),
                },
                "path": {
                    "report": to_jsonable(path_result.report),
                    "attempted_paths": path_result.attempted_paths,
                    "plan": to_jsonable(plan),
                }
                if path_result is not None
                else None,
                "execution": to_jsonable(execution) if execution is not None else None,
                "elapsed_seconds": perf_counter() - started,
            }
        )

    def nudge_clear(self, side: str, *, distance: float = 0.03) -> ExecutionReport | None:
        """Lift one arm a little while tolerating whatever incidental contacts
        it is currently in (an idle hand resting a fingertip on a wheel, a
        finger grazing the table), so later motions start contact-free.
        Does nothing when the arm is already clear."""

        live = self.context.checker.check_live_contacts(ConstraintSet())
        offending = [contact for contact in live.contacts if not contact.allowed]
        if not offending:
            return None
        rules = tuple(
            AllowedContactRule(contact.first, contact.second, penetration_tolerance=0.012)
            for contact in offending
        )
        policy = AllowedContactPolicy(rules=rules, penetration_tolerance=0.012)
        state = collect_scene_state(self.env)
        current = state.robot.arms[side].ee_pose
        target = Pose(
            (current.position[0], current.position[1], current.position[2] + distance),
            current.quaternion_wxyz,
        )
        execution, _, _ = self.move(
            name=f"nudge_clear_{side}",
            targets={side: target},
            constraints=ConstraintSet(allowed_contacts=policy),
            require_holds=False,
        )
        return execution

    def retreat(
        self,
        side: str,
        *,
        distance: float = 0.08,
        constraints: ConstraintSet | None = None,
    ) -> ExecutionReport | None:
        state = collect_scene_state(self.env)
        current = state.robot.arms[side].ee_pose
        target = Pose(
            (
                current.position[0],
                current.position[1],
                current.position[2] + distance,
            ),
            current.quaternion_wxyz,
        )
        execution, _, _ = self.move(
            name=f"checked_retreat_{side}",
            targets={side: target},
            constraints=constraints,
            require_holds=False,
        )
        return execution

    def park_idle_arm(self, side: str, *, distance: float = 0.05) -> ExecutionReport | None:
        """Raise the arm that will stay idle during a single-arm skill. Idle
        hands start a centimeter or so above scene parts in several tasks
        (valve wheels, box flaps) and settle onto them while the other arm
        works, which then vetoes every plan for the working arm."""

        execution = self.retreat(side, distance=distance)
        if execution is not None and execution.success:
            return execution
        return self.nudge_clear(side, distance=distance)

    @staticmethod
    def failure(
        request: SkillRequest,
        state: Any,
        code: FailureCode,
        message: str,
        reports: list[ExecutionReport],
        diagnostics: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(
            request=request,
            success=False,
            message=message,
            state=state,
            failure_code=code,
            execution_reports=tuple(reports),
            diagnostics=diagnostics or {},
        )
