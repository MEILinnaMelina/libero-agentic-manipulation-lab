"""Simple, fully checked joint-space planning for Agentic v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.types import (
    ConstraintSet,
    FailureCode,
    FeasibilityReport,
    IKCandidate,
    MotionPlan,
    TrajectoryPoint,
)


@dataclass(frozen=True)
class PathSearchResult:
    plan: MotionPlan | None
    report: FeasibilityReport
    attempted_paths: int


class JointPathPlanner:
    """Try straight joint paths, alternate IK goals, then checked waypoints."""

    def __init__(
        self,
        env: Any,
        collision_checker: CollisionChecker | None = None,
        *,
        maximum_joint_step: float = 0.04,
        maximum_velocity: float = 0.8,
        maximum_acceleration: float = 2.0,
    ) -> None:
        self.env = env
        self.checker = collision_checker or CollisionChecker(env)
        self.maximum_joint_step = float(maximum_joint_step)
        self.maximum_velocity = float(maximum_velocity)
        self.maximum_acceleration = float(maximum_acceleration)
        self.control_dt = 1.0 / float(env.control_frequency)
        self.initial = np.asarray(
            env.robot.get_initial_qpos()[:-len(env.robot.grippers)],
            dtype=float,
        )

    def plan_to_candidates(
        self,
        start: Sequence[float],
        candidates: Sequence[IKCandidate],
        *,
        constraints: ConstraintSet | None = None,
        name: str = "joint_path",
    ) -> PathSearchResult:
        constraints = constraints or ConstraintSet()
        goals = [np.asarray(candidate.joint_positions, dtype=float) for candidate in candidates if candidate.feasible]
        if not goals:
            return PathSearchResult(
                None,
                FeasibilityReport(False, FailureCode.IK_UNREACHABLE, "no feasible IK goals"),
                0,
            )
        start_array = np.asarray(start, dtype=float)
        attempted = 0
        failures: list[tuple[str, int, FeasibilityReport]] = []
        for goal_index, goal in enumerate(goals):
            attempted += 1
            points, failure = self._checked_segment(start_array, goal, constraints)
            if failure is None:
                return self._success_plan(name, points, constraints, attempted, goal_index, "straight")
            failures.append(("straight", goal_index, failure))

        for waypoint_name, waypoint in self._recovery_waypoints(start_array):
            for goal_index, goal in enumerate(goals):
                attempted += 1
                first, first_failure = self._checked_segment(start_array, waypoint, constraints)
                if first_failure is not None:
                    failures.append((waypoint_name, goal_index, first_failure))
                    continue
                second, second_failure = self._checked_segment(waypoint, goal, constraints)
                if second_failure is not None:
                    failures.append((waypoint_name, goal_index, second_failure))
                    continue
                combined = first + second[1:]
                return self._success_plan(
                    name, combined, constraints, attempted, goal_index, waypoint_name
                )
        last_failure = failures[-1][2] if failures else FeasibilityReport(False)
        return PathSearchResult(
            None,
            FeasibilityReport(
                False,
                FailureCode.PATH_BLOCKED,
                f"all {attempted} candidate path(s) blocked",
                contacts=last_failure.contacts,
                diagnostics={
                    "attempts": [
                        {
                            "route": route,
                            "goal_index": index,
                            "code": report.failure_code.value if report.failure_code else None,
                            "sample": report.diagnostics.get("sample"),
                        }
                        for route, index, report in failures
                    ]
                },
            ),
            attempted,
        )

    def _checked_segment(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        constraints: ConstraintSet,
    ) -> tuple[list[np.ndarray], FeasibilityReport | None]:
        if start.shape != goal.shape or start.shape != (len(self.checker.qpos_addresses),):
            raise ValueError("joint path endpoints have the wrong shape")
        displacement = np.abs(goal - start)
        maximum_displacement = float(np.max(displacement))
        duration_velocity = 1.5 * maximum_displacement / self.maximum_velocity
        duration_acceleration = np.sqrt(
            6.0 * maximum_displacement / self.maximum_acceleration
        ) if maximum_displacement else 0.0
        duration = max(self.control_dt, duration_velocity, duration_acceleration)
        count = max(
            2,
            int(np.ceil(1.5 * maximum_displacement / self.maximum_joint_step)) + 1,
            int(np.ceil(duration / self.control_dt)) + 1,
        )
        result: list[np.ndarray] = []
        for index, unit in enumerate(np.linspace(0.0, 1.0, count)):
            alpha = 3.0 * unit * unit - 2.0 * unit * unit * unit
            point = start + alpha * (goal - start)
            feasibility = self.checker.check(point, constraints)
            if not feasibility.feasible:
                return result, FeasibilityReport(
                    False,
                    feasibility.failure_code,
                    feasibility.message,
                    contacts=feasibility.contacts,
                    diagnostics={
                        **dict(feasibility.diagnostics),
                        "sample": index,
                        "sample_count": count,
                        "fraction": float(unit),
                    },
                )
            result.append(point)
        return result, None

    def _recovery_waypoints(self, start: np.ndarray) -> list[tuple[str, np.ndarray]]:
        waypoints: list[tuple[str, np.ndarray]] = []
        if np.linalg.norm(start - self.initial) > 1e-4:
            waypoints.append(("initial", self.initial.copy()))
        outward = self.initial.copy()
        outward[0] = np.clip(-0.45, self.checker.lower[0], self.checker.upper[0])
        outward[7] = np.clip(0.45, self.checker.lower[7], self.checker.upper[7])
        outward[1] = np.clip(-0.25, self.checker.lower[1], self.checker.upper[1])
        outward[8] = np.clip(-0.25, self.checker.lower[8], self.checker.upper[8])
        waypoints.append(("arms_outward", outward))
        return waypoints

    def _success_plan(
        self,
        name: str,
        positions: list[np.ndarray],
        constraints: ConstraintSet,
        attempted: int,
        goal_index: int,
        route: str,
    ) -> PathSearchResult:
        points = tuple(
            TrajectoryPoint(tuple(position), index * self.control_dt)
            for index, position in enumerate(positions)
        )
        path_length = float(
            sum(
                np.linalg.norm(np.asarray(second.joint_positions) - np.asarray(first.joint_positions))
                for first, second in zip(points, points[1:])
            )
        )
        plan = MotionPlan(
            name=name,
            points=points,
            constraints=constraints,
            score=path_length,
            diagnostics={
                "route": route,
                "goal_index": goal_index,
                "attempted_paths": attempted,
                "path_length": path_length,
                **self.dynamics(points),
            },
        )
        return PathSearchResult(
            plan,
            FeasibilityReport(True, message=f"selected {route} route"),
            attempted,
        )

    @staticmethod
    def dynamics(points: Sequence[TrajectoryPoint]) -> dict[str, float]:
        if len(points) < 2:
            return {"maximum_velocity": 0.0, "maximum_acceleration": 0.0}
        positions = np.asarray([point.joint_positions for point in points])
        times = np.asarray([point.time_from_start for point in points])
        dt = np.diff(times)
        velocity = np.diff(positions, axis=0) / dt[:, None]
        maximum_velocity = float(np.max(np.abs(velocity)))
        if len(velocity) < 2:
            maximum_acceleration = 0.0
        else:
            acceleration_dt = (dt[:-1] + dt[1:]) / 2.0
            acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
            maximum_acceleration = float(np.max(np.abs(acceleration)))
        return {
            "maximum_velocity": maximum_velocity,
            "maximum_acceleration": maximum_acceleration,
        }
