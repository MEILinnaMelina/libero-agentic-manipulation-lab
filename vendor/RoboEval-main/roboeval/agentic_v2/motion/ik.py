"""Deterministic multi-start IK with explicit convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import warnings

from dm_control.utils.inverse_kinematics import qpos_from_site_pose
import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.types import (
    ConstraintSet,
    FailureCode,
    FeasibilityReport,
    IKCandidate,
    Pose,
)


@dataclass(frozen=True)
class IKSearchResult:
    accepted: tuple[IKCandidate, ...]
    rejected: tuple[IKCandidate, ...]
    report: FeasibilityReport


class MultiStartIK:
    """Wrap RoboEval's dm_control IK while retaining the evidence v1 drops."""

    def __init__(
        self,
        env: Any,
        collision_checker: CollisionChecker | None = None,
        *,
        random_seed: int = 20260904,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.12,
        maximum_steps: int | None = None,
    ) -> None:
        self.env = env
        self.solver = env.inverse_kinematics
        self.physics = self.solver._physics
        self.arm_joints = self.solver._arm_joints
        self.arm_sites = {
            "left": self.solver._arm_sites[0],
            "right": self.solver._arm_sites[1],
        }
        self.joint_names = {
            side: [joint.name for joint in self.arm_joints if side in joint.name.lower()]
            for side in ("left", "right")
        }
        self.checker = collision_checker or CollisionChecker(env)
        self.random_seed = int(random_seed)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.maximum_steps = int(
            maximum_steps or self.solver._config.solver_max_steps
        )
        self.lower = self.checker.lower
        self.upper = self.checker.upper
        # Optional source of the last *commanded* arm joints; idle-side
        # joints are seeded from it so the idle arm is re-commanded to where
        # it was told to be, not to the slightly sagged pose it measures at.
        self.commanded_joints: Callable[[], np.ndarray | None] | None = None

    def _root_pose(self) -> Pose:
        pelvis = self.env.robot.pelvis
        if pelvis is None:
            return Pose.identity()
        return Pose(tuple(pelvis.get_position()), tuple(pelvis.get_quaternion()))

    def _to_solver_frame(self, target: Pose) -> Pose:
        return self._root_pose().inverse().compose(target)

    def _seeds(
        self,
        count: int,
        active_sides: set[str],
    ) -> list[tuple[str, np.ndarray]]:
        if count < 1:
            raise ValueError("IK candidate count must be positive")
        current = np.asarray(self.env.robot.qpos_actuated[:-len(self.env.robot.grippers)], dtype=float)
        initial = np.asarray(self.env.robot.get_initial_qpos()[:-len(self.env.robot.grippers)], dtype=float)
        midpoint = (self.lower + self.upper) / 2.0
        active = np.zeros_like(current, dtype=bool)
        for side in active_sides:
            start = 0 if side == "left" else len(current) // 2
            active[start : start + len(current) // 2] = True
        commanded = self.commanded_joints() if self.commanded_joints is not None else None
        if commanded is not None and np.shape(commanded) == current.shape:
            commanded = np.asarray(commanded, dtype=float)
            if np.all(commanded >= self.lower) and np.all(commanded <= self.upper):
                current = current.copy()
                current[~active] = commanded[~active]
        initial_seed = current.copy()
        initial_seed[active] = initial[active]
        midpoint_seed = current.copy()
        midpoint_seed[active] = midpoint[active]
        seeds: list[tuple[str, np.ndarray]] = [
            ("current", current.copy()),
            ("initial", initial_seed),
            ("joint_midpoint", midpoint_seed),
        ]
        rng = np.random.default_rng(self.random_seed)
        span = self.upper - self.lower
        while len(seeds) < count:
            perturbation = rng.normal(0.0, 0.12, size=current.shape) * span
            perturbation[~active] = 0.0
            seeds.append(
                (
                    f"perturbation_{len(seeds) - 2}",
                    np.clip(current + perturbation, self.lower, self.upper),
                )
            )
        return seeds[:count]

    def solve_candidates(
        self,
        targets: Mapping[str, Pose],
        *,
        count: int = 7,
        constraints: ConstraintSet | None = None,
    ) -> IKSearchResult:
        if not targets or set(targets) - {"left", "right"}:
            raise ValueError("targets must contain left and/or right poses")
        constraints = constraints or ConstraintSet()
        accepted: list[IKCandidate] = []
        rejected: list[IKCandidate] = []
        for seed_name, seed in self._seeds(count, set(targets)):
            candidate = self._solve_seed(seed_name, seed, targets, constraints)
            if candidate.feasible and not any(
                np.linalg.norm(
                    np.asarray(candidate.joint_positions) - np.asarray(existing.joint_positions)
                ) < 1e-3
                for existing in accepted
            ):
                accepted.append(candidate)
            else:
                rejected.append(candidate)
        accepted.sort(key=lambda candidate: candidate.score)
        if accepted:
            report = FeasibilityReport(
                True,
                message=f"{len(accepted)} feasible IK candidate(s)",
                diagnostics={"rejected": len(rejected)},
            )
        else:
            code = self._dominant_failure(rejected)
            report = FeasibilityReport(
                False,
                code,
                f"no feasible IK candidate from {count} seed(s)",
                diagnostics={"failures": [item.failure_code.value for item in rejected if item.failure_code]},
            )
        return IKSearchResult(tuple(accepted), tuple(rejected), report)

    def _solve_seed(
        self,
        seed_name: str,
        seed: np.ndarray,
        targets: Mapping[str, Pose],
        constraints: ConstraintSet,
    ) -> IKCandidate:
        bound_joints = self.physics.bind(self.arm_joints)
        original = (
            np.asarray(bound_joints.qpos).copy(),
            np.asarray(bound_joints.qvel).copy(),
            np.asarray(bound_joints.qacc).copy(),
        )
        solver_targets = {side: self._to_solver_frame(pose) for side, pose in targets.items()}
        results: dict[str, Any] = {}
        try:
            bound_joints.qpos = seed
            bound_joints.qvel = np.zeros_like(seed)
            bound_joints.qacc = np.zeros_like(seed)
            self.physics.forward()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for side, target in solver_targets.items():
                    results[side] = qpos_from_site_pose(
                        self.physics,
                        self.arm_sites[side].name,
                        target_pos=target.position,
                        target_quat=target.quaternion_wxyz,
                        joint_names=self.joint_names[side],
                        tol=min(self.position_tolerance, self.orientation_tolerance),
                        max_steps=self.maximum_steps,
                        inplace=True,
                    )
            solution = np.asarray(bound_joints.qpos, dtype=float).copy()
            errors = {side: self._pose_errors(side, target) for side, target in solver_targets.items()}
        finally:
            bound_joints.qpos, bound_joints.qvel, bound_joints.qacc = original
            self.physics.forward()
        return self._candidate_from_results(
            seed_name,
            solution,
            errors,
            results,
            constraints,
        )

    def _candidate_from_results(
        self,
        seed_name: str,
        solution: np.ndarray,
        errors: Mapping[str, tuple[float, float]],
        results: Mapping[str, Any],
        constraints: ConstraintSet,
    ) -> IKCandidate:
        position_error = max(value[0] for value in errors.values())
        orientation_error = max(value[1] for value in errors.values())
        iterations = sum(int(result.steps) for result in results.values())
        diagnostics = {
            "per_side": {
                side: {
                    "position_error": errors[side][0],
                    "orientation_error": errors[side][1],
                    "solver_error_norm": float(result.err_norm),
                    "solver_success": bool(result.success),
                    "steps": int(result.steps),
                }
                for side, result in results.items()
            }
        }
        converged = (
            all(bool(result.success) for result in results.values())
            and position_error <= self.position_tolerance
            and orientation_error <= self.orientation_tolerance
        )
        if not converged:
            return IKCandidate(
                seed_name, tuple(solution), False, position_error,
                orientation_error, iterations,
                failure_code=FailureCode.IK_UNREACHABLE,
                diagnostics=diagnostics,
            )
        return self._check_solution(
            seed_name, solution, position_error, orientation_error,
            iterations, constraints, diagnostics,
        )

    def _check_solution(
        self,
        seed_name: str,
        solution: np.ndarray,
        position_error: float,
        orientation_error: float,
        iterations: int,
        constraints: ConstraintSet,
        diagnostics: dict[str, Any],
    ) -> IKCandidate:
        outside = np.flatnonzero((solution < self.lower) | (solution > self.upper))
        if outside.size:
            return IKCandidate(
                seed_name, tuple(solution), True, position_error,
                orientation_error, iterations,
                failure_code=FailureCode.JOINT_LIMIT,
                diagnostics={**diagnostics, "limit_indices": outside.tolist()},
            )
        collision = self.checker.check(solution, constraints)
        if not collision.feasible:
            return IKCandidate(
                seed_name, tuple(solution), True, position_error,
                orientation_error, iterations,
                failure_code=collision.failure_code,
                diagnostics={
                    **diagnostics,
                    "collision_message": collision.message,
                    "contacts": [
                        {
                            "geom1_name": contact.geom1_name,
                            "geom2_name": contact.geom2_name,
                            "first": contact.first,
                            "second": contact.second,
                            "distance": contact.distance,
                            "kind": contact.kind.value,
                            "allowed": contact.allowed,
                        }
                        for contact in collision.contacts
                    ],
                },
            )
        current = np.asarray(
            self.env.robot.qpos_actuated[:-len(self.env.robot.grippers)],
            dtype=float,
        )
        span = self.upper - self.lower
        normalized_distance = float(np.linalg.norm((solution - current) / span))
        normalized_margin = np.minimum(
            (solution - self.lower) / span,
            (self.upper - solution) / span,
        )
        score = (
            10.0 * position_error
            + orientation_error
            + normalized_distance
            + 0.1 * (1.0 - float(np.min(normalized_margin)))
        )
        return IKCandidate(
            seed_name, tuple(solution), True, position_error,
            orientation_error, iterations, score=score,
            diagnostics=diagnostics,
        )

    def _pose_errors(self, side: str, target: Pose) -> tuple[float, float]:
        bound = self.physics.bind(self.arm_sites[side])
        position_error = float(
            np.linalg.norm(np.asarray(bound.xpos) - np.asarray(target.position))
        )
        current_rotation = Rotation.from_matrix(np.asarray(bound.xmat).reshape(3, 3))
        target_rotation = Rotation.from_matrix(target.as_matrix()[:3, :3])
        orientation_error = float((current_rotation.inv() * target_rotation).magnitude())
        return position_error, orientation_error

    @staticmethod
    def _dominant_failure(rejected: list[IKCandidate]) -> FailureCode:
        failures = {candidate.failure_code for candidate in rejected}
        for code in (
            FailureCode.SELF_COLLISION,
            FailureCode.HELD_OBJECT_COLLISION,
            FailureCode.ENV_COLLISION,
            FailureCode.JOINT_LIMIT,
            FailureCode.IK_UNREACHABLE,
        ):
            if code in failures:
                return code
        return FailureCode.IK_UNREACHABLE
