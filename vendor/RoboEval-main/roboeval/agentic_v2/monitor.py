"""Execution invariants for bounded Agentic v2 trajectory execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.motion.collision_checker import CollisionChecker
from roboeval.agentic_v2.types import (
    ConstraintSet,
    FailureCode,
    MonitorEvent,
    Pose,
    SceneState,
)


@dataclass(frozen=True)
class MonitorConfig:
    tracking_tolerance: float = 0.30
    tracking_patience: int = 5
    maximum_joint_velocity: float = 8.0
    velocity_patience: int = 3
    protected_object_tolerance: float = 0.025
    hold_patience: int = 3


class ExecutionMonitor:
    def __init__(
        self,
        collision_checker: CollisionChecker,
        config: MonitorConfig | None = None,
    ) -> None:
        self.checker = collision_checker
        self.config = config or MonitorConfig()
        self.reset()

    def reset(self) -> None:
        self._tracking_failures = 0
        self._velocity_failures = 0
        self._hold_failures = 0

    def evaluate(
        self,
        *,
        step: int,
        target_joints: Sequence[float],
        state: SceneState,
        constraints: ConstraintSet,
        protected_objects: Mapping[str, Pose] | None = None,
        require_holds: bool = True,
    ) -> MonitorEvent | None:
        joints = np.asarray(state.robot.joint_positions, dtype=float)
        velocities = np.asarray(state.robot.joint_velocities, dtype=float)
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(velocities)):
            return MonitorEvent(step, FailureCode.EXECUTION_DIVERGED, "non-finite robot state")

        tracking_error = float(np.max(np.abs(joints - np.asarray(target_joints, dtype=float))))
        self._tracking_failures = self._tracking_failures + 1 if tracking_error > self.config.tracking_tolerance else 0
        if self._tracking_failures >= self.config.tracking_patience:
            return MonitorEvent(
                step,
                FailureCode.EXECUTION_DIVERGED,
                "joint tracking error remained above tolerance",
                {"tracking_error": tracking_error},
            )

        maximum_velocity = float(np.max(np.abs(velocities)))
        self._velocity_failures = self._velocity_failures + 1 if maximum_velocity > self.config.maximum_joint_velocity else 0
        if self._velocity_failures >= self.config.velocity_patience:
            return MonitorEvent(
                step,
                FailureCode.EXECUTION_DIVERGED,
                "joint velocity remained above the stability limit",
                {"maximum_joint_velocity": maximum_velocity},
            )

        live_contacts = self.checker.check_live_contacts(constraints)
        if not live_contacts.feasible:
            return MonitorEvent(
                step,
                live_contacts.failure_code or FailureCode.ENV_COLLISION,
                live_contacts.message,
                {"contacts": [contact.__dict__ for contact in live_contacts.contacts]},
            )

        if require_holds:
            # A weak-but-real grip can read as momentarily "not holding" for
            # a single physics step without actually having slipped -
            # tracking error and velocity above already debounce the same
            # way; a hold check with zero patience was the odd one out.
            hold_event = self._check_attachments(step, state, constraints)
            self._hold_failures = self._hold_failures + 1 if hold_event is not None else 0
            if self._hold_failures >= self.config.hold_patience:
                return hold_event

        for name, reference in (protected_objects or {}).items():
            current = state.objects[name].pose
            displacement = float(np.linalg.norm(np.asarray(current.position) - np.asarray(reference.position)))
            if displacement > self.config.protected_object_tolerance:
                return MonitorEvent(
                    step,
                    FailureCode.OBJECT_DISPLACED,
                    f"protected object {name} moved unexpectedly",
                    {"displacement": displacement},
                )
        return None

    def _check_attachments(
        self,
        step: int,
        state: SceneState,
        constraints: ConstraintSet,
    ) -> MonitorEvent | None:
        for attachment in constraints.held_objects:
            obj = state.objects[attachment.object_name]
            if attachment.side not in obj.held_by:
                return MonitorEvent(
                    step,
                    FailureCode.SLIP_DETECTED,
                    f"{attachment.side} lost {attachment.object_name}",
                )
            observed = state.robot.arms[attachment.side].ee_pose.inverse().compose(obj.pose)
            position_error = float(
                np.linalg.norm(
                    np.asarray(observed.position) - np.asarray(attachment.ee_to_object.position)
                )
            )
            expected_rotation = Rotation.from_matrix(attachment.ee_to_object.as_matrix()[:3, :3])
            observed_rotation = Rotation.from_matrix(observed.as_matrix()[:3, :3])
            orientation_error = float((expected_rotation.inv() * observed_rotation).magnitude())
            if (
                position_error > constraints.position_tolerance
                or orientation_error > constraints.orientation_tolerance
            ):
                return MonitorEvent(
                    step,
                    FailureCode.CONSTRAINT_VIOLATION,
                    f"grasp transform drifted for {attachment.object_name}",
                    {
                        "side": attachment.side,
                        "position_error": position_error,
                        "orientation_error": orientation_error,
                    },
                )
        return None
