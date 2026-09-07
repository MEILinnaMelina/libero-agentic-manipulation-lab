"""Bimanual transform and coordination constraints."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.types import HeldObjectAttachment, Pose, SceneState


def ee_targets_for_object_pose(
    object_pose: Pose,
    attachments: Sequence[HeldObjectAttachment],
) -> dict[str, Pose]:
    """Solve T_world_ee = T_world_object @ inverse(T_ee_object)."""

    targets: dict[str, Pose] = {}
    for attachment in attachments:
        targets[attachment.side] = object_pose.compose(attachment.ee_to_object.inverse())
    return targets


def attachment_errors(
    state: SceneState,
    attachments: Sequence[HeldObjectAttachment],
) -> dict[str, dict[str, float]]:
    result = {}
    for attachment in attachments:
        observed = (
            state.robot.arms[attachment.side].ee_pose.inverse()
            .compose(state.objects[attachment.object_name].pose)
        )
        position = float(
            np.linalg.norm(
                np.asarray(observed.position)
                - np.asarray(attachment.ee_to_object.position)
            )
        )
        expected_rotation = Rotation.from_matrix(attachment.ee_to_object.as_matrix()[:3, :3])
        observed_rotation = Rotation.from_matrix(observed.as_matrix()[:3, :3])
        orientation = float((expected_rotation.inv() * observed_rotation).magnitude())
        result[attachment.side] = {
            "position_error": position,
            "orientation_error": orientation,
        }
    return result


def object_tilt(pose: Pose) -> float:
    local_up = pose.as_matrix()[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.arccos(np.clip(np.dot(local_up, (0.0, 0.0, 1.0)), -1.0, 1.0)))


def object_tilt_change(start: Pose, final: Pose) -> float:
    """Angle by which the object's orientation relative to gravity changed
    between two poses, ignoring yaw. Unlike `object_tilt`, this is correct
    for objects whose body Z is not world-up at rest (the tray's body Y is
    up, the book's mesh is rotated inside its body)."""

    up = np.array([0.0, 0.0, 1.0])
    start_up = start.as_matrix()[:3, :3].T @ up
    final_up = final.as_matrix()[:3, :3].T @ up
    return float(np.arccos(np.clip(np.dot(start_up, final_up), -1.0, 1.0)))


def synchronized_velocity_difference(
    previous: SceneState,
    current: SceneState,
) -> float:
    left = np.linalg.norm(current.robot.arms["left"].joint_velocities)
    right = np.linalg.norm(current.robot.arms["right"].joint_velocities)
    return float(abs(left - right))
