"""Read-only conversion from a live RoboEval episode to typed scene state."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.model import (
    body_velocity,
    contact_pairs,
    geom_labels,
    object_collider_geoms,
    object_geom_ids,
    task_objects,
)
from roboeval.agentic_v2.task_specs import task_key_from_env
from roboeval.agentic_v2.types import ArmState, HeldObjectAttachment, ObjectState, Pose, RobotState, SceneState


def _metrics(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metrics(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _body_pose(body: Any) -> Pose:
    return Pose(tuple(body.get_position()), tuple(body.get_quaternion()))


def _object_extent(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(obj, "world_size", None) is not None:
        return (
            np.asarray(obj.world_center, dtype=float),
            np.asarray(obj.world_size, dtype=float),
        )
    if getattr(obj, "aabb", None) is not None:
        return (
            np.asarray(obj.aabb.get_position(), dtype=float),
            np.asarray(obj.aabb.mjcf.size, dtype=float),
        )
    bbox = getattr(obj, "bbox", None)
    if bbox is not None:
        bbox.update()
        minimum = np.asarray(bbox.min, dtype=float)
        maximum = np.asarray(bbox.max, dtype=float)
        if np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)):
            return (minimum + maximum) / 2.0, maximum - minimum
    return np.asarray(obj.body.get_position(), dtype=float), np.zeros(3)


_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=float,
)


def _wxyz_matrix(quaternion: Any) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return Rotation.from_quat((x, y, z, w)).as_matrix()


def _geom_to_root_transform(physics: Any, geom_element: Any, root_element: Any):
    """Pose of a geom in its root body's frame, composed through fixed
    intermediate bodies; None when a joint sits on the chain (the geom then
    belongs to an articulated part whose pose is not fixed in the root)."""

    rotation = _wxyz_matrix(physics.bind(geom_element).quat)
    position = np.asarray(physics.bind(geom_element).pos, dtype=float)
    body = geom_element.parent
    while body is not None and body is not root_element and body.tag == "body":
        if body.find_all("joint", immediate_children_only=True):
            return None
        body_rotation = _wxyz_matrix(physics.bind(body).quat)
        body_position = np.asarray(physics.bind(body).pos, dtype=float)
        rotation = body_rotation @ rotation
        position = body_rotation @ position + body_position
        body = body.parent
    return rotation, position


def _canonical_size(physics: Any, obj: Any) -> np.ndarray:
    """Rotation-invariant physical extent, from the object's own geoms in its
    body frame - unlike `_object_extent`, this does not inflate when the
    object tilts, so it is safe for gripper-aperture/fit checks. Each geom's
    own placement inside the body (offset and rotation - the book's mesh,
    for instance, is rotated 90 degrees relative to its body) is honored."""

    override = getattr(obj, "canonical_size_override", None)
    if override is not None:
        return np.asarray(override, dtype=float)
    bbox = getattr(obj, "bbox", None)
    body = getattr(bbox, "body", None) or getattr(obj, "body", None)
    if body is not None and getattr(body, "geoms", None):
        root_element = body.mjcf
        minimum = np.array([np.inf, np.inf, np.inf])
        maximum = np.array([-np.inf, -np.inf, -np.inf])
        for geom in body.geoms:
            transform = _geom_to_root_transform(physics, geom.mjcf, root_element)
            if transform is None:
                continue
            rotation, position = transform
            aabb = np.asarray(physics.bind(geom.mjcf).aabb, dtype=float)
            corners = (_CORNER_SIGNS * aabb[3:] + aabb[:3]) @ rotation.T + position
            minimum = np.minimum(minimum, corners.min(axis=0))
            maximum = np.maximum(maximum, corners.max(axis=0))
        if np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)):
            return maximum - minimum
    site = getattr(bbox, "site", None)
    if site is not None:
        return np.asarray(physics.bind(site.mjcf).size, dtype=float) * 2.0
    return np.zeros(3)


def vertical_half_extent(obj: ObjectState) -> float:
    """Half of the object's extent along world Z at its *current* orientation,
    from the rotation-invariant canonical size - the right half-height for
    resting-surface math whether the object is lying flat or standing on
    end (canonical_size[2]/2 is only correct for world-aligned bodies)."""

    rotation = np.asarray(obj.pose.as_matrix())[:3, :3]
    return float(0.5 * np.abs(rotation[2, :]) @ np.asarray(obj.canonical_size))


def vertical_extent(obj: ObjectState) -> float:
    return 2.0 * vertical_half_extent(obj)


def world_extent(obj: ObjectState, axis: Any) -> float:
    """Object extent along an arbitrary world direction from the canonical
    size and the live rotation (an upper bound: the projected bounding box)."""

    rotation = np.asarray(obj.pose.as_matrix())[:3, :3]
    direction = np.asarray(axis, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    return float(np.abs(rotation.T @ direction) @ np.asarray(obj.canonical_size))


def collect_scene_state(env: Any, info: dict[str, Any] | None = None) -> SceneState:
    """Collect current state without stepping or modifying the environment."""

    info = dict(env.get_info() if info is None else info)
    objects = task_objects(env)
    labels = geom_labels(env)
    object_geoms = object_geom_ids(env)
    contacts: dict[str, set[str]] = {name: set() for name in objects}
    for geom1, geom2, distance in contact_pairs(env.mojo.physics.data):
        if distance > 1e-8:
            continue
        for name, ids in object_geoms.items():
            if geom1 in ids:
                contacts[name].add(labels.get(geom2, f"geom:{geom2}"))
            if geom2 in ids:
                contacts[name].add(labels.get(geom1, f"geom:{geom1}"))

    object_states: dict[str, ObjectState] = {}
    for name, obj in objects.items():
        center, size = _object_extent(obj)
        canonical_size = _canonical_size(env.mojo.physics, obj)
        linear_velocity, angular_velocity = body_velocity(env, obj.body)
        fixed = bool(getattr(obj, "is_fixture", False))
        colliders = [] if fixed else object_collider_geoms(env, obj)
        held_by = tuple(
            side.name.lower()
            for side in env.robot.grippers
            if colliders and env.robot.is_gripper_holding_object(colliders, side)
        )
        object_states[name] = ObjectState(
            name=name,
            pose=_body_pose(obj.body),
            aabb_center=tuple(center),
            size=tuple(size),
            linear_velocity=tuple(linear_velocity),
            angular_velocity=tuple(angular_velocity),
            contacts=tuple(sorted(contacts[name])),
            held_by=held_by,
            canonical_size=tuple(canonical_size),
            fixed=fixed,
        )

    arm_qpos = np.asarray(env.robot.qpos_actuated[:-len(env.robot.grippers)], dtype=float)
    arm_qvel = np.asarray(env.robot.qvel_actuated[:-len(env.robot.grippers)], dtype=float)
    last_action = np.asarray(env.action, dtype=float)
    arm_count = len(env.robot.grippers)
    joints_per_arm = len(arm_qpos) // arm_count
    arms: dict[str, ArmState] = {}
    for index, (side, gripper) in enumerate(env.robot.grippers.items()):
        key = side.name.lower()
        start = index * joints_per_arm
        stop = start + joints_per_arm
        command_index = len(last_action) - arm_count + index
        arms[key] = ArmState(
            side=key,
            joint_positions=tuple(arm_qpos[start:stop]),
            joint_velocities=tuple(arm_qvel[start:stop]),
            ee_pose=Pose(tuple(gripper.wrist_position), tuple(gripper.wrist_orientation.elements)),
            gripper_command=float(last_action[command_index]),
            gripper_aperture_m=float(gripper.aperture_m),
            holding=tuple(sorted(name for name, state in object_states.items() if key in state.held_by)),
        )

    return SceneState(
        task_key=task_key_from_env(env),
        task_name=str(env.task_name),
        seed=int(env.seed),
        control_frequency=int(env.control_frequency),
        action_shape=tuple(env.action_space.shape),
        robot=RobotState(
            joint_positions=tuple(arm_qpos),
            joint_velocities=tuple(arm_qvel),
            arms=arms,
        ),
        objects=object_states,
        metrics=_metrics(info),
    )


def infer_handover_roles(state: SceneState, object_name: str) -> dict[str, str]:
    """Infer donor and receiver from the live hold state."""

    obj = state.objects[object_name]
    if len(obj.held_by) != 1:
        raise ValueError(
            f"handover requires exactly one current holder; {object_name} is held by {obj.held_by}"
        )
    donor = obj.held_by[0]
    receiver = "right" if donor == "left" else "left"
    return {"donor": donor, "receiver": receiver}


def capture_attachment(state: SceneState, object_name: str, side: str) -> HeldObjectAttachment:
    """Capture T_ee_object from an observed verified grasp."""

    arm = state.robot.arms[side]
    obj = state.objects[object_name]
    ee_to_object = arm.ee_pose.inverse().compose(obj.pose)
    return HeldObjectAttachment(object_name=object_name, side=side, ee_to_object=ee_to_object)
