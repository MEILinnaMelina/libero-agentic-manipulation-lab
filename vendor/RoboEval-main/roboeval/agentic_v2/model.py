"""MuJoCo model indexing helpers used without changing RoboEval internals."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mujoco
import numpy as np

from roboeval.utils.physics_utils import get_colliders


def _base_task_objects(env: Any) -> dict[str, Any]:
    """Named task objects for the original three base tasks."""

    objects: dict[str, Any] = {}
    if hasattr(env, "kitchenpot"):
        objects["kitchenpot"] = env.kitchenpot
    if hasattr(env, "cube"):
        objects["cube"] = env.cube
    if hasattr(env, "blocks"):
        for idx, block in enumerate(env.blocks):
            objects[f"block_{idx}"] = block
    if hasattr(env, "packing_box"):
        objects["packing_box"] = env.packing_box
    if hasattr(env, "valves"):
        for idx, valve in enumerate(env.valves):
            objects[f"valve_{idx}"] = valve
    return objects


def element_id(physics: Any, element: Any) -> int:
    mjcf_element = getattr(element, "mjcf", element)
    return int(physics.bind(mjcf_element).element_id)


def geom_ids(physics: Any, obj: Any) -> set[int]:
    return {element_id(physics, geom) for geom in get_colliders(obj)}


class StaticPart:
    """A fixed scene fixture (shelf plank, ...) exposed as a named object.

    Never held and never moved; reported at its world-aligned box center so
    placement math can treat it exactly like any other support object."""

    is_fixture = True

    def __init__(self, env: Any, name: str, geom: Any) -> None:
        physics = env.mojo.physics
        bound = physics.bind(geom.mjcf)
        aabb = np.asarray(bound.aabb, dtype=float)
        rotation = np.asarray(bound.xmat, dtype=float).reshape(3, 3)
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        corners = (signs * aabb[3:] + aabb[:3]) @ rotation.T + np.asarray(bound.xpos, dtype=float)
        minimum, maximum = corners.min(axis=0), corners.max(axis=0)
        self.name = name
        self.geom = geom
        self.colliders = [geom]
        self.world_center = (minimum + maximum) / 2.0
        self.world_size = maximum - minimum
        self.canonical_size_override = self.world_size.copy()
        self.body = _FixtureBody(geom, self.world_center)


class _FixtureBody:
    def __init__(self, geom: Any, center: np.ndarray) -> None:
        self.mjcf = geom.mjcf
        self.geoms = [geom]
        self._center = np.asarray(center, dtype=float)

    def get_position(self) -> np.ndarray:
        return self._center.copy()

    @staticmethod
    def get_quaternion() -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])


def object_collider_geoms(env: Any, obj: Any) -> list[Any]:
    """Collision geoms of a task object, robust to props whose collider
    cache is empty because their geoms get contype/conaffinity from an
    MJCF default class (e.g. the garage valves) rather than explicitly."""

    colliders = list(get_colliders(obj))
    if colliders:
        return colliders
    body = getattr(obj, "body", None)
    physics = env.mojo.physics
    result = []
    for geom in getattr(body, "geoms", None) or []:
        bound = physics.bind(geom.mjcf)
        if int(bound.contype) or int(bound.conaffinity):
            result.append(geom)
    return result


def task_objects(env: Any) -> dict[str, Any]:
    """Named task objects for every Agentic v2 base task: the frozen v1 set
    plus the tray, books, and the bookshelf planks as static fixtures."""

    objects: dict[str, Any] = dict(_base_task_objects(env))
    tray = getattr(env, "breakfast_tray", None)
    if tray is not None:
        objects["tray"] = tray
    books = getattr(env, "books", None)
    if books:
        if len(books) == 1:
            objects["book"] = books[0]
        else:
            for index, book in enumerate(books):
                objects[f"book_{index}"] = book
    shelf = getattr(env, "book_shelf", None)
    if shelf is not None:
        objects["lower_shelf"] = StaticPart(env, "lower_shelf", shelf.lower_shelf_body)
        objects["upper_shelf"] = StaticPart(env, "upper_shelf", shelf.upper_shelf_body)
    return objects


def safe_name(model: Any, element_id_value: int, kind: str) -> str:
    name = model.id2name(int(element_id_value), kind)
    return str(name) if name else f"{kind}_{int(element_id_value)}"


def arm_joint_addresses(env: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return full-model qpos/qvel addresses in v2's 14-joint action order."""

    model = env.mojo.physics.model
    qpos_addresses: list[int] = []
    qvel_addresses: list[int] = []
    for actuator in env.robot.limb_actuators:
        if actuator.joint is None:
            raise ValueError(f"limb actuator {actuator.full_identifier} has no joint")
        joint_id = element_id(env.mojo.physics, actuator.joint)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
    return tuple(qpos_addresses), tuple(qvel_addresses)


def wrist_site_ids(env: Any) -> dict[str, int]:
    return {
        side.name.lower(): element_id(env.mojo.physics, site)
        for side, site in env.robot._wrist_sites.items()
    }


def object_geom_ids(env: Any) -> dict[str, set[int]]:
    physics = env.mojo.physics
    return {
        name: {element_id(physics, geom) for geom in object_collider_geoms(env, obj)}
        for name, obj in task_objects(env).items()
    }


def robot_geom_ids(env: Any) -> set[int]:
    if hasattr(env, "_robot_geoms"):
        return {int(value) for value in env._robot_geoms}
    model = env.mojo.physics.model
    ids = set()
    for geom_id in range(model.ngeom):
        name = safe_name(model, geom_id, "geom").lower()
        if "panda" in name or "hand_" in name or "robot" in name:
            ids.add(geom_id)
    return ids


def _named_scene_geoms(env: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    candidates = [
        ("table", getattr(env, "table", None)),
        ("floor", getattr(env, "floor", None)),
        ("cabinet_1", getattr(env, "cabinet_1", None)),
        ("cabinet_2", getattr(env, "cabinet_2", None)),
        # The bookshelf counter is the support surface books rest on; label
        # it as the table so the ordinary tabletop contact rules apply.
        ("table", getattr(getattr(env, "book_shelf", None), "counter", None)),
    ]
    for name, obj in candidates:
        if obj is None:
            continue
        for geom_id in geom_ids(env.mojo.physics, obj):
            result[geom_id] = f"scene:{name}"
    return result


def geom_labels(env: Any) -> dict[int, str]:
    """Build stable selectors consumed by AllowedContactPolicy."""

    model = env.mojo.physics.model
    labels = {
        geom_id: f"scene:{safe_name(model, geom_id, 'geom')}"
        for geom_id in range(model.ngeom)
    }
    labels.update(_named_scene_geoms(env))
    for object_name, ids in object_geom_ids(env).items():
        for geom_id in ids:
            labels[geom_id] = f"object:{object_name}"
    for geom_id in robot_geom_ids(env):
        name = safe_name(model, geom_id, "geom").lower()
        body_name = safe_name(
            model, int(model.geom_bodyid[geom_id]), "body"
        ).lower()
        if "nohand_left/" in name or "hand_left/" in name:
            side = "left"
        elif "nohand_right/" in name or "hand_right/" in name:
            side = "right"
        else:
            side = "body"
        part = (
            "finger"
            if "finger" in name or "pad" in name or "finger" in body_name
            else "link"
        )
        labels[geom_id] = f"robot:{side}:{part}"
    return labels


def object_freejoint_address(env: Any, object_name: str) -> int:
    """Find the seven-value free-joint qpos address for a task object."""

    obj = task_objects(env)[object_name]
    if getattr(obj, "is_fixture", False):
        raise ValueError(f"object {object_name!r} is a fixed scene part and cannot be held")
    body_id = element_id(env.mojo.physics, obj.body)
    model = env.mojo.physics.model
    body_id_cursor = body_id
    while body_id_cursor > 0:
        start = int(model.body_jntadr[body_id_cursor])
        count = int(model.body_jntnum[body_id_cursor])
        for joint_id in range(start, start + count):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(model.jnt_qposadr[joint_id])
        body_id_cursor = int(model.body_parentid[body_id_cursor])
    raise ValueError(f"object {object_name!r} has no free joint")


def contact_pairs(data: Any) -> Iterable[tuple[int, int, float]]:
    for contact in data.contact:
        yield int(contact.geom1), int(contact.geom2), float(contact.dist)


def body_velocity(env: Any, body: Any) -> tuple[np.ndarray, np.ndarray]:
    bound = env.mojo.physics.bind(getattr(body, "mjcf", body))
    try:
        cvel = np.asarray(bound.cvel, dtype=float).reshape(-1)
    except AttributeError:  # fixtures are bound to a geom, which has no cvel
        return np.zeros(3), np.zeros(3)
    if cvel.size != 6:
        return np.zeros(3), np.zeros(3)
    return cvel[3:].copy(), cvel[:3].copy()
