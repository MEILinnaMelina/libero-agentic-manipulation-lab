"""Side-effect-free MuJoCo feasibility checks for candidate joint states."""

from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from roboeval.agentic_v2.model import (
    arm_joint_addresses,
    contact_pairs,
    geom_labels,
    object_freejoint_address,
    object_geom_ids,
    robot_geom_ids,
    safe_name,
    wrist_site_ids,
)
from roboeval.agentic_v2.types import (
    AllowedContactPolicy,
    CollisionContact,
    ConstraintSet,
    ContactKind,
    FailureCode,
    FeasibilityReport,
    Pose,
)


def data_signature(data: Any) -> dict[str, Any]:
    """Capture all live fields that planning is required not to mutate."""

    return {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "qacc": np.asarray(data.qacc).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "act": np.asarray(data.act).copy(),
        "time": float(data.time),
        "ncon": int(data.ncon),
        "contacts": tuple(contact_pairs(data)),
    }


def signatures_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first.keys() != second.keys():
        return False
    for key in ("qpos", "qvel", "qacc", "ctrl", "act"):
        if not np.array_equal(first[key], second[key]):
            return False
    return all(first[key] == second[key] for key in ("time", "ncon", "contacts"))


class CollisionChecker:
    """Evaluate candidates on independent MjData snapshots."""

    def __init__(
        self,
        env: Any,
        *,
        contact_margin: float = 1e-8,
        enforce_contacts: bool = True,
    ) -> None:
        self.env = env
        self.physics = env.mojo.physics
        self.model = self.physics.model
        self.live_data = self.physics.data
        self.contact_margin = float(contact_margin)
        self.enforce_contacts = bool(enforce_contacts)
        self.qpos_addresses, _ = arm_joint_addresses(env)
        self.labels = geom_labels(env)
        self.robot_geoms = robot_geom_ids(env)
        self.object_geoms = object_geom_ids(env)
        self.site_ids = wrist_site_ids(env)
        arm_count = len(self.qpos_addresses)
        self.lower = np.asarray(env.action_space.low[:arm_count], dtype=float)
        self.upper = np.asarray(env.action_space.high[:arm_count], dtype=float)

    def clone_live_data(self) -> Any:
        clone = copy.copy(self.live_data)
        if clone.ptr is self.live_data.ptr or np.shares_memory(clone.qpos, self.live_data.qpos):
            raise RuntimeError("MuJoCo planning clone shares writable state with live data")
        return clone

    def check(
        self,
        joint_positions: np.ndarray | tuple[float, ...],
        constraints: ConstraintSet | None = None,
    ) -> FeasibilityReport:
        constraints = constraints or ConstraintSet()
        candidate = np.asarray(joint_positions, dtype=float)
        if candidate.shape != (len(self.qpos_addresses),) or not np.all(np.isfinite(candidate)):
            return FeasibilityReport(
                False,
                FailureCode.JOINT_LIMIT,
                f"expected {len(self.qpos_addresses)} finite arm joints",
            )
        outside = np.flatnonzero((candidate < self.lower) | (candidate > self.upper))
        if outside.size:
            return FeasibilityReport(
                False,
                FailureCode.JOINT_LIMIT,
                f"joint indices outside limits: {outside.tolist()}",
                diagnostics={"indices": outside.tolist()},
            )

        before = data_signature(self.live_data)
        clone = self.clone_live_data()
        clone.qpos[np.asarray(self.qpos_addresses)] = candidate
        clone.qvel[:] = 0.0
        clone.qacc[:] = 0.0
        mujoco.mj_fwdPosition(self.model.ptr, clone.ptr)

        attachment_error = self._propagate_held_objects(clone, constraints)
        if attachment_error is not None:
            return attachment_error
        if constraints.held_objects:
            mujoco.mj_fwdPosition(self.model.ptr, clone.ptr)

        contacts = self._classify_contacts(clone, constraints.allowed_contacts, constraints)
        forbidden = tuple(contact for contact in contacts if not contact.allowed)
        after = data_signature(self.live_data)
        if not signatures_equal(before, after):
            raise RuntimeError("collision planning mutated the live MuJoCo state")
        if forbidden and self.enforce_contacts:
            code = self._failure_code(forbidden)
            return FeasibilityReport(
                False,
                code,
                f"{len(forbidden)} forbidden contact(s)",
                contacts=contacts,
            )
        return FeasibilityReport(True, contacts=contacts)

    def check_live_contacts(
        self,
        constraints: ConstraintSet | None = None,
    ) -> FeasibilityReport:
        """Classify actual live contacts without moving joints or held objects."""

        constraints = constraints or ConstraintSet()
        before = data_signature(self.live_data)
        contacts = self._classify_contacts(
            self.live_data,
            constraints.allowed_contacts,
            constraints,
        )
        forbidden = tuple(contact for contact in contacts if not contact.allowed)
        if not signatures_equal(before, data_signature(self.live_data)):
            raise RuntimeError("live contact classification mutated MuJoCo state")
        if forbidden and self.enforce_contacts:
            return FeasibilityReport(
                False,
                self._failure_code(forbidden),
                f"{len(forbidden)} live forbidden contact(s)",
                contacts=contacts,
            )
        return FeasibilityReport(True, contacts=contacts)

    def _site_pose(self, data: Any, side: str) -> Pose:
        site_id = self.site_ids[side]
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
        matrix[:3, 3] = np.asarray(data.site_xpos[site_id])
        return Pose.from_matrix(matrix)

    def _propagate_held_objects(
        self,
        data: Any,
        constraints: ConstraintSet,
    ) -> FeasibilityReport | None:
        predictions: dict[str, list[Pose]] = defaultdict(list)
        for attachment in constraints.held_objects:
            if attachment.side not in self.site_ids:
                return FeasibilityReport(
                    False,
                    FailureCode.CONSTRAINT_VIOLATION,
                    f"unknown attachment side {attachment.side!r}",
                )
            predictions[attachment.object_name].append(
                self._site_pose(data, attachment.side).compose(attachment.ee_to_object)
            )

        for object_name, poses in predictions.items():
            positions = np.asarray([pose.position for pose in poses])
            if len(poses) > 1:
                position_spread = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
                rotations = [pose.as_matrix()[:3, :3] for pose in poses]
                first_rotation = Rotation.from_matrix(rotations[0])
                rotation_spread = max(
                    (first_rotation.inv() * Rotation.from_matrix(rotation)).magnitude()
                    for rotation in rotations
                )
                if (
                    position_spread > constraints.position_tolerance
                    or rotation_spread > constraints.orientation_tolerance
                ):
                    return FeasibilityReport(
                        False,
                        FailureCode.CONSTRAINT_VIOLATION,
                        f"attachments disagree for {object_name}",
                        diagnostics={
                            "position_spread": position_spread,
                            "orientation_spread": float(rotation_spread),
                        },
                    )
            mean_position = positions.mean(axis=0)
            rotations = Rotation.from_matrix(
                np.asarray([pose.as_matrix()[:3, :3] for pose in poses])
            )
            mean_xyzw = rotations.mean().as_quat()
            qpos_address = object_freejoint_address(self.env, object_name)
            data.qpos[qpos_address : qpos_address + 3] = mean_position
            data.qpos[qpos_address + 3 : qpos_address + 7] = (
                mean_xyzw[3],
                mean_xyzw[0],
                mean_xyzw[1],
                mean_xyzw[2],
            )
        return None

    def _classify_contacts(
        self,
        data: Any,
        policy: AllowedContactPolicy,
        constraints: ConstraintSet,
    ) -> tuple[CollisionContact, ...]:
        held_names = {attachment.object_name for attachment in constraints.held_objects}
        held_geoms = {
            geom_id
            for name in held_names
            for geom_id in self.object_geoms.get(name, set())
        }
        all_object_geoms = {
            geom_id for ids in self.object_geoms.values() for geom_id in ids
        }
        result: list[CollisionContact] = []
        for geom1, geom2, distance in contact_pairs(data):
            if distance > self.contact_margin:
                continue
            robot1, robot2 = geom1 in self.robot_geoms, geom2 in self.robot_geoms
            held1, held2 = geom1 in held_geoms, geom2 in held_geoms
            object1, object2 = geom1 in all_object_geoms, geom2 in all_object_geoms
            held_hits_other_object = (
                (held1 and object2 and not held2)
                or (held2 and object1 and not held1)
            )
            if robot1 and robot2:
                kind = ContactKind.SELF
            elif held_hits_other_object:
                kind = ContactKind.OTHER
            elif held1 or held2:
                kind = ContactKind.HELD_OBJECT
            elif robot1 or robot2:
                kind = ContactKind.ENVIRONMENT
            elif object1 and object2:
                kind = ContactKind.OBJECT_OBJECT
            else:
                continue
            first = self.labels.get(geom1, f"geom:{geom1}")
            second = self.labels.get(geom2, f"geom:{geom2}")
            if kind == ContactKind.SELF and first == second and first.endswith(":finger"):
                # The two fingers of one hand touch each other whenever the
                # gripper is fully closed on nothing; that is the commanded
                # state, not a self-collision.
                continue
            allowed = policy.allows(first, second, distance)
            if kind == ContactKind.OBJECT_OBJECT and not (held1 or held2) and not allowed:
                continue
            result.append(
                CollisionContact(
                    geom1_id=geom1,
                    geom2_id=geom2,
                    geom1_name=safe_name(self.model, geom1, "geom"),
                    geom2_name=safe_name(self.model, geom2, "geom"),
                    first=first,
                    second=second,
                    distance=distance,
                    kind=kind,
                    allowed=allowed,
                )
            )
        return tuple(result)

    @staticmethod
    def _failure_code(contacts: tuple[CollisionContact, ...]) -> FailureCode:
        kinds = {contact.kind for contact in contacts}
        if ContactKind.SELF in kinds:
            return FailureCode.SELF_COLLISION
        if ContactKind.HELD_OBJECT in kinds:
            return FailureCode.HELD_OBJECT_COLLISION
        if ContactKind.OTHER in kinds:
            return FailureCode.OTHER_OBJECT_COLLISION
        return FailureCode.ENV_COLLISION
