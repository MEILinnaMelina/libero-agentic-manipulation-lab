"""Typed contracts shared by the RoboEval Agentic-TAMP stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from fnmatch import fnmatchcase
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class FailureCode(str, Enum):
    IK_UNREACHABLE = "IK_UNREACHABLE"
    JOINT_LIMIT = "JOINT_LIMIT"
    SELF_COLLISION = "SELF_COLLISION"
    ENV_COLLISION = "ENV_COLLISION"
    HELD_OBJECT_COLLISION = "HELD_OBJECT_COLLISION"
    OTHER_OBJECT_COLLISION = "OTHER_OBJECT_COLLISION"
    NO_VALID_GRASP = "NO_VALID_GRASP"
    GRIPPER_APERTURE_MISMATCH = "GRIPPER_APERTURE_MISMATCH"
    HANDOVER_REGION_EMPTY = "HANDOVER_REGION_EMPTY"
    PLACEMENT_UNREACHABLE = "PLACEMENT_UNREACHABLE"
    RELEASE_FAILED = "RELEASE_FAILED"
    PATH_BLOCKED = "PATH_BLOCKED"
    APPROACH_FAILED = "APPROACH_FAILED"
    GRASP_FAILED = "GRASP_FAILED"
    OBJECT_DISPLACED = "OBJECT_DISPLACED"
    SLIP_DETECTED = "SLIP_DETECTED"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    EXECUTION_DIVERGED = "EXECUTION_DIVERGED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
    TERMINATED = "TERMINATED"


class ContactKind(str, Enum):
    SELF = "self"
    ENVIRONMENT = "environment"
    HELD_OBJECT = "held_object"
    OBJECT_OBJECT = "object_object"
    ALLOWED = "allowed"
    OTHER = "other"


class SkillName(str, Enum):
    GRASP = "grasp"
    BIMANUAL_GRASP = "bimanual_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    HANDOVER = "handover"
    PLACE = "place"
    CLOSE_FLAP = "close_flap"
    ROTATE = "rotate"
    FINISH = "finish"


def _tuple(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def to_jsonable(value: Any) -> Any:
    """Recursively convert contracts and numpy values to JSON-compatible data."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _tuple(self.position, 3, "position"))
        quat = np.asarray(_tuple(self.quaternion_wxyz, 4, "quaternion_wxyz"))
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            raise ValueError("quaternion_wxyz cannot be zero")
        object.__setattr__(self, "quaternion_wxyz", tuple((quat / norm).tolist()))

    @classmethod
    def identity(cls) -> "Pose":
        return cls((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Pose":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("pose matrix must have shape (4, 4)")
        xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return cls(tuple(matrix[:3, 3]), (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    @classmethod
    def from_xyz_rpy(cls, xyz: Sequence[float], rpy: Sequence[float]) -> "Pose":
        xyzw = Rotation.from_euler("xyz", _tuple(rpy, 3, "rpy")).as_quat()
        return cls(_tuple(xyz, 3, "xyz"), (xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    def as_matrix(self) -> np.ndarray:
        w, x, y, z = self.quaternion_wxyz
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat((x, y, z, w)).as_matrix()
        matrix[:3, 3] = self.position
        return matrix

    def as_xyz_rpy(self) -> np.ndarray:
        w, x, y, z = self.quaternion_wxyz
        rpy = Rotation.from_quat((x, y, z, w)).as_euler("xyz")
        return np.concatenate((np.asarray(self.position), rpy))

    def compose(self, other: "Pose") -> "Pose":
        return Pose.from_matrix(self.as_matrix() @ other.as_matrix())

    def inverse(self) -> "Pose":
        return Pose.from_matrix(np.linalg.inv(self.as_matrix()))


@dataclass(frozen=True)
class AllowedContactRule:
    first: str
    second: str
    # Optional per-rule penetration cap; falls back to the policy's
    # tolerance when None. Lets a light fingertip-support-surface touch be
    # tolerated at a few mm without loosening finger-object contact.
    penetration_tolerance: float | None = None

    def matches(self, first: str, second: str) -> bool:
        return (
            fnmatchcase(first, self.first) and fnmatchcase(second, self.second)
        ) or (
            fnmatchcase(first, self.second) and fnmatchcase(second, self.first)
        )


@dataclass(frozen=True)
class AllowedContactPolicy:
    rules: tuple[AllowedContactRule, ...] = ()
    penetration_tolerance: float = 1e-6

    def allows(self, first: str, second: str, distance: float = 0.0) -> bool:
        for rule in self.rules:
            if not rule.matches(first, second):
                continue
            tolerance = (
                self.penetration_tolerance
                if rule.penetration_tolerance is None
                else rule.penetration_tolerance
            )
            if distance >= -tolerance:
                return True
        return False


@dataclass(frozen=True)
class HeldObjectAttachment:
    object_name: str
    side: str
    ee_to_object: Pose


@dataclass(frozen=True)
class ConstraintSet:
    allowed_contacts: AllowedContactPolicy = field(default_factory=AllowedContactPolicy)
    held_objects: tuple[HeldObjectAttachment, ...] = ()
    position_tolerance: float = 0.02
    orientation_tolerance: float = 0.15
    maximum_object_tilt: float | None = None


@dataclass(frozen=True)
class ArmState:
    side: str
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    ee_pose: Pose
    gripper_command: float
    gripper_aperture_m: float
    holding: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobotState:
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    arms: Mapping[str, ArmState]


@dataclass(frozen=True)
class ObjectState:
    name: str
    pose: Pose
    aabb_center: tuple[float, float, float]
    size: tuple[float, float, float]
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    contacts: tuple[str, ...] = ()
    held_by: tuple[str, ...] = ()
    # Rotation-invariant physical extent, from the object's own geoms in its
    # body frame. `size`/`aabb_center` are a live world-frame AABB (correct
    # for placement/height math against the *current* pose) but inflate as
    # soon as the object tilts even slightly - unsafe for aperture/fit
    # checks, which need the object's true, fixed cross-section.
    canonical_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Fixed scene parts (shelf planks) that are exposed as named placement
    # supports but can never be grasped, moved, or used as a resting
    # reference for other objects.
    fixed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "aabb_center", _tuple(self.aabb_center, 3, "aabb_center"))
        object.__setattr__(self, "size", _tuple(self.size, 3, "size"))
        object.__setattr__(self, "linear_velocity", _tuple(self.linear_velocity, 3, "linear_velocity"))
        object.__setattr__(self, "angular_velocity", _tuple(self.angular_velocity, 3, "angular_velocity"))
        object.__setattr__(self, "canonical_size", _tuple(self.canonical_size, 3, "canonical_size"))


@dataclass(frozen=True)
class SceneState:
    task_key: str
    task_name: str
    seed: int
    control_frequency: int
    action_shape: tuple[int, ...]
    robot: RobotState
    objects: Mapping[str, ObjectState]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SceneState":
        robot_value = value["robot"]
        arms = {
            side: ArmState(
                side=arm["side"],
                joint_positions=tuple(arm["joint_positions"]),
                joint_velocities=tuple(arm["joint_velocities"]),
                ee_pose=Pose(**arm["ee_pose"]),
                gripper_command=float(arm["gripper_command"]),
                gripper_aperture_m=float(arm["gripper_aperture_m"]),
                holding=tuple(arm.get("holding", ())),
            )
            for side, arm in robot_value["arms"].items()
        }
        objects = {
            name: ObjectState(
                name=obj["name"],
                pose=Pose(**obj["pose"]),
                aabb_center=tuple(obj["aabb_center"]),
                size=tuple(obj["size"]),
                linear_velocity=tuple(obj.get("linear_velocity", (0.0, 0.0, 0.0))),
                angular_velocity=tuple(obj.get("angular_velocity", (0.0, 0.0, 0.0))),
                contacts=tuple(obj.get("contacts", ())),
                held_by=tuple(obj.get("held_by", ())),
                canonical_size=tuple(obj.get("canonical_size", (0.0, 0.0, 0.0))),
                fixed=bool(obj.get("fixed", False)),
            )
            for name, obj in value["objects"].items()
        }
        return cls(
            task_key=str(value["task_key"]),
            task_name=str(value["task_name"]),
            seed=int(value["seed"]),
            control_frequency=int(value["control_frequency"]),
            action_shape=tuple(value["action_shape"]),
            robot=RobotState(
                joint_positions=tuple(robot_value["joint_positions"]),
                joint_velocities=tuple(robot_value["joint_velocities"]),
                arms=arms,
            ),
            objects=objects,
            metrics=dict(value.get("metrics", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "SceneState":
        return cls.from_dict(json.loads(payload))


_LOW_LEVEL_FIELDS = frozenset(
    {
        "qpos",
        "joint_positions",
        "joint_values",
        "pose",
        "target_pose",
        "offset",
        "ee_offset",
        "yaw",
        "steps",
        "gain",
        "gains",
        "tolerance",
        "controller",
    }
)


@dataclass(frozen=True)
class SkillRequest:
    skill: SkillName
    object_name: str | None = None
    roles: Mapping[str, str] = field(default_factory=dict)
    goal: str = ""
    strategy: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill, SkillName):
            object.__setattr__(self, "skill", SkillName(str(self.skill)))
        invalid_roles = set(self.roles) - {"left", "right", "donor", "receiver"}
        if invalid_roles:
            raise ValueError(f"invalid role keys: {sorted(invalid_roles)}")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.roles.items()
        ):
            raise ValueError("role keys and values must be strings")
        if self.skill != SkillName.FINISH and not (self.object_name or self.roles):
            raise ValueError("a non-finish skill requires an object or arm roles")
        if any(key.lower() in _LOW_LEVEL_FIELDS for key in self.roles):
            raise ValueError("roles cannot contain low-level control fields")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "skill": self.skill.value,
            "object": self.object_name,
            "roles": dict(self.roles),
            "goal": self.goal,
        }
        if self.strategy is not None:
            result["strategy"] = self.strategy
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillRequest":
        unknown = set(value) - {"skill", "object", "roles", "goal", "strategy"}
        forbidden = unknown & _LOW_LEVEL_FIELDS
        if forbidden:
            raise ValueError(f"low-level fields are forbidden: {sorted(forbidden)}")
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        return cls(
            skill=SkillName(str(value["skill"]).lower()),
            object_name=value.get("object"),
            roles=dict(value.get("roles") or {}),
            goal=str(value.get("goal") or ""),
            strategy=value.get("strategy"),
        )


@dataclass(frozen=True)
class CollisionContact:
    geom1_id: int
    geom2_id: int
    geom1_name: str
    geom2_name: str
    first: str
    second: str
    distance: float
    kind: ContactKind
    allowed: bool = False


@dataclass(frozen=True)
class IKCandidate:
    seed_name: str
    joint_positions: tuple[float, ...]
    converged: bool
    position_error: float
    orientation_error: float
    iterations: int = 0
    score: float = float("inf")
    failure_code: FailureCode | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.converged and self.failure_code is None


@dataclass(frozen=True)
class MotionCandidate:
    name: str
    target_poses: Mapping[str, Pose]
    ik: IKCandidate
    score: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspCandidate:
    name: str
    object_name: str
    side: str
    pregrasp_pose: Pose
    grasp_pose: Pose
    approach_axis: tuple[float, float, float]
    required_aperture: float
    contact_policy: AllowedContactPolicy
    score: float
    # True for a horizontal-hand pinch of a thin object's overhanging edge
    # (fingers above and below it) rather than a top-down pinch; the grasp
    # skill departs slightly backward as well as upward for these.
    edge_grasp: bool = False


@dataclass(frozen=True)
class RendezvousCandidate:
    name: str
    object_pose: Pose
    receiver_pregrasp_pose: Pose
    receiver_grasp_pose: Pose
    receiver_side: str
    contact_policy: AllowedContactPolicy
    score: float
    # Object width across this candidate's finger-closing axis. A real grip
    # leaves the aperture near this value; fingers that closed *beside*
    # the object (glancing pad contact still reads as "holding") close to
    # ~0 - the aperture is the discriminator pad contact alone is not.
    grip_width: float = 0.0


@dataclass(frozen=True)
class PlacementCandidate:
    name: str
    object_name: str
    support_name: str
    preplace_object_pose: Pose
    placed_object_pose: Pose
    contact_policy: AllowedContactPolicy
    score: float


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    failure_code: FailureCode | None = None
    message: str = ""
    contacts: tuple[CollisionContact, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryPoint:
    joint_positions: tuple[float, ...]
    time_from_start: float


@dataclass(frozen=True)
class MotionPlan:
    name: str
    points: tuple[TrajectoryPoint, ...]
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    score: float = 0.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorEvent:
    step: int
    code: FailureCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReport:
    success: bool
    plan_name: str
    executed_points: int
    final_state: SceneState
    events: tuple[MonitorEvent, ...] = ()
    failure_code: FailureCode | None = None
    benchmark_success: float = 0.0
    subtask_progress: float = 0.0
    behavior_quality: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SkillResult:
    request: SkillRequest
    success: bool
    message: str
    state: SceneState
    failure_code: FailureCode | None = None
    execution_reports: tuple[ExecutionReport, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class TrialReport:
    task_key: str
    seed: int
    method: str
    benchmark_success: float
    subtask_progress: float
    behavior_quality: Mapping[str, Any]
    skill_results: tuple[SkillResult, ...]
    final_state: SceneState
    failure_code: FailureCode | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
