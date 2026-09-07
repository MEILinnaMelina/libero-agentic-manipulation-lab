"""Task registry and fixed environment construction for Agentic v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roboeval.action_modes import JointPositionActionMode
from roboeval.envs.lift_pot import LiftPot
from roboeval.envs.lift_tray import LiftTray
from roboeval.envs.manipulation import CubeHandover, StackTwoBlocks, VerticalCubeHandover
from roboeval.envs.pack_objects import PackBox
from roboeval.envs.rotate_utility_objects import RotateValve
from roboeval.envs.stack_books import PickSingleBookFromTable, StackSingleBookShelf
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import CameraConfig, ObservationConfig


@dataclass(frozen=True)
class TaskSpec:
    key: str
    env_cls: type
    objects: tuple[str, ...]
    success_condition: str
    stage_meaning: dict[int, str]


TASK_SPECS: dict[str, TaskSpec] = {
    "lift_pot": TaskSpec(
        key="lift_pot",
        env_cls=LiftPot,
        objects=("kitchenpot",),
        success_condition=(
            "Lift the pot at least 0.10 m without cabinet/floor collision while "
            "both grippers hold it and its orientation remains valid."
        ),
        stage_meaning={
            1: "left gripper holds the pot",
            2: "right gripper holds the pot",
            3: "pot exceeds the lift threshold",
            4: "both holds, lift, and orientation are valid",
        },
    ),
    "cube_handover": TaskSpec(
        key="cube_handover",
        env_cls=CubeHandover,
        objects=("cube",),
        success_condition=(
            "Establish an initial holder, transfer the rod to the opposite "
            "gripper, and release the initial gripper without dropping it."
        ),
        stage_meaning={
            1: "one gripper has held the rod",
            2: "the opposite gripper has held the rod",
        },
    ),
    "vertical_cube_handover": TaskSpec(
        key="vertical_cube_handover",
        env_cls=VerticalCubeHandover,
        objects=("cube",),
        success_condition=(
            "The rod starts standing on end. Establish an initial holder, "
            "transfer the rod to the opposite gripper, and release the initial "
            "gripper without the rod ever touching the floor."
        ),
        stage_meaning={
            1: "one gripper has held the rod",
            2: "the opposite gripper has held the rod",
        },
    ),
    "stack_two_blocks": TaskSpec(
        key="stack_two_blocks",
        env_cls=StackTwoBlocks,
        objects=("block_0", "block_1"),
        success_condition=(
            "One block rests on the table, the other contacts only that block, "
            "and neither gripper holds a block."
        ),
        stage_meaning={
            1: "at least one block has been held",
            2: "the other block has also been held",
            3: "blocks have stacked contact without upper-table contact",
        },
    ),
    "lift_tray": TaskSpec(
        key="lift_tray",
        env_cls=LiftTray,
        objects=("tray",),
        success_condition=(
            "Both grippers hold the tray (one on each long rim) and the tray is "
            "lifted so it no longer touches the table."
        ),
        stage_meaning={
            1: "left gripper holds the tray",
            2: "right gripper holds the tray",
            3: "both hold and the tray is clear of the table and floor",
        },
    ),
    "pack_box": TaskSpec(
        key="pack_box",
        env_cls=PackBox,
        objects=("packing_box",),
        success_condition=(
            "Both lid flaps of the packing box are folded closed over the box "
            "(each hinge within 10% of its closed limit). The right arm closes "
            "the flap on its side, the left arm the other."
        ),
        stage_meaning={
            1: "left gripper touched the left flap",
            2: "right gripper touched the right flap",
            3: "right flap is closed",
            4: "left flap is closed",
            5: "both flaps are closed",
        },
    ),
    "pick_single_book": TaskSpec(
        key="pick_single_book",
        env_cls=PickSingleBookFromTable,
        objects=("book", "lower_shelf", "upper_shelf"),
        success_condition=(
            "A gripper holds the book with its center at least 0.05 m above its "
            "resting height and the book touches neither the counter nor the floor."
        ),
        stage_meaning={
            1: "a gripper holds the book",
            2: "the held book is clear of the counter and floor",
        },
    ),
    "stack_single_book_shelf": TaskSpec(
        key="stack_single_book_shelf",
        env_cls=StackSingleBookShelf,
        objects=("book", "lower_shelf", "upper_shelf"),
        success_condition=(
            "The book rests in contact with the lower or upper shelf plank and "
            "no gripper holds it."
        ),
        stage_meaning={
            1: "a gripper holds the book",
            2: "the book contacts a shelf plank",
            3: "the book rests on a shelf plank with no gripper holding it",
        },
    ),
    "rotate_valve": TaskSpec(
        key="rotate_valve",
        env_cls=RotateValve,
        objects=("valve_0", "valve_1"),
        success_condition=(
            "Both valve handwheels are turned counterclockwise (seen from above) "
            "past 10% of their travel (about 0.47 rad each). valve_0 is on the "
            "right arm's side, valve_1 on the left arm's side."
        ),
        stage_meaning={
            1: "a gripper holds valve_0's wheel",
            2: "valve_0 is turned past the threshold",
            3: "a gripper holds valve_1's wheel",
            4: "valve_1 is turned past the threshold",
        },
    ),
}

BASE_TASK_KEYS = (
    "cube_handover",
    "lift_pot",
    "stack_two_blocks",
    "vertical_cube_handover",
    "lift_tray",
    "pack_box",
    "pick_single_book",
    "stack_single_book_shelf",
    "rotate_valve",
)


def task_key_from_env(env: Any) -> str:
    for key, spec in TASK_SPECS.items():
        if type(env) is spec.env_cls:
            return key
    for key, spec in TASK_SPECS.items():
        if isinstance(env, spec.env_cls):
            return key
    raise ValueError(f"unsupported Agentic v2 environment: {type(env).__name__}")


def make_task_env(
    task_key: str,
    *,
    render_mode: str | None = None,
    control_frequency: int = 20,
    include_camera: bool = False,
    camera_resolution: tuple[int, int] = (256, 256),
) -> Any:
    """Construct a base task with v2's fixed absolute joint action mode."""

    spec = TASK_SPECS[task_key]
    cameras = []
    if include_camera:
        cameras.append(
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                resolution=camera_resolution,
            )
        )
    return spec.env_cls(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            ee=False,
            floating_dofs=[],
        ),
        render_mode=render_mode,
        control_frequency=control_frequency,
        robot_cls=BimanualPanda,
        observation_config=ObservationConfig(cameras=cameras),
    )
