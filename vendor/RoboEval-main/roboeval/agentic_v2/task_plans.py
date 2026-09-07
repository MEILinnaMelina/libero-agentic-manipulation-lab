"""Task-level fixed semantic plans used to gate the robotics stack."""

from __future__ import annotations

from roboeval.agentic_v2.types import SkillName, SkillRequest


BASE_TASKS = (
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


_FIXED_PLANS: dict[str, tuple[SkillRequest, ...]] = {
    "cube_handover": (
        SkillRequest(
            SkillName.GRASP,
            object_name="cube",
            roles={"right": "grasping_arm"},
            goal="stable_hold",
        ),
        SkillRequest(
            SkillName.HANDOVER,
            object_name="cube",
            goal="transfer_to_other_arm",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "vertical_cube_handover": (
        SkillRequest(
            SkillName.GRASP,
            object_name="cube",
            roles={"right": "grasping_arm"},
            goal="stable_hold",
        ),
        SkillRequest(
            SkillName.HANDOVER,
            object_name="cube",
            goal="transfer_to_other_arm",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "lift_pot": (
        SkillRequest(
            SkillName.BIMANUAL_GRASP,
            object_name="kitchenpot",
            roles={"left": "left_handle", "right": "right_handle"},
            goal="stable_bimanual_hold",
        ),
        SkillRequest(
            SkillName.LIFT,
            object_name="kitchenpot",
            roles={"left": "left_handle", "right": "right_handle"},
            goal="task_success_height",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "lift_tray": (
        SkillRequest(
            SkillName.BIMANUAL_GRASP,
            object_name="tray",
            roles={"left": "left_rim", "right": "right_rim"},
            goal="stable_bimanual_hold",
        ),
        SkillRequest(
            SkillName.LIFT,
            object_name="tray",
            roles={"left": "left_rim", "right": "right_rim"},
            goal="clear_table",
            strategy="clear_table",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "stack_two_blocks": (
        SkillRequest(
            SkillName.GRASP,
            object_name="block_0",
            goal="stable_hold",
            strategy="nearest_arm",
        ),
        SkillRequest(
            SkillName.HANDOVER,
            object_name="block_0",
            goal="transfer_for_opposite_workspace",
        ),
        SkillRequest(
            SkillName.PLACE,
            object_name="block_0",
            goal="on:block_1",
            strategy="stack",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "pack_box": (
        SkillRequest(
            SkillName.CLOSE_FLAP,
            object_name="packing_box",
            roles={"right": "right_flap"},
            goal="closed",
        ),
        SkillRequest(
            SkillName.CLOSE_FLAP,
            object_name="packing_box",
            roles={"left": "left_flap"},
            goal="closed",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "pick_single_book": (
        SkillRequest(
            SkillName.GRASP,
            object_name="book",
            goal="stable_hold",
            strategy="nearest_arm",
        ),
        SkillRequest(
            SkillName.LIFT,
            object_name="book",
            goal="clear_table",
            strategy="clear_table",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "stack_single_book_shelf": (
        SkillRequest(
            SkillName.GRASP,
            object_name="book",
            goal="stable_hold",
            strategy="nearest_arm",
        ),
        SkillRequest(
            SkillName.PLACE,
            object_name="book",
            goal="on:lower_shelf",
        ),
        SkillRequest(SkillName.FINISH),
    ),
    "rotate_valve": (
        SkillRequest(
            SkillName.ROTATE,
            object_name="valve_0",
            roles={"right": "turning_arm"},
            goal="rotated",
        ),
        SkillRequest(
            SkillName.ROTATE,
            object_name="valve_1",
            roles={"left": "turning_arm"},
            goal="rotated",
        ),
        SkillRequest(SkillName.FINISH),
    ),
}


def fixed_plan(task_key: str) -> tuple[SkillRequest, ...]:
    """Return a pose-free, timing-free fixed plan for one base task."""

    try:
        return _FIXED_PLANS[task_key]
    except KeyError as error:
        raise ValueError(
            f"Agentic v2 supports only {', '.join(BASE_TASKS)}; got {task_key!r}"
        ) from error
