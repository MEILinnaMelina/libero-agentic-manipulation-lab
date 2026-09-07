"""Versioned semantic prompt schema for online LLM planning."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from roboeval.agentic_v2.task_specs import TaskSpec
from roboeval.agentic_v2.types import SceneState, SkillName, SkillResult, to_jsonable


SCHEMA_VERSION = "roboeval.agentic_v2.skill_request.v1"

SKILL_PRECONDITIONS = {
    SkillName.GRASP.value: "The named object exists and is not already held by the chosen arm.",
    SkillName.BIMANUAL_GRASP.value: "The named object exists and supports a two-arm grasp.",
    SkillName.LIFT.value: "The named object is held by one or both arms.",
    SkillName.TRANSPORT.value: "The named object is held; goal is a symbolic region.",
    SkillName.HANDOVER.value: "Exactly one arm currently holds the named object.",
    SkillName.PLACE.value: "The named object is held and goal names a support as on:<object> (shelf planks are valid supports).",
    SkillName.CLOSE_FLAP.value: "The named object is an open box with hinged lid flaps; roles name which arm closes the flap on its own side (one flap per request).",
    SkillName.ROTATE.value: "The named object is a valve handwheel; the chosen arm grips it from above and turns it counterclockwise until the task threshold.",
    SkillName.FINISH.value: "Use only when raw benchmark success is already observed or no recovery remains.",
}

REQUEST_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "thought", "request"],
    "properties": {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "thought": {"type": "string"},
        "request": {
            "type": "object",
            "additionalProperties": False,
            "required": ["skill", "object", "roles", "goal", "strategy"],
            "properties": {
                "skill": {"type": "string", "enum": [item.value for item in SkillName]},
                "object": {"type": ["string", "null"]},
                "roles": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["left", "right", "donor", "receiver"],
                    "properties": {
                        "left": {"type": ["string", "null"]},
                        "right": {"type": ["string", "null"]},
                        "donor": {"type": ["string", "null"]},
                        "receiver": {"type": ["string", "null"]},
                    },
                },
                "goal": {"type": "string"},
                "strategy": {"type": ["string", "null"]},
            },
        },
    },
}


def compact_state(state: SceneState) -> dict[str, Any]:
    """Keep decision-relevant typed state while avoiding simulator internals."""

    return {
        "task_key": state.task_key,
        "seed": state.seed,
        "metrics": to_jsonable(state.metrics),
        "robot": {
            "joint_positions": list(state.robot.joint_positions),
            "joint_velocities": list(state.robot.joint_velocities),
            "arms": {
                side: {
                    "ee_pose": to_jsonable(arm.ee_pose),
                    "gripper_aperture_m": arm.gripper_aperture_m,
                    "holding": list(arm.holding),
                }
                for side, arm in state.robot.arms.items()
            },
        },
        "objects": {
            name: {
                "pose": to_jsonable(obj.pose),
                "size": list(obj.size),
                "linear_velocity": list(obj.linear_velocity),
                "angular_velocity": list(obj.angular_velocity),
                "contacts": list(obj.contacts),
                "held_by": list(obj.held_by),
                "fixed": obj.fixed,
            }
            for name, obj in state.objects.items()
        },
    }


def build_planner_prompts(
    spec: TaskSpec,
    state: SceneState,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    last_result: SkillResult | None = None,
    memory_notes: Sequence[str] = (),
) -> tuple[str, str]:
    """Build one observe-decide prompt with no low-level action authority."""

    system = (
        "You are the semantic task planner for a bimanual RoboEval agent. "
        "Choose exactly one skill request. Deterministic robotics code alone "
        "chooses poses, offsets, joint values, trajectories, gains, tolerances, "
        "and step counts. Never output any of those low-level fields. Return "
        "only one JSON object matching the supplied schema. Re-observe failures, "
        "change semantic strategy when useful, and never declare success unless "
        "the raw benchmark metric supports it."
    )
    last = None
    if last_result is not None:
        last = {
            "request": last_result.request.to_dict(),
            "success": last_result.success,
            "message": last_result.message,
            "failure_code": (
                last_result.failure_code.value if last_result.failure_code else None
            ),
            "diagnostics": to_jsonable(last_result.diagnostics),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "key": spec.key,
            "success_condition": spec.success_condition,
            "stage_meaning": spec.stage_meaning,
        },
        "available_skills": SKILL_PRECONDITIONS,
        "output_schema": REQUEST_JSON_SCHEMA,
        "current_state": compact_state(state),
        "last_result": last,
        "within_trial_history": list(history)[-6:],
        "cross_trial_memory": list(memory_notes),
        "rules": [
            "Use only named objects present in current_state.",
            "Roles may identify left/right or donor/receiver, but the live hold state is authoritative.",
            "Use symbolic goals such as stable_hold, handover_region, task_success_height, clear_table, closed, rotated, or on:<object>.",
            "Objects marked fixed are scene fixtures: they can be placement supports but can never be grasped, lifted, or moved.",
            "Do not repeat an unchanged failed request without a semantic recovery reason.",
        ],
    }
    return system, json.dumps(payload, separators=(",", ":"), sort_keys=True)
