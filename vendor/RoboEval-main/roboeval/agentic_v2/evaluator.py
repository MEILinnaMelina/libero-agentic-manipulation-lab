"""Evaluation helpers that never redefine RoboEval's raw task success."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from roboeval.agentic_v2.types import FailureCode, SceneState, SkillResult, TrialReport


def benchmark_success(state: SceneState) -> float:
    return float(state.metrics.get("task_success", state.metrics.get("success", 0.0)) or 0.0)


def subtask_progress(state: SceneState) -> float:
    return float(state.metrics.get("subtask_progress", 0.0) or 0.0)


def assess_behavior_quality(state: SceneState) -> dict[str, Any]:
    """Return transparent diagnostics, separate from benchmark success."""

    metrics = state.metrics
    checks = {
        "no_self_collision": int(metrics.get("self_collision_count", 0) or 0) == 0,
        "no_environment_collision": int(metrics.get("env_collision_count", 0) or 0) == 0,
        "no_slip": int(metrics.get("slip_count", 0) or 0) == 0,
        "finite_robot_state": bool(
            np.all(np.isfinite(state.robot.joint_positions))
            and np.all(np.isfinite(state.robot.joint_velocities))
        ),
    }
    if state.task_key == "lift_pot":
        checks["both_grippers_hold"] = set(state.objects["kitchenpot"].held_by) == {"left", "right"}
    elif state.task_key == "cube_handover":
        # "exactly one holder" alone is trivially true if the donor simply
        # never let go - it doesn't prove a transfer happened. RoboEval's
        # own stage tracking (task_stage_reached[2] = "opposite gripper has
        # held the rod") is ground truth for whether a handover actually
        # occurred at some point in the episode.
        stage_reached = metrics.get("task_stage_reached") or {}
        checks["reached_transfer_stage"] = bool(
            stage_reached.get("2") or stage_reached.get(2)
        )
        checks["exactly_one_final_holder"] = len(state.objects["cube"].held_by) == 1
    elif state.task_key == "stack_two_blocks":
        checks["both_grippers_released"] = not any(
            obj.held_by for obj in state.objects.values()
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def build_trial_report(
    *,
    task_key: str,
    seed: int,
    method: str,
    state: SceneState,
    skill_results: Sequence[SkillResult],
    failure_code: FailureCode | None = None,
    artifacts: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> TrialReport:
    metrics = dict(state.metrics)
    metrics.update(dict(extra_metrics or {}))
    return TrialReport(
        task_key=task_key,
        seed=seed,
        method=method,
        benchmark_success=benchmark_success(state),
        subtask_progress=subtask_progress(state),
        behavior_quality=assess_behavior_quality(state),
        skill_results=tuple(skill_results),
        final_state=state,
        failure_code=failure_code,
        metrics=metrics,
        artifacts=dict(artifacts or {}),
        metadata=dict(metadata or {}),
    )
