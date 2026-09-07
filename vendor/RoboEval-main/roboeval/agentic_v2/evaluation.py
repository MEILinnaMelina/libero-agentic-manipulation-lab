"""Reproducible Phase 11 report aggregation and comparison helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from roboeval.agentic_v2.artifacts import write_json


METHOD_SPECS: dict[str, dict[str, Any]] = {
    "v1-p22-independent": {
        "commit": "fb3876d",
        "feasibility_gate": "none",
        "semantic_llm": True,
        "online_replan": "text_feedback_only",
        "cross_trial_memory": False,
        "runner": "historical_worktree",
    },
    "v1-p23-memory": {
        "commit": "9400dda",
        "feasibility_gate": "none",
        "semantic_llm": True,
        "online_replan": "text_feedback_only",
        "cross_trial_memory": True,
        "runner": "historical_worktree",
    },
    "v2-ik-only": {
        "feasibility_gate": "ik_only",
        "semantic_llm": True,
        "online_replan": True,
        "cross_trial_memory": False,
        "runner": "v2",
    },
    "v2-fixed": {
        "feasibility_gate": "full",
        "semantic_llm": False,
        "online_replan": False,
        "cross_trial_memory": False,
        "runner": "v2",
    },
    "v2-full-no-replan": {
        "feasibility_gate": "full",
        "semantic_llm": True,
        "online_replan": False,
        "cross_trial_memory": False,
        "runner": "v2",
    },
    "v2-full": {
        "feasibility_gate": "full",
        "semantic_llm": True,
        "online_replan": True,
        "cross_trial_memory": False,
        "runner": "v2",
    },
    "v2-full-memory": {
        "feasibility_gate": "full",
        "semantic_llm": True,
        "online_replan": True,
        "cross_trial_memory": True,
        "runner": "v2",
    },
}

LAUNCHABLE_V2_METHODS = (
    "v2-ik-only",
    "v2-fixed",
    "v2-full-no-replan",
    "v2-full",
    "v2-full-memory",
)

LAUNCHABLE_METHODS = tuple(METHOD_SPECS)


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_trial_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_report_path"] = str(path.resolve())
        reports.append(value)
    return reports


def aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for report in reports:
        groups.setdefault(
            (str(report["method"]), str(report["task_key"])), []
        ).append(report)
    rows = []
    for (method, task), values in sorted(groups.items()):
        successes = sum(
            float(item.get("benchmark_success", 0.0)) >= 1.0 for item in values
        )
        low, high = wilson_interval(successes, len(values))
        rows.append(
            {
                "method": method,
                "task_key": task,
                "trials": len(values),
                "successes": successes,
                "success_rate": successes / len(values),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "mean_subtask_progress": _mean(values, "subtask_progress"),
                "mean_env_collision_count": _metric_mean(
                    values, "env_collision_count"
                ),
                "mean_self_collision_count": _metric_mean(
                    values, "self_collision_count"
                ),
                "mean_slip_count": _metric_mean(values, "slip_count"),
                "mean_path_length": _first_metric_mean(
                    values,
                    (
                        "total_cartesian_path_length",
                        "avg_cartesian_path_length",
                    ),
                ),
                "mean_execution_time_s": _metric_mean(
                    values, "trial_elapsed_seconds"
                ),
                "mean_llm_planning_time_s": _metric_mean(
                    values, "llm_planning_seconds"
                ),
                "mean_llm_calls": _metric_mean(values, "llm_calls"),
                "mean_llm_cost_usd": _metric_mean(values, "llm_cost_usd"),
                "mean_replans": _metric_mean(values, "replan_count"),
                "behavior_quality_rate": mean(
                    bool(item.get("behavior_quality", {}).get("passed", False))
                    for item in values
                ),
                "terminal_failures": _failure_histogram(values),
            }
        )
    return rows


def paired_comparison(
    reports: Sequence[Mapping[str, Any]],
    baseline: str = "v1-p22-independent",
    method: str = "v2-full",
) -> list[dict[str, Any]]:
    indexed = {
        (str(item["method"]), str(item["task_key"]), int(item["seed"])): item
        for item in reports
    }
    rows = []
    tasks = sorted({str(item["task_key"]) for item in reports})
    for task in tasks:
        pairs = []
        for seed in range(10):
            first = indexed.get((baseline, task, seed))
            second = indexed.get((method, task, seed))
            if first is None or second is None:
                continue
            first_success = float(first.get("benchmark_success", 0.0)) >= 1.0
            second_success = float(second.get("benchmark_success", 0.0)) >= 1.0
            pairs.append(
                {
                    "seed": seed,
                    "baseline_success": first_success,
                    "method_success": second_success,
                    "delta": int(second_success) - int(first_success),
                }
            )
        rows.append(
            {
                "task_key": task,
                "baseline": baseline,
                "method": method,
                "paired_seeds": len(pairs),
                "mean_success_delta": (
                    mean(item["delta"] for item in pairs) if pairs else None
                ),
                "pairs": pairs,
            }
        )
    return rows


def write_evaluation_outputs(
    output_dir: Path,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trials.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
    rows = aggregate_reports(reports)
    csv_path = output_dir / "aggregate.csv"
    fields = [key for key in rows[0] if key != "terminal_failures"] if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
    summary_path = output_dir / "evaluation_summary.json"
    write_json(
        summary_path,
        {
            "methods": METHOD_SPECS,
            "aggregate": rows,
            "primary_paired_comparison": paired_comparison(reports),
        },
    )
    return {
        "jsonl": str(jsonl_path.resolve()),
        "aggregate_csv": str(csv_path.resolve()),
        "summary": str(summary_path.resolve()),
    }


def _mean(
    values: Sequence[Mapping[str, Any]],
    key: str,
) -> float | None:
    present = [
        float(item[key])
        for item in values
        if isinstance(item.get(key), (int, float))
    ]
    return mean(present) if present else None


def _metric_mean(
    values: Sequence[Mapping[str, Any]],
    key: str,
) -> float | None:
    present = [
        float(item.get("metrics", {})[key])
        for item in values
        if isinstance(item.get("metrics", {}).get(key), (int, float))
    ]
    return mean(present) if present else None


def _first_metric_mean(
    values: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> float | None:
    present = []
    for item in values:
        metrics = item.get("metrics", {})
        for key in keys:
            if isinstance(metrics.get(key), (int, float)):
                present.append(float(metrics[key]))
                break
    return mean(present) if present else None


def _failure_histogram(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in values:
        key = str(item.get("failure_code") or "NONE")
        result[key] = result.get(key, 0) + 1
    return result
