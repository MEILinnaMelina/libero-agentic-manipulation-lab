"""Run one Agentic v2 trial with a fixed or online semantic planner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from absl import logging as absl_logging

from roboeval.agentic_v2.llm_planner import (
    AnthropicTextPlanner,
    OpenAITextPlanner,
)
from roboeval.agentic_v2.replanner import (
    FixedSemanticPlanner,
    OnlineReplanner,
)
from roboeval.agentic_v2.runner import AgenticV2Runner
from roboeval.agentic_v2.task_specs import BASE_TASK_KEYS, make_task_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=BASE_TASK_KEYS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planner",
        choices=("fixed", "openai", "anthropic"),
        default="fixed",
    )
    parser.add_argument("--model")
    parser.add_argument("--method")
    parser.add_argument(
        "--render",
        choices=("none", "window", "rgb"),
        default="none",
    )
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--frame-every", type=int, default=8)
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--no-replan", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--memory-file", type=Path)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument(
        "--feasibility-gate",
        choices=("full", "ik-only"),
        default="full",
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=_optional_float("OPENAI_INPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=_optional_float("OPENAI_OUTPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "agentic_v2",
    )
    return parser.parse_args()


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def make_semantic_planner(args: argparse.Namespace):
    if args.planner == "fixed":
        return FixedSemanticPlanner(args.task)
    if args.planner == "openai":
        client = OpenAITextPlanner(
            args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        client = AnthropicTextPlanner(
            args.model,
            max_output_tokens=args.max_output_tokens,
        )
    notes = []
    if args.memory:
        if args.memory_file is None:
            raise ValueError("--memory requires --memory-file")
        if args.memory_file.exists():
            notes = json.loads(args.memory_file.read_text(encoding="utf-8"))
    return OnlineReplanner(
        client,
        allow_failure_replan=not args.no_replan,
        memory_notes=notes,
    )


def main() -> None:
    absl_logging.set_verbosity(absl_logging.ERROR)
    args = parse_args()
    method = args.method or (
        "v2-fixed"
        if args.planner == "fixed"
        else "v2-ik-only"
        if args.feasibility_gate == "ik-only"
        else "v2-full-no-replan"
        if args.no_replan
        else "v2-full-memory"
        if args.memory
        else "v2-full"
    )
    trial_dir = (
        args.output_dir / method / args.task / f"seed_{args.seed:03d}"
    )
    include_camera = args.record_gif or args.render == "rgb"
    env = make_task_env(
        args.task,
        render_mode=(
            "human"
            if args.render == "window"
            else "rgb_array"
            if include_camera
            else None
        ),
        include_camera=include_camera,
    )
    try:
        env.reset(seed=args.seed)
        planner = make_semantic_planner(args)
        report = AgenticV2Runner(
            env,
            planner,
            method=method,
            output_dir=trial_dir,
            render=args.render == "window",
            record_gif=args.record_gif,
            frame_every=args.frame_every,
            feasibility_gate=args.feasibility_gate,
        ).run(
            max_skills=args.max_skills,
            run_config={
                "planner": args.planner,
                "model": getattr(planner, "model", None),
                "render": args.render,
                "record_gif": args.record_gif,
                "max_skills": args.max_skills,
                # FixedSemanticPlanner never replans regardless of --no-replan;
                # record that truthfully instead of implying it does.
                "online_replan": args.planner != "fixed" and not args.no_replan,
                "cross_trial_memory": args.memory,
                "pricing_usd_per_million": {
                    "input": args.input_cost_per_million,
                    "output": args.output_cost_per_million,
                },
            },
        )
    finally:
        env.close()
    print(
        json.dumps(
            {
                "task": report.task_key,
                "seed": report.seed,
                "method": report.method,
                "benchmark_success": report.benchmark_success,
                "subtask_progress": report.subtask_progress,
                "behavior_quality": report.behavior_quality,
                "failure_code": (
                    report.failure_code.value if report.failure_code else None
                ),
                "report": str((trial_dir / "trial_report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
