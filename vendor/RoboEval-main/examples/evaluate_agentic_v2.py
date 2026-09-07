"""Launch isolated Agentic v2 trials or aggregate saved reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from roboeval.agentic_v2.artifacts import write_json
from roboeval.agentic_v2.evaluation import (
    LAUNCHABLE_METHODS,
    LAUNCHABLE_V2_METHODS,
    METHOD_SPECS,
    load_trial_reports,
    write_evaluation_outputs,
)
from roboeval.agentic_v2.task_specs import BASE_TASK_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=LAUNCHABLE_METHODS,
        default=list(LAUNCHABLE_V2_METHODS),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=BASE_TASK_KEYS,
        default=list(BASE_TASK_KEYS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--v1-p22-root", type=Path)
    parser.add_argument("--v1-p23-root", type=Path)
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "agentic_v2_eval",
    )
    return parser.parse_args()


def command_for(
    args: argparse.Namespace,
    method: str,
    task: str,
    seed: int,
    memory_file: Path,
) -> list[str]:
    if method.startswith("v1-"):
        source_root = (
            args.v1_p22_root
            if method == "v1-p22-independent"
            else args.v1_p23_root
        )
        if source_root is None:
            option = (
                "--v1-p22-root"
                if method == "v1-p22-independent"
                else "--v1-p23-root"
            )
            raise ValueError(f"{method} requires {option}")
        command = [
            sys.executable,
            "examples/run_agentic_v1_baseline.py",
            "--method",
            method,
            "--source-root",
            str(source_root),
            "--task",
            task,
            "--seed",
            str(seed),
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--max-steps",
            str(args.max_skills),
            "--output-dir",
            str(
                args.output_dir
                / method
                / task
                / f"seed_{seed:03d}"
            ),
        ]
        if method == "v1-p23-memory":
            command.extend(["--memory-file", str(memory_file)])
        if args.record_gif:
            command.append("--record-gif")
        return command
    planner = "fixed" if method == "v2-fixed" else args.provider
    command = [
        sys.executable,
        "examples/run_agentic_v2.py",
        "--task",
        task,
        "--seed",
        str(seed),
        "--planner",
        planner,
        "--method",
        method,
        "--max-skills",
        str(args.max_skills),
        "--output-dir",
        str(args.output_dir),
    ]
    if planner != "fixed":
        command.extend(["--model", args.model])
    if method == "v2-full-no-replan":
        command.append("--no-replan")
    if method == "v2-ik-only":
        command.extend(["--feasibility-gate", "ik-only"])
    if method == "v2-full-memory":
        command.extend(
            ["--memory", "--memory-file", str(memory_file)]
        )
    if args.record_gif:
        command.append("--record-gif")
    return command


def update_memory(path: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    notes = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else []
    )
    requests = [
        result["request"]
        for result in report.get("skill_results", [])
        if result.get("success")
    ]
    memory_note = report.get("metadata", {}).get("memory_note")
    notes.append(
        memory_note
        or json.dumps(
            {
                "seed": report["seed"],
                "benchmark_success": report["benchmark_success"],
                "failure_code": report.get("failure_code"),
                "successful_requests": requests,
            },
            separators=(",", ":"),
        )
    )
    write_json(path, notes[-10:])


def launch(args: argparse.Namespace) -> None:
    manifest = []
    for method in args.methods:
        for task in args.tasks:
            memory_file = (
                args.output_dir / "memory" / method / f"{task}.json"
            )
            if method in {"v1-p23-memory", "v2-full-memory"}:
                write_json(memory_file, [])
            for seed in args.seeds:
                command = command_for(
                    args,
                    method,
                    task,
                    seed,
                    memory_file,
                )
                entry = {
                    "method": method,
                    "task": task,
                    "seed": seed,
                    "command": command,
                }
                manifest.append(entry)
                write_json(
                    args.output_dir / "execution_manifest.json",
                    {
                        "method_specs": METHOD_SPECS,
                        "trials": manifest,
                    },
                )
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                )
                entry["return_code"] = completed.returncode
                report_path = (
                    args.output_dir
                    / method
                    / task
                    / f"seed_{seed:03d}"
                    / "trial_report.json"
                )
                if method in {"v1-p23-memory", "v2-full-memory"} and report_path.exists():
                    update_memory(memory_file, report_path)
    write_json(
        args.output_dir / "execution_manifest.json",
        {"method_specs": METHOD_SPECS, "trials": manifest},
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
        launch(args)
    else:
        write_json(
            args.output_dir / "execution_manifest.json",
            {
                "method_specs": METHOD_SPECS,
                "launchable_methods": list(LAUNCHABLE_METHODS),
                "tasks": args.tasks,
                "seeds": args.seeds,
                "note": "Pass --launch to incur simulator/API work.",
            },
        )
    report_paths = sorted(
        args.output_dir.glob("*/**/trial_report.json")
    )
    reports = load_trial_reports(report_paths)
    outputs = write_evaluation_outputs(args.output_dir, reports)
    print(json.dumps({"trials": len(reports), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
