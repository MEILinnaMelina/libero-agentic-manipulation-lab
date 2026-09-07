"""End-to-end deterministic/LLM Agentic v2 trial runner."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

from roboeval.agentic_v2.artifacts import FrameRecorder, write_json
from roboeval.agentic_v2.evaluator import benchmark_success, build_trial_report
from roboeval.agentic_v2.replanner import SemanticPlanner
from roboeval.agentic_v2.skills.base import SkillContext
from roboeval.agentic_v2.skills.registry import SkillRegistry
from roboeval.agentic_v2.state import collect_scene_state
from roboeval.agentic_v2.task_specs import BASE_TASK_KEYS, TASK_SPECS
from roboeval.agentic_v2.types import FailureCode, SkillName, SkillResult, TrialReport


class AgenticV2Runner:
    """Run one fresh episode through the semantic skill boundary."""

    def __init__(
        self,
        env: Any,
        planner: SemanticPlanner,
        *,
        method: str,
        output_dir: Path,
        render: bool = False,
        record_gif: bool = False,
        frame_every: int = 8,
        feasibility_gate: str = "full",
    ) -> None:
        self.env = env
        self.semantic_planner = planner
        self.method = method
        self.output_dir = Path(output_dir)
        self.frames = FrameRecorder(
            env,
            self.output_dir / "frames",
            enabled=record_gif,
            every=frame_every,
        )
        self.context = SkillContext.create(
            env,
            render=render,
            frame_callback=self.frames.callback if record_gif else None,
            feasibility_gate=feasibility_gate,
        )
        self.feasibility_gate = feasibility_gate
        self.registry = SkillRegistry(self.context)

    def run(
        self,
        *,
        max_skills: int = 10,
        run_config: Mapping[str, Any] | None = None,
    ) -> TrialReport:
        started = perf_counter()
        skill_results: list[SkillResult] = []
        terminal_failure: FailureCode | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames.capture("before")
        task_key = collect_scene_state(self.env).task_key
        if task_key not in BASE_TASK_KEYS:
            raise ValueError(f"task {task_key!r} is deferred outside Agentic v2")
        spec = TASK_SPECS[task_key]
        last_result: SkillResult | None = None

        for index in range(max(1, int(max_skills))):
            state = collect_scene_state(self.env)
            if benchmark_success(state) >= 1.0:
                break
            try:
                decision = self.semantic_planner.next_request(spec, state, last_result)
            except Exception as error:
                terminal_failure = FailureCode.INVALID_REQUEST
                write_json(
                    self.output_dir / "planner_error.json",
                    {"type": type(error).__name__, "message": str(error), "step": index},
                )
                break
            if decision.request.skill is SkillName.FINISH:
                if benchmark_success(state) < 1.0:
                    terminal_failure = (
                        last_result.failure_code
                        if last_result is not None and last_result.failure_code
                        else FailureCode.POSTCONDITION_FAILED
                    )
                break

            trace_start = len(self.context.planning_trace)
            try:
                result = self.registry.execute(decision.request)
            except Exception as error:
                result = SkillResult(
                    request=decision.request,
                    success=False,
                    message=f"{type(error).__name__}: {error}",
                    state=collect_scene_state(self.env),
                    failure_code=FailureCode.EXECUTION_DIVERGED,
                    diagnostics={"exception_type": type(error).__name__},
                )
            skill_results.append(result)
            last_result = result
            self.semantic_planner.record_result(result)
            self.frames.capture(f"skill_{index:02d}_{decision.request.skill.value}")
            write_json(
                self.output_dir / "skills" / f"{index:02d}_{decision.request.skill.value}.json",
                {
                    "decision": decision,
                    "state_before": state,
                    "result": result,
                    "planning_trace": self.context.planning_trace[trace_start:],
                },
            )
            if not result.success and not getattr(
                self.semantic_planner, "allow_failure_replan", False
            ):
                terminal_failure = result.failure_code or FailureCode.POSTCONDITION_FAILED
                break
        else:
            terminal_failure = FailureCode.TIMEOUT

        final_state = collect_scene_state(self.env)
        if benchmark_success(final_state) >= 1.0:
            terminal_failure = None
        elif terminal_failure is None:
            terminal_failure = (
                last_result.failure_code
                if last_result is not None and last_result.failure_code
                else FailureCode.POSTCONDITION_FAILED
            )

        self.frames.capture("after")
        gif = self.frames.write_gif(self.output_dir / "trajectory.gif")
        planner_trace = [
            decision.to_dict() for decision in self.semantic_planner.decisions
        ]
        planning_path = write_json(
            self.output_dir / "planning_trace.json", self.context.planning_trace
        )
        planner_path = write_json(
            self.output_dir / "planner_trace.json", planner_trace
        )
        config = dict(run_config or {})
        runtime_metrics = self._runtime_metrics(
            perf_counter() - started,
            config.get("pricing_usd_per_million"),
        )
        config_path = write_json(
            self.output_dir / "run_config.json",
            {
                **config,
                "task": task_key,
                "seed": final_state.seed,
                "method": self.method,
                "feasibility_gate": self.feasibility_gate,
                "environment": environment_metadata(),
            },
        )
        artifacts = {
            "planning_trace": planning_path,
            "planner_trace": planner_path,
            "run_config": config_path,
        }
        if gif:
            artifacts["trajectory_gif"] = gif
        report = build_trial_report(
            task_key=task_key,
            seed=final_state.seed,
            method=self.method,
            state=final_state,
            skill_results=skill_results,
            failure_code=terminal_failure,
            artifacts=artifacts,
            metadata={
                "planner_provider": self.semantic_planner.provider,
                "planner_model": self.semantic_planner.model,
                "schema": "roboeval.agentic_v2.skill_request.v1",
                "runtime": runtime_metrics,
            },
            extra_metrics=runtime_metrics,
        )
        write_json(self.output_dir / "trial_report.json", report)
        return report

    def _runtime_metrics(
        self,
        elapsed_seconds: float,
        pricing: Any = None,
    ) -> dict[str, Any]:
        decisions = self.semantic_planner.decisions
        usage: dict[str, float] = {}
        for decision in decisions:
            for key, value in decision.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0.0) + float(value)
        event_counts: dict[str, int] = {}
        executed_points = 0
        for trace in self.context.planning_trace:
            execution = trace.get("execution")
            if not execution:
                continue
            executed_points += int(execution.get("executed_points", 0))
            for event in execution.get("events", ()):
                code = str(event.get("code"))
                event_counts[code] = event_counts.get(code, 0) + 1
        input_tokens = usage.get("input_tokens", 0.0)
        output_tokens = usage.get("output_tokens", 0.0)
        cost = None
        if isinstance(pricing, Mapping):
            input_rate = pricing.get("input")
            output_rate = pricing.get("output")
            if isinstance(input_rate, (int, float)) and isinstance(
                output_rate, (int, float)
            ):
                cost = (
                    input_tokens * float(input_rate)
                    + output_tokens * float(output_rate)
                ) / 1_000_000.0
        return {
            "trial_elapsed_seconds": elapsed_seconds,
            "llm_planning_seconds": sum(item.latency_seconds for item in decisions),
            "deterministic_pipeline_seconds": sum(
                float(item.get("elapsed_seconds", 0.0))
                for item in self.context.planning_trace
            ),
            "llm_calls": sum(1 for item in decisions if item.provider != "deterministic"),
            "llm_usage": usage,
            "llm_cost_usd": cost,
            "replan_count": sum(1 for item in decisions if item.is_replan),
            "motion_plans": len(self.context.planning_trace),
            "executed_trajectory_points": executed_points,
            "monitor_failure_counts": event_counts,
        }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("roboeval", "numpy", "mujoco", "gymnasium", "openai", "anthropic"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # A clean git_commit SHA is misleading on its own: uncommitted local
        # edits at run time would make the recorded commit look correct
        # while not actually matching what ran.
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
