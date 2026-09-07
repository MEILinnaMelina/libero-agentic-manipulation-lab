"""Fixed and online semantic planners for the observe-decide-execute loop."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from roboeval.agentic_v2.llm_planner import (
    PlannerDecision,
    TextPlannerClient,
    request_from_client,
)
from roboeval.agentic_v2.prompts import build_planner_prompts
from roboeval.agentic_v2.task_plans import fixed_plan
from roboeval.agentic_v2.task_specs import TaskSpec
from roboeval.agentic_v2.types import (
    SceneState,
    SkillName,
    SkillRequest,
    SkillResult,
    to_jsonable,
)


class SemanticPlanner(Protocol):
    provider: str
    model: str
    decisions: list[PlannerDecision]

    def next_request(
        self,
        spec: TaskSpec,
        state: SceneState,
        last_result: SkillResult | None,
    ) -> PlannerDecision:
        ...

    def record_result(self, result: SkillResult) -> None:
        ...


class FixedSemanticPlanner:
    provider = "deterministic"
    model = "fixed-semantic-v1"

    def __init__(self, task_key: str) -> None:
        self.plan = fixed_plan(task_key)
        self.index = 0
        self.decisions: list[PlannerDecision] = []

    def next_request(self, spec, state, last_result) -> PlannerDecision:
        request = (
            self.plan[self.index]
            if self.index < len(self.plan)
            else SkillRequest(SkillName.FINISH)
        )
        self.index += 1
        decision = PlannerDecision(
            thought="Execute the next fixed semantic gate step.",
            request=request,
            provider=self.provider,
            model=self.model,
        )
        self.decisions.append(decision)
        return decision

    def record_result(self, result: SkillResult) -> None:
        return None


class OnlineReplanner:
    """Re-observe after every skill and ask the provider for one semantic action."""

    def __init__(
        self,
        client: TextPlannerClient,
        *,
        allow_failure_replan: bool = True,
        memory_notes: Sequence[str] = (),
    ) -> None:
        self.client = client
        self.provider = client.provider
        self.model = client.model
        self.allow_failure_replan = bool(allow_failure_replan)
        self.memory_notes = tuple(memory_notes)
        self.decisions: list[PlannerDecision] = []
        self.history: list[dict[str, Any]] = []

    def next_request(self, spec, state, last_result) -> PlannerDecision:
        if (
            last_result is not None
            and not last_result.success
            and not self.allow_failure_replan
        ):
            decision = PlannerDecision(
                thought="Failure replanning is disabled for this ablation.",
                request=SkillRequest(SkillName.FINISH),
                provider=self.provider,
                model=self.model,
            )
            self.decisions.append(decision)
            return decision
        is_replan = bool(last_result is not None and not last_result.success)
        system, user = build_planner_prompts(
            spec,
            state,
            history=self.history,
            last_result=last_result,
            memory_notes=self.memory_notes,
        )
        decision = request_from_client(
            self.client, system, user, is_replan=is_replan
        )
        self.decisions.append(decision)
        return decision

    def record_result(self, result: SkillResult) -> None:
        decision = self.decisions[-1]
        self.history.append(
            {
                "request": decision.request.to_dict(),
                "thought": decision.thought,
                "result": {
                    "success": result.success,
                    "message": result.message,
                    "failure_code": (
                        result.failure_code.value if result.failure_code else None
                    ),
                    "diagnostics": to_jsonable(result.diagnostics),
                },
            }
        )
