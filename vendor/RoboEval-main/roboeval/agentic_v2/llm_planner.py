"""OpenAI/Anthropic clients and strict semantic response validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import time
from time import perf_counter
from typing import Any, Mapping, Protocol

from roboeval.agentic_v2.prompts import REQUEST_JSON_SCHEMA, SCHEMA_VERSION
from roboeval.agentic_v2.types import SkillRequest, to_jsonable


@dataclass(frozen=True)
class PlannerDecision:
    thought: str
    request: SkillRequest
    provider: str
    model: str
    raw_response: str = ""
    latency_seconds: float = 0.0
    usage: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""
    is_replan: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


class TextPlannerClient(Protocol):
    provider: str
    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, Mapping[str, Any]]:
        ...


def parse_semantic_response(
    raw_response: str,
    *,
    provider: str,
    model: str,
    latency_seconds: float = 0.0,
    usage: Mapping[str, Any] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
    is_replan: bool = False,
) -> PlannerDecision:
    """Parse exactly one versioned request and reject every extra field."""

    value = json.loads(raw_response.strip())
    if not isinstance(value, dict):
        raise ValueError("planner response must be one JSON object")
    unknown = set(value) - {"schema_version", "thought", "request"}
    missing = {"schema_version", "thought", "request"} - set(value)
    if unknown or missing:
        raise ValueError(
            f"planner response keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported planner schema {value['schema_version']!r}")
    if not isinstance(value["thought"], str):
        raise ValueError("thought must be a string")
    if not isinstance(value["request"], dict):
        raise ValueError("request must be an object")
    request_value = dict(value["request"])
    roles = request_value.get("roles") or {}
    request_value["roles"] = {
        key: role for key, role in roles.items() if role is not None
    }
    request = SkillRequest.from_dict(request_value)
    return PlannerDecision(
        thought=value["thought"].strip(),
        request=request,
        provider=provider,
        model=model,
        raw_response=raw_response,
        latency_seconds=float(latency_seconds),
        usage=dict(usage or {}),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        is_replan=is_replan,
    )


def _usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return to_jsonable(value)
    result = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


class OpenAITextPlanner:
    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        reasoning_effort: str | None = "low",
        max_output_tokens: int = 800,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-5.6-terra"
        self.client = OpenAI()
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, Mapping[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "roboeval_semantic_skill",
                    "strict": True,
                    "schema": REQUEST_JSON_SCHEMA,
                }
            },
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)
        return response.output_text, _usage_dict(getattr(response, "usage", None))


class AnthropicTextPlanner:
    provider = "anthropic"

    def __init__(self, model: str | None = None, *, max_output_tokens: int = 800) -> None:
        from anthropic import Anthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL") or ""
        if not self.model:
            raise ValueError("set --model or ANTHROPIC_MODEL")
        self.client = Anthropic()
        self.max_output_tokens = int(max_output_tokens)

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, Mapping[str, Any]]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        return raw, _usage_dict(getattr(response, "usage", None))


_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
        "DeadlineExceededError",
    }
)


def _is_retryable(error: Exception) -> bool:
    # Both openai and anthropic raise similarly-named exceptions for
    # transient network/server problems; match by name instead of importing
    # both SDKs unconditionally (each client is already lazily imported).
    return type(error).__name__ in _RETRYABLE_ERROR_NAMES


def request_from_client(
    client: TextPlannerClient,
    system_prompt: str,
    user_prompt: str,
    *,
    is_replan: bool = False,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
) -> PlannerDecision:
    started = perf_counter()
    for attempt in range(max_attempts):
        try:
            raw, usage = client.complete(system_prompt, user_prompt)
        except Exception as error:
            # Only retry transient connection/server errors - a malformed
            # or schema-invalid response is real signal about model
            # behavior and should surface immediately, not be masked by
            # retrying into a possibly-identical bad response.
            if not _is_retryable(error) or attempt == max_attempts - 1:
                raise
            time.sleep(backoff_seconds * (2**attempt))
            continue
        return parse_semantic_response(
            raw,
            provider=client.provider,
            model=client.model,
            latency_seconds=perf_counter() - started,
            usage=usage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=is_replan,
        )
    raise AssertionError("unreachable: loop always returns or raises")
