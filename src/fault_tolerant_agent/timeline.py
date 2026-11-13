"""Records every real model/tool attempt so retries, fallbacks and refusals are visible.

``TimelineMiddleware`` is listed LAST in the middleware stack. Because wrap-style hooks nest
(first listed = outermost), it sits innermost and therefore sees every individual attempt that
the retry / fallback layers make, not just the final outcome.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import AgentMiddleware


def model_label(model: Any) -> str:
    for attr in ("tag", "deployment_name", "model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


@dataclass
class Event:
    t: float
    kind: str  # "model" | "tool"
    name: str
    outcome: str  # "ok" | "refusal" | "error:<ExceptionType>"
    ms: float
    key: str = ""  # tool_call id for tools (same id => retry of the same call)


@dataclass
class Timeline:
    events: list[Event] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def record(self, kind: str, name: str, outcome: str, ms: float, key: str = "") -> None:
        self.events.append(Event(time.monotonic() - self.started, kind, name, outcome, ms, key or name))

    def render(self) -> str:
        return "\n".join(
            f"t+{e.t:6.2f}s  {e.kind.upper():5} {e.name:<22} {e.outcome:<28} {e.ms:7.1f} ms"
            for e in self.events
        )

    def summary(self) -> dict[str, int]:
        """Counts attempts, retries (same target retried after a failure) and fallbacks."""
        out = {"model_attempts": 0, "model_retries": 0, "fallbacks": 0, "refusals": 0,
               "model_errors": 0, "tool_attempts": 0, "tool_retries": 0, "tool_errors": 0}
        # Tool calls can run in parallel and interleave, so track the previous attempt per call id.
        prev: dict[tuple[str, str], Event] = {}
        last_model: Event | None = None
        for e in self.events:
            out[f"{e.kind}_attempts"] += 1
            if e.outcome == "refusal":
                out["refusals"] += 1
            elif e.outcome.startswith("error"):
                out[f"{e.kind}_errors"] += 1
            if e.kind == "tool":
                p = prev.get(("tool", e.key))
                if p is not None and p.outcome != "ok":
                    out["tool_retries"] += 1
                prev[("tool", e.key)] = e
            else:
                p = last_model
                if p is not None and p.outcome != "ok":
                    out["model_retries" if p.name == e.name else "fallbacks"] += 1
                last_model = e
        return out


def _has_refusal(response: Any) -> bool:
    for message in getattr(response, "result", []):
        if any(block.get("type") == "refusal" for block in message.content_blocks):
            return True
    return False


class TimelineMiddleware(AgentMiddleware):
    def __init__(self, timeline: Timeline) -> None:
        super().__init__()
        self.timeline = timeline

    # ---- model ----
    def wrap_model_call(self, request, handler):
        label, t0 = model_label(request.model), time.monotonic()
        try:
            response = handler(request)
        except Exception as exc:
            self.timeline.record("model", label, f"error:{type(exc).__name__}", (time.monotonic() - t0) * 1000)
            raise
        outcome = "refusal" if _has_refusal(response) else "ok"
        self.timeline.record("model", label, outcome, (time.monotonic() - t0) * 1000)
        return response

    async def awrap_model_call(self, request, handler: Callable[..., Awaitable[Any]]):
        label, t0 = model_label(request.model), time.monotonic()
        try:
            response = await handler(request)
        except Exception as exc:
            self.timeline.record("model", label, f"error:{type(exc).__name__}", (time.monotonic() - t0) * 1000)
            raise
        outcome = "refusal" if _has_refusal(response) else "ok"
        self.timeline.record("model", label, outcome, (time.monotonic() - t0) * 1000)
        return response

    # ---- tools ----
    def wrap_tool_call(self, request, handler):
        name, key, t0 = request.tool_call["name"], request.tool_call.get("id") or "", time.monotonic()
        try:
            result = handler(request)
        except Exception as exc:
            self.timeline.record("tool", name, f"error:{type(exc).__name__}", (time.monotonic() - t0) * 1000, key)
            raise
        self.timeline.record("tool", name, "ok", (time.monotonic() - t0) * 1000, key)
        return result

    async def awrap_tool_call(self, request, handler: Callable[..., Awaitable[Any]]):
        name, key, t0 = request.tool_call["name"], request.tool_call.get("id") or "", time.monotonic()
        try:
            result = await handler(request)
        except Exception as exc:
            self.timeline.record("tool", name, f"error:{type(exc).__name__}", (time.monotonic() - t0) * 1000, key)
            raise
        self.timeline.record("tool", name, "ok", (time.monotonic() - t0) * 1000, key)
        return result
