"""Agent factory: the resilience middleware stack, in a deliberate order.

Wrap-style hooks nest: the FIRST middleware listed is the OUTERMOST. before_* hooks run
first->last, after_* hooks last->first.

    ModelCallLimit      budget: counts agent model steps (not retries/fallbacks)
    ToolCallLimit       budget: caps tool executions
    Summarization       context: compacts history before the model call
    ReplyOnRefusal      (optional) turns "every model refused" into a normal reply
    ModelFallback       outage: next model when the inner stack raises
      ModelRetry        transient: retries the SAME model with backoff, then raises (on_failure="error")
        ModelRefusal    refusal -> ModelRefusalError (listed after Fallback so Fallback can catch it)
    ToolError           LLM-recoverable tool exceptions -> error ToolMessage; others propagate
      ToolRetry         transient tool failures (inner, on_failure="error" so ToolError sees the final one)
    TimelineMiddleware  innermost: records every real attempt
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRefusalMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage

from .config import Settings
from .timeline import Timeline, TimelineMiddleware

Profile = Literal["minimal", "resilient", "full"]

TRANSIENT_TOOL_ERRORS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)

SYSTEM_PROMPT = (
    "You are a concise assistant with tools for currency rates, discounts and reports. "
    "If a tool reports an error, correct your input and try again, or explain the problem."
)


def tool_error_message(exc: Exception, request: ToolCallRequest) -> str | None:
    """LLM-recoverable errors become an error ToolMessage; everything else propagates (None).

    Return the exception *type*, not the raw message, so internals do not leak to the model.
    """
    if isinstance(exc, ValueError):
        return f"Tool `{request.tool_call['name']}` failed: {type(exc).__name__}. Fix the input and retry."
    return None


@wrap_model_call
def reply_on_refusal(request, handler):
    """Outermost model wrapper: if every model refuses, answer instead of failing the run."""
    try:
        return handler(request)
    except ModelRefusalError:
        from langchain.agents.middleware import ModelResponse

        return ModelResponse(result=[AIMessage("I can't help with that request.")])


def build_agent(
    settings: Settings | None = None,
    *,
    model: Any,
    tools: Sequence[Any],
    fallback_models: Sequence[Any] = (),
    summary_model: Any = None,
    checkpointer: Any = None,
    timeline: Timeline | None = None,
    profile: Profile = "full",
    reply_on_all_refused: bool = False,
    on_error: Callable[[Exception, ToolCallRequest], str | None] = tool_error_message,
    model_limit_exit: Literal["end", "error"] = "end",
    tool_limit_exit: Literal["continue", "error", "end"] = "continue",
    enable_todo: bool = False,
    redact_email: bool = False,
    interrupt_on: dict[str, Any] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Build the agent. Models are injected (real Azure models, or fakes in tests).

    profile:
      minimal   - ToolError only (shows what an unprotected agent does)
      resilient - retries, fallback, refusal, tool error/retry
      full      - resilient + call budgets + summarization (+ optional todo / PII / HITL)
    """
    s = settings or Settings()
    middleware: list[Any] = []

    if profile == "full":
        middleware += [
            ModelCallLimitMiddleware(
                thread_limit=s.model_thread_limit if checkpointer else None,
                run_limit=s.model_run_limit,
                exit_behavior=model_limit_exit,
            ),
            ToolCallLimitMiddleware(
                thread_limit=s.tool_thread_limit if checkpointer else None,
                run_limit=s.tool_run_limit,
                exit_behavior=tool_limit_exit,
            ),
            SummarizationMiddleware(
                model=summary_model or model,
                trigger=("tokens", s.summarize_trigger_tokens),
                keep=("messages", s.summarize_keep_messages),
            ),
        ]
        if enable_todo:
            middleware.append(TodoListMiddleware())
        if redact_email:
            middleware.append(PIIMiddleware("email", strategy="redact"))
        if interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    if profile in ("resilient", "full"):
        if reply_on_all_refused:
            middleware.append(reply_on_refusal)
        if fallback_models:
            middleware.append(ModelFallbackMiddleware(*fallback_models))
        middleware += [
            ModelRetryMiddleware(
                max_retries=s.model_retry_max,
                initial_delay=s.retry_initial_delay,
                backoff_factor=s.retry_backoff_factor,
                max_delay=s.retry_max_delay,
                jitter=s.retry_jitter,
                on_failure="error",  # default "continue" would hide the failure from Fallback
            ),
            ModelRefusalMiddleware(),
        ]

    middleware.append(ToolErrorMiddleware(on_error))  # outer
    if profile in ("resilient", "full"):
        middleware.append(
            ToolRetryMiddleware(
                max_retries=s.tool_retry_max,
                # The default retry_on also retries ValueError etc. Retry only transient errors;
                # input errors go to ToolError so the model can fix them.
                retry_on=TRANSIENT_TOOL_ERRORS,
                initial_delay=s.retry_initial_delay,
                backoff_factor=s.retry_backoff_factor,
                max_delay=s.retry_max_delay,
                jitter=s.retry_jitter,
                on_failure="error",  # re-raise so ToolError decides what the model sees
            )
        )
    if timeline is not None:
        middleware.append(TimelineMiddleware(timeline))

    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
    )
