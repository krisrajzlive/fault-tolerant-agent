# fault-tolerant-agent

[![tests](https://github.com/krisrajzlive/fault-tolerant-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/krisrajzlive/fault-tolerant-agent/actions/workflows/ci.yml)

A LangChain agent hardened against the failures production agents actually hit: rate limits, provider outages,
model refusals, flaky and badly-called tools, runaway loops and growing context. It composes LangChain's resilience,
budget and context-management middleware in a deliberate order, and every behaviour is covered by failure-injection
tests that use a scripted fake model, so the suite needs **no network, no keys and no spend**.

Verified against `langchain 1.5.0`, `langgraph 1.2.20`, `langchain-core 1.8.0` (Python 3.11+). Models run on Azure AI
Foundry with Entra ID auth, so there are no API keys to manage.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv/Scripts/python.exe -m pytest                              # 28 tests, offline
.venv/Scripts/python.exe -m fault_tolerant_agent.cli scenarios  # timelines of every failure mode, offline
```

Sample timeline (`scenarios outage`): the primary is retried with growing backoff, then the fallback answers.

```
t+  0.01s  MODEL primary     error:ModelAPIError
t+  0.22s  MODEL primary     error:ModelAPIError
t+  0.62s  MODEL primary     error:ModelAPIError
t+  0.62s  MODEL fallback    ok
summary: {'model_attempts': 4, 'model_retries': 2, 'fallbacks': 1, 'model_errors': 3}
```

Live (real Azure AI Foundry deployments, hard-capped to a handful of tiny calls). Needs `az login` and an existing
resource group you can write to:

```bash
powershell ./infra/provision.ps1 -ResourceGroup <your-rg> -WhatIf   # prints the plan; drop -WhatIf to create it
.venv/Scripts/python.exe -m fault_tolerant_agent.cli live
powershell ./infra/teardown.ps1                                      # deletes the Foundry account it created
```

A live run against two real deployments answered a two-tool question correctly, with the injected `ConnectionError` on
the first rate lookup retried by `ToolRetryMiddleware` and no change to the final answer.

## The stack (`factory.py`)

Wrap-style hooks nest: the **first listed is outermost**. `before_*` run first to last, `after_*` last to first
(pinned by `tests/test_middleware_order.py`).

| # | Middleware | Job |
|---|------------|-----|
| 1 | `ModelCallLimitMiddleware` | Budget: caps agent model steps per run / per thread (thread limit needs a checkpointer) |
| 2 | `ToolCallLimitMiddleware` | Budget: caps tool executions |
| 3 | `SummarizationMiddleware` | Compacts history past a token threshold, using a cheap model |
| 4 | `reply_on_refusal` (optional) | Turns "every model refused" into a normal reply; must sit before Fallback |
| 5 | `ModelFallbackMiddleware` | Moves to the next model when the inner stack raises |
| 6 | `ModelRetryMiddleware` | Retries the same model with exponential backoff and jitter, then raises |
| 7 | `ModelRefusalMiddleware` | Raises `ModelRefusalError` for refusals; listed after Fallback so Fallback catches it |
| 8 | `ToolErrorMiddleware` (outer) | `ValueError` becomes an error `ToolMessage` the model can act on; everything else propagates |
| 9 | `ToolRetryMiddleware` (inner) | Retries transient tool failures; `on_failure="error"` so #8 sees the final error |
| 10 | `TimelineMiddleware` | Innermost: records every real attempt (retries, fallbacks, refusals) |

Plus: a `rate_limiter` (`InMemoryRateLimiter`) and `timeout` on each model, a checkpointer, and optional
`TodoListMiddleware`, `PIIMiddleware("email", "redact")` and `HumanInTheLoopMiddleware`.

Profiles: `minimal` (ToolError only, shows the unprotected baseline), `resilient`, `full`.

## Error type -> strategy

| Error | Who fixes it | Strategy | Where |
|-------|--------------|----------|-------|
| Transient (429, 5xx, network) | System | Retry with backoff | `ModelRetry`, `ToolRetry` |
| LLM-recoverable (bad tool input) | The model | Error `ToolMessage`, model corrects | `ToolError` |
| Provider outage | System | Fall back to another model | `ModelFallback` |
| Refusal | System / developer | Fall back, then catch `ModelRefusalError` or reply | `ModelRefusal` + `ModelFallback` |
| Needs a human | Human | `interrupt()` | `HumanInTheLoopMiddleware` |
| Runaway loop | System | Cap calls | `ModelCallLimit`, `ToolCallLimit` |
| Unexpected | Developer | Let it propagate | no middleware |

## Pitfalls (each one is pinned by a test)

- **Retry layers multiply.** The SDK retries 6 times by default, `ModelRetryMiddleware` adds more. The factory sets the
  SDK's `max_retries=0` so there is one visible retry layer. Default stacking would mean roughly 18 calls before a fallback.
- **`ModelRetryMiddleware` defaults to `on_failure="continue"`.** It swallows the final error into an `AIMessage`, so the
  fallback never fires on an outage. Use `"error"` (`test_pitfall_retry_on_failure_continue_hides_outage_from_fallback`).
- **`ToolRetryMiddleware`'s default `retry_on` retries `ValueError` too.** Bad input is retried pointlessly before the model
  can fix it. The factory restricts retries to `ConnectionError` and `TimeoutError`.
- **Fallback fires on any exception,** including a 401. A bad key silently shifts traffic to the fallback model. Retry,
  however, respects `is_retryable=False`, so the primary is called once (`langchain>=1.3.16`).
- **Refusals are successful responses.** Without `ModelRefusalMiddleware` the run looks successful. Only `refusal` content
  blocks are detected, not refusals written as plain text, and streaming can leak partial text first.
- **Non-idempotent tools under retry** run their side effect again. Retried tools should be idempotent or take an
  idempotency key. LangGraph node retries re-run the whole node, so the same applies there.
- **`on_failure="continue"` hides errors.** Use it only where a degraded answer is acceptable.
- **Fallback quality and cost.** A fallback model can answer worse or cost more. `InMemoryRateLimiter` limits requests only,
  not tokens, and lives in one process; use a gateway or queue for a fleet.
- **Limits count agent steps, not attempts:** retries and fallbacks inside one step do not use the run budget.

## Layout

```
src/fault_tolerant_agent/  config.py  models.py  tools.py  timeline.py  factory.py  cli.py
tests/                     fakes.py (ScriptedChatModel) + one file per concern
infra/                     provision.ps1  teardown.ps1
```

## Cost safety (live runs)

Pay-per-token deployments at 1K TPM, `max_tokens=300`, `model_run_limit=6`, `tool_run_limit=4`. The test suite makes no
API calls. `teardown.ps1` removes the account this project created (a role without purge rights leaves an unbilled
soft-deleted record).

## Roadmap

- LangGraph variant: node `retry_policy`, `TimeoutPolicy` and `error_handler` (timeouts require async nodes)
- LangSmith tracing with per-run counters for retries, fallbacks and refusals

## License

MIT
