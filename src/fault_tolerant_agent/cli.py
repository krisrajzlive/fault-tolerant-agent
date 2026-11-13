"""CLI: replay failure scenarios offline (no API calls) or run one capped live request.

    ft-agent scenarios [name|all]     offline, scripted fake models, prints a timeline
    ft-agent live [--prompt TEXT]     real Azure deployments, hard-capped (see Settings)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from langchain_core.exceptions import ModelAPIError, ModelRateLimitError, ModelRefusalError
from langchain_core.messages import AIMessage

from .config import Settings
from .factory import build_agent
from .timeline import Timeline
from .tools import FlakyUpstream, build_tools


def _scripted():
    """Import the fake model from the test helpers (kept out of the installed package)."""
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
    import fakes  # type: ignore

    return fakes


def _scenarios(f):
    M, call, refusal = f.ScriptedChatModel, f.tool_call, f.refusal
    return {
        "rate-limit": dict(
            desc="429 twice, then success -> ModelRetry",
            primary=lambda: M(tag="primary", script=[ModelRateLimitError("429"), ModelRateLimitError("429"), AIMessage("done")]),
        ),
        "outage": dict(
            desc="primary down -> retries exhausted -> ModelFallback",
            primary=lambda: M(tag="primary", script=[ModelAPIError("503")] * 3),
            fallback=lambda: M(tag="fallback", script=[AIMessage("answered by fallback")]),
        ),
        "refusal": dict(
            desc="primary refuses -> fallback answers",
            primary=lambda: M(tag="primary", script=[refusal()]),
            fallback=lambda: M(tag="fallback", script=[AIMessage("answered by fallback")]),
        ),
        "all-refuse": dict(
            desc="every model refuses -> ModelRefusalError reaches the caller",
            primary=lambda: M(tag="primary", script=[refusal()]),
            fallback=lambda: M(tag="fallback", script=[refusal()]),
        ),
        "bad-input": dict(
            desc="tool ValueError -> error ToolMessage -> model corrects itself",
            primary=lambda: M(tag="primary", script=[
                call("apply_discount", price=100, percent=150),
                call("apply_discount", price=100, percent=15),
                AIMessage("85.00 after the discount"),
            ]),
        ),
        "flaky-tool": dict(
            desc="tool ConnectionError twice -> ToolRetry -> success",
            primary=lambda: M(tag="primary", script=[call("get_exchange_rate", base="USD", quote="EUR"), AIMessage("1 USD = 0.92 EUR")]),
            upstream_fails=2,
        ),
        "tool-outage": dict(
            desc="tool keeps failing -> retried, then surfaced",
            primary=lambda: M(tag="primary", script=[call("get_exchange_rate", base="USD", quote="EUR")]),
            upstream_fails=99,
        ),
        "runaway": dict(
            desc="agent loops on a tool -> budget stops it",
            primary=lambda: M(tag="loop", script=[call("generate_report", topic="x")], loop=True),
            run_limit=4,
        ),
    }


def run_scenario(name: str, spec: dict) -> None:
    s = replace(Settings(), retry_initial_delay=0.2, retry_backoff_factor=2.0, retry_jitter=False,
                model_run_limit=spec.get("run_limit", 10))
    timeline, upstream = Timeline(), FlakyUpstream(fail_times=spec.get("upstream_fails", 0))
    fallback = spec.get("fallback")
    agent = build_agent(
        s, model=spec["primary"](), fallback_models=[fallback()] if fallback else [],
        tools=build_tools(upstream, slow_seconds=0.05), timeline=timeline, profile="full",
    )
    print(f"\n=== {name}: {spec['desc']}")
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        print(f"result: {result['messages'][-1].content!r}")
    except (ModelRefusalError, ConnectionError) as exc:
        print(f"raised: {type(exc).__name__}: {exc}")
    print(timeline.render())
    print("summary:", {k: v for k, v in timeline.summary().items() if v})


def run_live(prompt: str) -> None:
    from .models import make_azure_model

    s = replace(Settings.from_env(), model_run_limit=6, tool_run_limit=4)
    print(f"LIVE: primary={s.primary_model} fallback={s.fallback_model} max_tokens={s.max_tokens} "
          f"model_run_limit={s.model_run_limit} (hard caps; budget-sensitive)")
    timeline, upstream = Timeline(), FlakyUpstream(fail_times=1)  # first rate lookup fails once -> ToolRetry
    agent = build_agent(
        s,
        model=make_azure_model(s.primary_model, s),
        fallback_models=[make_azure_model(s.fallback_model, s)],
        summary_model=make_azure_model(s.summary_model, s),
        tools=build_tools(upstream, slow_seconds=0.2),
        timeline=timeline,
        profile="full",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    print("\nanswer:", result["messages"][-1].content)
    print(timeline.render())
    print("summary:", {k: v for k, v in timeline.summary().items() if v})


def main() -> None:
    p = argparse.ArgumentParser(prog="ft-agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scenarios", help="offline failure scenarios")
    sc.add_argument("name", nargs="?", default="all")
    lv = sub.add_parser("live", help="one capped request against Azure Foundry")
    lv.add_argument("--prompt", default="What is 1 USD in EUR? Then apply a 15% discount to 250.")
    args = p.parse_args()

    if args.cmd == "live":
        run_live(args.prompt)
        return
    scenarios = _scenarios(_scripted())
    names = list(scenarios) if args.name == "all" else [args.name]
    for n in names:
        if n not in scenarios:
            sys.exit(f"unknown scenario {n!r}; choose from {', '.join(scenarios)}")
        run_scenario(n, scenarios[n])


if __name__ == "__main__":
    main()
