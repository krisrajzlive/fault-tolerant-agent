import pytest
from fakes import ScriptedChatModel
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError, ModelRateLimitError
from langchain_core.messages import AIMessage

from conftest import HUMAN


def final_text(result) -> str:
    return result["messages"][-1].content


def test_rate_limit_twice_then_success(make):
    model = ScriptedChatModel(tag="primary", script=[ModelRateLimitError("429"), ModelRateLimitError("429"), AIMessage("done")])
    agent, timeline, _ = make(model)
    assert final_text(agent.invoke(HUMAN)) == "done"
    assert model.calls == 3
    s = timeline.summary()
    assert s["model_retries"] == 2 and s["fallbacks"] == 0


def test_retries_exhausted_raises_when_no_fallback(make):
    model = ScriptedChatModel(tag="primary", script=[ModelRateLimitError("429")] * 3)
    agent, _, _ = make(model)
    with pytest.raises(ModelRateLimitError):
        agent.invoke(HUMAN)
    assert model.calls == 3  # 1 initial + model_retry_max (2)


def test_provider_outage_uses_fallback(make):
    primary = ScriptedChatModel(tag="primary", script=[ModelAPIError("503")] * 3)
    fallback = ScriptedChatModel(tag="fallback", script=[AIMessage("from fallback")])
    agent, timeline, _ = make(primary, fallbacks=[fallback])
    assert final_text(agent.invoke(HUMAN)) == "from fallback"
    assert (primary.calls, fallback.calls) == (3, 1)  # retried on the primary first, then fell back
    assert timeline.summary()["fallbacks"] == 1


def test_non_retryable_error_is_not_retried(make):
    model = ScriptedChatModel(tag="primary", script=[ModelAuthenticationError("401")] * 3)
    agent, _, _ = make(model)
    with pytest.raises(ModelAuthenticationError):
        agent.invoke(HUMAN)
    assert model.calls == 1  # is_retryable=False is respected (langchain>=1.3.16)


def test_auth_error_is_not_retried_but_still_triggers_fallback(make):
    primary = ScriptedChatModel(tag="primary", script=[ModelAuthenticationError("401")] * 3)
    fallback = ScriptedChatModel(tag="fallback", script=[AIMessage("rescued")])
    agent, _, _ = make(primary, fallbacks=[fallback])
    assert final_text(agent.invoke(HUMAN)) == "rescued"
    assert primary.calls == 1  # fallback fires on ANY exception: a pitfall, a bad key silently shifts traffic


def test_pitfall_retry_on_failure_continue_hides_outage_from_fallback():
    """ModelRetryMiddleware defaults to on_failure='continue': it swallows the final error into an
    AIMessage, so the fallback never sees an exception. The factory sets on_failure='error'."""
    primary = ScriptedChatModel(tag="primary", script=[ModelAPIError("503")] * 3)
    fallback = ScriptedChatModel(tag="fallback", script=[AIMessage("from fallback")])
    agent = create_agent(
        model=primary,
        tools=[],
        middleware=[
            ModelFallbackMiddleware(fallback),
            ModelRetryMiddleware(max_retries=2, initial_delay=0, backoff_factor=0, jitter=False),  # default continue
        ],
    )
    result = agent.invoke(HUMAN)
    assert fallback.calls == 0
    assert "503" in result["messages"][-1].content or "ModelAPIError" in result["messages"][-1].content
