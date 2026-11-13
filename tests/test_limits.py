from dataclasses import replace

import pytest
from fakes import ScriptedChatModel, tool_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from conftest import HUMAN


def looping_model():
    return ScriptedChatModel(tag="loop", script=[tool_call("generate_report", topic="x")], loop=True)


def test_run_limit_stops_looping_agent_gracefully(make, settings):
    model = looping_model()
    agent, _, _ = make(model, profile="full", s=replace(settings, model_run_limit=3))
    result = agent.invoke(HUMAN)
    assert model.calls == 3  # the 4th model call was never made
    assert "limit" in result["messages"][-1].content.lower()


def test_run_limit_error_mode_raises(make, settings):
    model = looping_model()
    agent, _, _ = make(model, profile="full", s=replace(settings, model_run_limit=2), model_limit_exit="error")
    with pytest.raises(ModelCallLimitExceededError):
        agent.invoke(HUMAN)
    assert model.calls == 2


def test_tool_call_limit_error_mode_raises(make, settings):
    model = looping_model()
    s = replace(settings, model_run_limit=20, tool_run_limit=2)
    agent, _, _ = make(model, profile="full", s=s, tool_limit_exit="error")
    with pytest.raises(ToolCallLimitExceededError):
        agent.invoke(HUMAN)


def test_tool_call_limit_continue_mode_blocks_extra_calls(make, settings):
    model = looping_model()
    s = replace(settings, model_run_limit=6, tool_run_limit=2)
    agent, timeline, _ = make(model, profile="full", s=s)  # default exit_behavior="continue"
    agent.invoke(HUMAN)
    assert timeline.summary()["tool_attempts"] == 2  # later calls are blocked with an error message


def test_thread_limit_spans_runs_and_needs_a_checkpointer(make, settings):
    model = ScriptedChatModel(tag="p", script=[AIMessage("a"), AIMessage("b"), AIMessage("c")])
    s = replace(settings, model_thread_limit=2, model_run_limit=10)
    agent, _, _ = make(model, profile="full", s=s, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    for _ in range(3):
        agent.invoke(HUMAN, cfg)
    assert model.calls == 2  # the third run was stopped by the thread-wide budget


def test_limits_count_agent_steps_not_retries(make, settings):
    from langchain_core.exceptions import ModelRateLimitError

    model = ScriptedChatModel(tag="p", script=[ModelRateLimitError("429"), ModelRateLimitError("429"), AIMessage("fine")])
    agent, _, _ = make(model, profile="full", s=replace(settings, model_run_limit=1))
    assert agent.invoke(HUMAN)["messages"][-1].content == "fine"  # 3 attempts, still within run_limit=1
