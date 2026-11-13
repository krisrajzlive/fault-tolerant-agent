import pytest
from fakes import ScriptedChatModel, refusal
from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from conftest import HUMAN


def test_refusal_falls_back_to_next_model(make):
    primary = ScriptedChatModel(tag="primary", script=[refusal()])
    fallback = ScriptedChatModel(tag="fallback", script=[AIMessage("fallback answer")])
    agent, timeline, _ = make(primary, fallbacks=[fallback])
    assert agent.invoke(HUMAN)["messages"][-1].content == "fallback answer"
    assert primary.calls == 1  # retry middleware does not retry refusals
    s = timeline.summary()
    assert s["refusals"] == 1 and s["fallbacks"] == 1


def test_every_model_refuses_raises(make):
    primary = ScriptedChatModel(tag="primary", script=[refusal()])
    fallback = ScriptedChatModel(tag="fallback", script=[refusal()])
    agent, _, _ = make(primary, fallbacks=[fallback])
    with pytest.raises(ModelRefusalError):
        agent.invoke(HUMAN)
    assert (primary.calls, fallback.calls) == (1, 1)


def test_refused_reply_is_not_saved_and_thread_stays_usable(make):
    primary = ScriptedChatModel(tag="primary", script=[refusal(), AIMessage("second try ok")])
    agent, _, _ = make(primary, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    with pytest.raises(ModelRefusalError):
        agent.invoke(HUMAN, cfg)
    result = agent.invoke({"messages": [{"role": "user", "content": "try something else"}]}, cfg)
    assert result["messages"][-1].content == "second try ok"
    assert not any(isinstance(m.content, list) for m in result["messages"])  # refused reply never stored


def test_reply_on_all_refused_returns_normal_message(make):
    primary = ScriptedChatModel(tag="primary", script=[refusal()])
    fallback = ScriptedChatModel(tag="fallback", script=[refusal()])
    agent, _, _ = make(primary, fallbacks=[fallback], reply_on_all_refused=True)
    assert "can't help" in agent.invoke(HUMAN)["messages"][-1].content
