import pytest
from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage, ToolMessage

from conftest import HUMAN
from fault_tolerant_agent import FlakyUpstream


def tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_value_error_becomes_error_toolmessage_and_model_corrects(make):
    model = ScriptedChatModel(
        tag="primary",
        script=[
            tool_call("apply_discount", price=100, percent=150),  # bad input
            tool_call("apply_discount", price=100, percent=15),  # corrected
            AIMessage("The discounted price is 85.00"),
        ],
    )
    agent, timeline, _ = make(model)
    result = agent.invoke(HUMAN)
    first, second = tool_messages(result)
    assert first.status == "error"
    assert "ValueError" in first.content
    assert "between 0 and 100" not in first.content  # type only, no raw exception text
    assert second.content == "85.00"
    assert "ValueError" in model.seen[1][-1].content  # the model saw the error and could correct itself
    assert result["messages"][-1].content.endswith("85.00")
    assert timeline.summary()["tool_retries"] == 0  # a ValueError is not retried: it needs a different input


def test_connection_error_retried_then_surfaced(make):
    upstream = FlakyUpstream(fail_times=99)
    model = ScriptedChatModel(tag="primary", script=[tool_call("get_exchange_rate", base="USD", quote="EUR")])
    agent, timeline, _ = make(model, upstream=upstream)
    with pytest.raises(ConnectionError):  # ToolError returns None for it, so it propagates
        agent.invoke(HUMAN)
    assert upstream.calls == 3  # 1 initial + tool_retry_max (2)
    assert timeline.summary()["tool_retries"] == 2


def test_transient_tool_failure_recovers(make):
    upstream = FlakyUpstream(fail_times=2)
    model = ScriptedChatModel(
        tag="primary",
        script=[tool_call("get_exchange_rate", base="USD", quote="EUR"), AIMessage("1 USD = 0.92 EUR")],
    )
    agent, _, _ = make(model, upstream=upstream)
    result = agent.invoke(HUMAN)
    assert tool_messages(result)[0].content == "1 USD = 0.92 EUR"
    assert upstream.calls == 3


def test_minimal_profile_has_no_tool_retry(make):
    upstream = FlakyUpstream(fail_times=1)
    model = ScriptedChatModel(tag="primary", script=[tool_call("get_exchange_rate", base="USD", quote="EUR")])
    agent, _, _ = make(model, upstream=upstream, profile="minimal")
    with pytest.raises(ConnectionError):
        agent.invoke(HUMAN)
    assert upstream.calls == 1


def test_unknown_pair_value_error_from_upstream_is_llm_recoverable(make):
    model = ScriptedChatModel(
        tag="primary",
        script=[tool_call("get_exchange_rate", base="USD", quote="XXX"), AIMessage("That pair is not supported.")],
    )
    agent, _, _ = make(model)
    result = agent.invoke(HUMAN)
    assert tool_messages(result)[0].status == "error"
