from fakes import ScriptedChatModel, tool_call
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver


def test_email_is_redacted_before_reaching_the_model(make, settings):
    model = ScriptedChatModel(tag="p", script=[AIMessage("ok")])
    agent, _, _ = make(model, profile="full", redact_email=True)
    agent.invoke({"messages": [{"role": "user", "content": "contact me at jane.doe@example.com please"}]})
    sent = " ".join(str(m.content) for m in model.seen[0])
    assert "jane.doe@example.com" not in sent
    assert "REDACTED" in sent.upper()


def test_human_in_the_loop_pauses_before_sensitive_tool(make):
    model = ScriptedChatModel(tag="p", script=[tool_call("apply_discount", price=10, percent=5)])
    agent, _, _ = make(model, profile="full", checkpointer=InMemorySaver(), interrupt_on={"apply_discount": True})
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "discount it"}]},
        {"configurable": {"thread_id": "hitl"}},
    )
    assert result.get("__interrupt__")  # paused: the tool has not run
