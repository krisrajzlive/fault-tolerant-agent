import time
from dataclasses import replace

from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from conftest import HUMAN


def history(n: int, size: int = 120) -> dict:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i} " + "x" * size})
    msgs.append({"role": "user", "content": "latest question"})
    return {"messages": msgs}


def test_summarization_triggers_at_token_threshold(make, settings):
    main = ScriptedChatModel(tag="main", script=[AIMessage("answer")])
    summarizer = ScriptedChatModel(tag="summarizer", script=[AIMessage("SUMMARY OF EARLIER CHAT")])
    s = replace(settings, summarize_trigger_tokens=300, summarize_keep_messages=4)
    agent, _, _ = make(main, profile="full", s=s, summary_model=summarizer)
    agent.invoke(history(30))
    assert summarizer.calls == 1
    seen = main.seen[0]
    assert len(seen) < 20  # 31 messages were compacted
    assert any("SUMMARY OF EARLIER CHAT" in str(m.content) for m in seen)
    assert "latest question" in str(seen[-1].content)  # recent messages are kept


def test_summarization_not_triggered_below_threshold(make, settings):
    main = ScriptedChatModel(tag="main", script=[AIMessage("answer")])
    summarizer = ScriptedChatModel(tag="summarizer", script=[AIMessage("unused")])
    agent, _, _ = make(main, profile="full", s=settings, summary_model=summarizer)  # trigger = 4000 tokens
    agent.invoke(history(4, size=20))
    assert summarizer.calls == 0


def test_timeline_counts_retries_per_call_when_parallel_tool_calls_interleave():
    from fault_tolerant_agent import Timeline

    t = Timeline()
    t.record("tool", "get_exchange_rate", "error:ConnectionError", 1, key="a")
    t.record("tool", "apply_discount", "ok", 1, key="b")  # parallel call finishes in between
    t.record("tool", "get_exchange_rate", "ok", 1, key="a")  # retry of call "a"
    s = t.summary()
    assert s["tool_retries"] == 1 and s["tool_errors"] == 1


def test_rate_limiter_throttles_requests_proactively():
    limiter = InMemoryRateLimiter(requests_per_second=10, check_every_n_seconds=0.01, max_bucket_size=1)
    model = ScriptedChatModel(tag="p", script=[AIMessage("ok")], loop=True, rate_limiter=limiter)
    t0 = time.monotonic()
    for _ in range(4):
        model.invoke("hi")
    assert time.monotonic() - t0 >= 0.25  # ~10 req/s means at least ~0.3s for 4 requests; never faster
