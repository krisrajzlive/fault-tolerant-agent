"""A scripted chat model for failure injection. No network, no API calls."""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count(1)


def tool_call(name: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def refusal(text: str = "I can't help with that.") -> AIMessage:
    return AIMessage(content=[{"type": "refusal", "refusal": text}])


class ScriptedChatModel(BaseChatModel):
    """Plays back ``script``: an Exception is raised, a message is returned.

    ``loop=True`` repeats the last item forever (simulates a runaway agent).
    ``calls`` counts every invocation; ``seen`` keeps the message list of each call.
    """

    tag: str = "fake"
    script: list[Any] = Field(default_factory=list)
    loop: bool = False
    calls: int = 0
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # the agent binds tools; the script ignores them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        idx = self.calls
        self.calls += 1
        if not self.script:
            item: Any = AIMessage("ok")
        elif idx < len(self.script):
            item = self.script[idx]
        elif self.loop:
            item = self.script[-1]
            if isinstance(item, AIMessage) and item.tool_calls:  # fresh ids, or the agent treats them as done
                item = item.model_copy(
                    update={"tool_calls": [{**tc, "id": f"call_{next(_ids)}"} for tc in item.tool_calls]}
                )
        else:
            raise AssertionError(f"[{self.tag}] script exhausted after {idx} calls")
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])
