"""Documents the execution order the factory relies on, so a reordering cannot slip in silently."""

from fakes import ScriptedChatModel
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from conftest import HUMAN


class Recorder(AgentMiddleware):
    def __init__(self, name, log):
        super().__init__()
        self._name, self._log = name, log

    @property
    def name(self) -> str:  # middleware names must be unique within one agent
        return f"Recorder{self._name}"

    def before_model(self, state, runtime):
        self._log.append(f"before:{self._name}")

    def after_model(self, state, runtime):
        self._log.append(f"after:{self._name}")

    def wrap_model_call(self, request, handler):
        self._log.append(f"wrap-in:{self._name}")
        out = handler(request)
        self._log.append(f"wrap-out:{self._name}")
        return out


def test_before_first_to_last_wrap_nested_after_reversed():
    log: list[str] = []
    agent = create_agent(
        model=ScriptedChatModel(tag="m", script=[AIMessage("hi")]),
        tools=[],
        middleware=[Recorder("A", log), Recorder("B", log)],
    )
    agent.invoke(HUMAN)
    assert log == [
        "before:A", "before:B",  # first -> last
        "wrap-in:A", "wrap-in:B", "wrap-out:B", "wrap-out:A",  # first listed is outermost
        "after:B", "after:A",  # last -> first
    ]
