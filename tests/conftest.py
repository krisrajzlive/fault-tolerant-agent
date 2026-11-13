import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fault_tolerant_agent import FlakyUpstream, Settings, Timeline, build_agent, build_tools  # noqa: E402

HUMAN = {"messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def settings() -> Settings:
    return Settings().fast()


@pytest.fixture
def make(settings):
    """Factory returning (agent, timeline, upstream)."""

    def _make(model, *, fallbacks=(), upstream=None, profile="resilient", s=None, **kwargs):
        upstream = upstream or FlakyUpstream()
        timeline = Timeline()
        agent = build_agent(
            s or settings,
            model=model,
            fallback_models=fallbacks,
            tools=build_tools(upstream, slow_seconds=0),
            timeline=timeline,
            profile=profile,
            **kwargs,
        )
        return agent, timeline, upstream

    return _make
