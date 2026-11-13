"""Three small tools, each chosen to trigger a different failure strategy.

* ``get_exchange_rate``  - flaky external call: raises ConnectionError (transient -> retry).
* ``apply_discount``     - rejects bad input with ValueError (LLM-recoverable -> ToolErrorMiddleware).
* ``generate_report``    - slow call (latency / timeout handling).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.tools import tool

_RATES = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09, ("USD", "JPY"): 150.0, ("JPY", "USD"): 0.0067}


@dataclass
class FlakyUpstream:
    """Fails the first ``fail_times`` calls, then succeeds. Counts every call."""

    fail_times: int = 0
    calls: int = 0
    log: list[str] = field(default_factory=list)

    def get_rate(self, base: str, quote: str) -> float:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("upstream rate service unavailable")
        try:
            return _RATES[(base.upper(), quote.upper())]
        except KeyError as exc:
            raise ValueError(f"unsupported pair {base}/{quote}") from exc


def build_tools(upstream: FlakyUpstream | None = None, slow_seconds: float = 1.0) -> list:
    upstream = upstream or FlakyUpstream()

    @tool
    def get_exchange_rate(base: str, quote: str) -> str:
        """Look up the current exchange rate for a currency pair, e.g. base='USD', quote='EUR'."""
        return f"1 {base.upper()} = {upstream.get_rate(base, quote)} {quote.upper()}"

    @tool
    def apply_discount(price: float, percent: float) -> str:
        """Apply a percentage discount (0-100) to a non-negative price and return the new price."""
        if price < 0:
            raise ValueError("price must be non-negative")
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        return f"{price * (1 - percent / 100):.2f}"

    @tool
    def generate_report(topic: str) -> str:
        """Generate a (slow) short report about a topic."""
        time.sleep(slow_seconds)
        return f"Report on {topic}: all systems nominal."

    return [get_exchange_rate, apply_discount, generate_report]
