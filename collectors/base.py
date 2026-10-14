"""Shared state and helpers for LLM metric collectors.

Every LLM collector (llama, vLLM, LiteLLM, Strata, upstream) maintains
the same three pieces of per-target scrape state:

* **last-good snapshot** – the most recent successful scrape result plus
  a monotonic timestamp.  On a failed scrape the UI keeps showing this
  data flagged ``"stale"`` for up to *stale_max_age_s* seconds so a
  transient timeout doesn't flicker the card to "unreachable".
* **previous-scrape counters** – the prior successful scrape's cumulative
  counters (and timestamp), used to compute rates from per-scrape deltas.
* **per-request rate window** – a bounded deque of recent per-request
  rate samples, used to report a smoothed rate instead of a single-tick
  spike.

:class:`BaseLLMCollector` encapsulates this shape so each collector
subclass (or instance) only needs to supply the scrape logic.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

# Default window for per-request rate tracking.
LLM_REQ_RATE_WINDOW = 5


class BaseLLMCollector:
    """Encapsulates the duplicated stale-hold / rate / state_key shape
    shared by every LLM collector.

    Each instance owns:

    * ``_last_good`` – per-key last successful snapshot + timestamp.
    * ``_prev`` – per-key previous scrape values (for counter deltas).
    * ``_req_rates`` – per-key sliding window of per-request rates.

    The :meth:`stale_or_none` method implements the stale-hold: on a
    failed scrape the last good snapshot is returned with
    ``status: "stale"`` for up to ``stale_max_age_s`` seconds so a
    transient timeout doesn't flicker the UI to "unreachable".
    """

    def __init__(
        self,
        stale_max_age_s: float,
        rate_window: int = LLM_REQ_RATE_WINDOW,
    ) -> None:
        self.stale_max_age_s = stale_max_age_s
        self._rate_window = rate_window
        self._last_good: dict[str, tuple[float, dict[str, Any]]] = {}
        self._prev: dict[str, dict[str, Any]] = {}
        self._req_rates: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------
    # Stale-hold
    # ------------------------------------------------------------------

    def stale_or_none(self, key: str) -> dict[str, Any] | None:
        """Return the last good snapshot flagged ``stale`` if within the
        hold window; otherwise ``None``."""
        held_good = self._last_good.get(key)
        if held_good is not None:
            ts, held = held_good
            if time.monotonic() - ts <= self.stale_max_age_s:
                out = dict(held)
                out["status"] = "stale"
                return out
        return None

    def record_good(self, key: str, data: dict[str, Any]) -> None:
        """Record a successful scrape result (starts the stale-hold clock)."""
        self._last_good[key] = (time.monotonic(), data)

    # ------------------------------------------------------------------
    # Previous-scrape state (for counter deltas)
    # ------------------------------------------------------------------

    def get_prev(self, key: str) -> dict[str, Any] | None:
        """Return the previous scrape values for *key*, or ``None``."""
        return self._prev.get(key)

    def set_prev(self, key: str, prev: dict[str, Any]) -> None:
        """Store the current scrape values for *key* (for next delta)."""
        self._prev[key] = prev

    # ------------------------------------------------------------------
    # Per-request rate window
    # ------------------------------------------------------------------

    def get_rates(self, key: str) -> deque[float]:
        """Return the sliding rate window for *key* (created on first use)."""
        return self._req_rates.setdefault(
            key, deque(maxlen=self._rate_window)
        )
