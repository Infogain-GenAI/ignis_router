"""Latency collector — wraps routing calls to measure and store timing."""

from __future__ import annotations

import time
from typing import Any


class LatencyCollector:
    """
    Context manager and utility to measure routing latency.

    Usage:
        from ignis_router.evaluation import LatencyCollector

        with LatencyCollector() as lc:
            result = router.chat(query)
        elapsed = lc.elapsed  # seconds
    """

    def __init__(self):
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "LatencyCollector":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000
