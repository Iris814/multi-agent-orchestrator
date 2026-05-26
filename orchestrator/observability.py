"""
Observability primitives. Used from Phase 1 onward so every agent run produces
a structured trace (latency, tokens, cost) — the rigor that distinguishes a
toy demo from an engineered system.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Per-model pricing (USD per 1M tokens). Cached input is ~10% of input.
# Keep this table updated as models change.
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cached_input": 0.10},
    "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-opus-4-7":            {"input": 15.00, "output": 75.00, "cached_input": 1.50},
}


@dataclass
class Trace:
    """One structured record per agent run. Append to JSONL for analytics."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: str = ""
    question: str = ""
    answer: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    n_tool_calls: int = 0
    n_delegations: int = 0
    cost_usd: float = 0.0
    passed: bool | None = None
    error: str | None = None

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cached_input_tokens
        return self.cached_input_tokens / total if total else 0.0

    def compute_cost(self) -> float:
        """Fill in cost_usd from the token counts using MODEL_PRICING."""
        pricing = MODEL_PRICING.get(self.model)
        if not pricing:
            return 0.0
        cost = (
            self.input_tokens        / 1_000_000 * pricing["input"]
            + self.output_tokens     / 1_000_000 * pricing["output"]
            + self.cached_input_tokens / 1_000_000 * pricing["cached_input"]
        )
        self.cost_usd = round(cost, 6)
        return self.cost_usd

    def to_dict(self) -> dict:
        return asdict(self)

    def append_jsonl(self, path: str | Path = "traces/traces.jsonl") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


class Timer:
    """`with Timer() as t: ...` then read t.elapsed_ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_ms = 0
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
