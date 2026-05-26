"""
Eval-driven development (Phase 9).

An "eval" runs the agent against a GOLDEN SET — a fixed list of questions
with known-correct answers — scores every answer, and reports aggregate
metrics (pass rate, latency, cost). This is how you MEASURE an agent
system instead of eyeballing one answer at a time.

This module holds the eval data types and scoring helpers. The eval loop
itself lives in the Phase 9 notebook so you can see and edit it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenCase:
    """One test case: a question plus the facts its answer must contain."""

    case_id: str
    question: str
    expected_substrings: list[str]


@dataclass
class EvalRow:
    """The result of running one golden case through the agent."""

    case_id: str
    passed: bool
    missing: list[str]
    latency_ms: int
    cost_usd: float


def load_golden_set(path: str = "tests/golden.json") -> list[GoldenCase]:
    """Load the golden set from a JSON file into GoldenCase objects."""
    raw = json.loads(Path(path).read_text())

    cases = []
    for item in raw:
        cases.append(
            GoldenCase(
                case_id=item["id"],
                question=item["question"],
                expected_substrings=item["expected_substrings"],
            )
        )
    return cases


def score_answer(answer: str, expected_substrings: list[str]) -> list[str]:
    """Score one answer: return the expected substrings NOT found in it.

    An empty list means every expected fact is present -> the case passed.
    """
    missing = []
    for substring in expected_substrings:
        if substring not in answer:
            missing.append(substring)
    return missing


def summarize(rows: list[EvalRow]) -> dict:
    """Aggregate a list of EvalRow results into headline metrics."""
    n_cases = len(rows)

    n_passed = 0
    total_latency = 0
    total_cost = 0.0
    for row in rows:
        if row.passed:
            n_passed += 1
        total_latency += row.latency_ms
        total_cost += row.cost_usd

    # Guard against an empty eval set (avoid dividing by zero)
    if n_cases > 0:
        pass_rate = n_passed / n_cases
        avg_latency_ms = total_latency / n_cases
    else:
        pass_rate = 0.0
        avg_latency_ms = 0.0

    return {
        "n_cases": n_cases,
        "n_passed": n_passed,
        "pass_rate": pass_rate,
        "avg_latency_ms": avg_latency_ms,
        "total_cost_usd": round(total_cost, 6),
    }
