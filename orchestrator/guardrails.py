"""
Guardrails (Phase 8) - checks an agent's answer BEFORE it reaches the user.

Two layers, catching two different kinds of mistake:

  Deterministic guardrails - plain Python rules. Fast, free, exact. They
    catch STRUCTURAL errors: an empty answer, a made-up customer segment,
    an absurdly large number.

  LLM-judge guardrail - a second LLM scores the answer against a rubric.
    It catches SEMANTIC errors that rigid rules cannot express, e.g.
    "this doesn't actually answer the question that was asked".

The deterministic checks live in this module. The LLM judge is wired up
in the Phase 8 notebook (it is just a run_agent call with a judging prompt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# The ONLY valid customer segments in this project's data.
VALID_SEGMENTS = {"VIP", "Regular", "At-Risk", "Churned"}

# Segment names that sound plausible but DO NOT exist in this data.
# If an agent uses one of these, it hallucinated a label.
HALLUCINATED_SEGMENTS = {
    "Premium", "Gold", "Silver", "Bronze", "Platinum",
    "Loyal", "Inactive", "High-Value", "Low-Value",
}


@dataclass
class GuardrailResult:
    """The outcome of running guardrails on one answer."""

    passed: bool = True
    violations: list[str] = field(default_factory=list)


def check_not_empty(answer: str) -> list[str]:
    """The answer must not be empty."""
    if answer.strip() == "":
        return ["answer is empty"]
    return []


def check_no_hallucinated_segments(answer: str) -> list[str]:
    """The answer must not mention customer segments that don't exist."""
    violations = []
    for bad_label in HALLUCINATED_SEGMENTS:
        if bad_label in answer:
            violations.append(
                f"mentions '{bad_label}', which is not a real customer segment"
            )
    return violations


def check_numbers_plausible(answer: str, max_value: float = 100_000_000) -> list[str]:
    """Flag any number larger than max_value - likely a hallucination.

    Retail revenue figures in this dataset never reach 100 million, so a
    larger number is almost certainly invented.
    """
    violations = []
    for token in re.findall(r"[\d,]+\.?\d*", answer):
        digits = token.replace(",", "")

        # Skip tokens that aren't a clean number
        if digits == "" or not digits.replace(".", "", 1).isdigit():
            continue

        if float(digits) > max_value:
            violations.append(f"implausibly large number: {token}")
    return violations


# Every deterministic check, gathered so they can be run together.
DETERMINISTIC_CHECKS = [
    check_not_empty,
    check_no_hallucinated_segments,
    check_numbers_plausible,
]


def run_deterministic_guardrails(answer: str) -> GuardrailResult:
    """Run every deterministic check and collect all violations."""
    all_violations = []
    for check in DETERMINISTIC_CHECKS:
        all_violations.extend(check(answer))

    return GuardrailResult(
        passed=(len(all_violations) == 0),
        violations=all_violations,
    )
