"""
Hierarchical memory (Phase 4).

Two tiers, loosely modelled on human memory:

  ShortTermMemory - the recent conversation. Bounded: it keeps only the
                    last N turns. When it overflows, the oldest turns are
                    compacted into a short running summary.
  LongTermMemory  - durable facts saved to a JSON file on disk. These
                    survive even after the program exits.

The orchestrator prepends memory to a prompt so the agent has context.
That is what makes a follow-up question like "what about 2011?" work -
on its own it is meaningless, but with the conversation prepended the
agent can see what "2011" refers to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    """One line of the conversation - who said it, and what they said."""

    role: str    # "user" or "assistant"
    text: str


class ShortTermMemory:
    """The recent conversation, kept to the last `max_turns` turns.

    When more turns are added than `max_turns`, the oldest turn is removed
    from the live list and folded into a short text `summary` instead.
    This is the "hierarchical" part: fresh turns are kept verbatim, older
    ones are compacted.
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self.summary: str = ""

    def add(self, role: str, text: str) -> None:
        """Record one turn. Compact the oldest turn if we are over capacity."""
        self.turns.append(Turn(role=role, text=text))

        # If we are over capacity, fold the oldest turn into the summary
        while len(self.turns) > self.max_turns:
            oldest = self.turns.pop(0)
            snippet = oldest.text[:150]
            self.summary = self.summary + f"\n- {oldest.role}: {snippet}"

    def as_context(self) -> str:
        """Render the whole memory as text, ready to prepend to a prompt."""
        lines = []

        if self.summary != "":
            lines.append("Earlier in the conversation (summarized):")
            lines.append(self.summary.strip())
            lines.append("")

        for turn in self.turns:
            lines.append(f"{turn.role}: {turn.text}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.turns)


class LongTermMemory:
    """Durable facts saved to a JSON file - they survive across sessions.

    Where ShortTermMemory forgets old turns, LongTermMemory keeps facts
    forever (until you delete the file).
    """

    def __init__(self, path: str = "memory/long_term.json"):
        self.path = Path(path)
        self.facts: list[str] = []

        # Load any facts saved on a previous run
        if self.path.exists():
            self.facts = json.loads(self.path.read_text())

    def remember(self, fact: str) -> None:
        """Save one durable fact to disk."""
        self.facts.append(fact)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.facts, indent=2))

    def recall(self, keyword: str = "") -> list[str]:
        """Return saved facts. With a keyword, return only matching facts."""
        if keyword == "":
            return list(self.facts)

        matches = []
        for fact in self.facts:
            if keyword.lower() in fact.lower():
                matches.append(fact)
        return matches
