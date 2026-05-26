"""
Orchestrator-worker multi-agent scaffolding (Phase 2+).

Vocabulary
----------
worker        : a specialized agent — its own system prompt, its own tools.
orchestrator  : coordinates workers; delegates a task, collects the result.
critic        : a worker whose job is to REVIEW another worker's output
                and return a verdict (reflection).

This module provides the *machinery* (run one agent, collect its result).
The orchestrator logic itself lives in the Phase 2 notebook so you can see
and edit it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


@dataclass
class AgentRun:
    """The result of running ONE worker agent once.

    Token counts are summed so the orchestrator can roll them into a Trace.
    """

    name: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    n_tool_calls: int = 0


async def _prompt_stream(text: str):
    """Wrap a plain string prompt as the one-message async stream that the
    SDK's streaming mode expects.

    Streaming mode is required once a `can_use_tool` callback is in play
    (Phase 7+). It also works fine without one - so run_agent always uses
    it, and the same helper covers every phase.
    """
    yield {
        "type": "user",
        "message": {"role": "user", "content": text},
        "parent_tool_use_id": None,
        "session_id": "default",
    }


async def run_agent(
    name: str,
    prompt: str,
    options,
    on_tool_use: Callable[[str, dict], None] | None = None,
) -> AgentRun:
    """Run a single worker agent turn.

    Sends `prompt` to an agent configured by `options`, drains the event
    stream, and returns an AgentRun with the final text + usage stats.

    If `on_tool_use` is given, it is called with (tool_name, tool_input)
    every time the agent invokes a tool — used by the UI to render live
    tool-call chips as the agent works.

    This is the Phase 1 agent loop, extracted into a reusable function —
    the orchestrator calls it once per delegation.
    """
    run = AgentRun(name=name)
    async for msg in query(prompt=_prompt_stream(prompt), options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    run.answer += block.text
                elif isinstance(block, ToolUseBlock):
                    run.n_tool_calls += 1
                    if on_tool_use is not None:
                        try:
                            on_tool_use(block.name, block.input or {})
                        except Exception:
                            pass
        elif isinstance(msg, ResultMessage):
            usage = msg.usage or {}
            run.input_tokens += usage.get("input_tokens", 0)
            run.output_tokens += usage.get("output_tokens", 0)
            run.cached_input_tokens += usage.get("cache_read_input_tokens", 0)
    return run


def parse_verdict(critic_answer: str) -> tuple[bool, str]:
    """Turn the critic's free text into a structured verdict.

    The critic is instructed to start its reply with APPROVE or REJECT.
    Returns (approved: bool, reason: str).
    """
    text = critic_answer.strip()

    # Grab the critic's first word (empty string if the reply was blank)
    words = text.split()
    if words:
        first_word = words[0].upper()
    else:
        first_word = ""

    approved = first_word.startswith("APPROVE")
    return approved, text


def parse_plan(planner_answer: str) -> list[str]:
    """Extract the list of sub-task strings from the planner's reply.

    The planner is told to reply with a JSON list of short questions.
    It may wrap that list in prose or code fences, so we locate the
    outermost [ ... ] and parse just that part.

    Returns a list of sub-task strings (empty list if nothing parseable).
    """
    text = planner_answer.strip()

    # Find the JSON list inside the reply
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    list_text = text[start:end + 1]
    try:
        steps = json.loads(list_text)
    except json.JSONDecodeError:
        return []

    # Keep only the string items, drop anything else
    clean_steps = []
    for step in steps:
        if isinstance(step, str):
            clean_steps.append(step)
    return clean_steps
