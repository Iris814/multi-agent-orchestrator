"""
Phase 10 capstone pipeline — wires all nine patterns into one callable.

Given a question, `run_pipeline()` runs the full orchestrated flow:

  1. PLAN      — a PlannerAgent splits the question into sub-tasks (Phase 3).
  2. RETRIEVE  — for each sub-task, TF-IDF RAG pulls relevant business
                 definitions and injects them into the prompt (Phase 5).
  3. DELEGATE  — a DataAgent answers each sub-task with the retail MCP tool
                 (Phases 2 + 6), behind a human-in-the-loop approval gate
                 that blocks oversized queries (Phase 7).
  4. REVIEW    — a CriticAgent approves or rejects the combined answer; if
                 rejected, the DataAgent revises once (Phase 2, reflection).
  5. GUARD     — deterministic checks plus an LLM judge screen the final
                 answer before it is shown (Phase 8).
  6. RECORD    — the whole run is summarized in one Trace: latency, tokens,
                 cost, cache hit rate (Phase 1, observability).

Memory (Phase 4) is owned by the caller — the Streamlit app keeps a
ShortTermMemory across questions and passes its rendered text in as
`memory_context`, so a follow-up like "what about 2011?" still makes sense.

The Streamlit UI only does presentation; all agent logic lives here so it
can be exercised from a plain Python script or a test.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    create_sdk_mcp_server,
    tool,
)

from orchestrator import tools
from orchestrator.agents import run_agent, parse_plan, parse_verdict
from orchestrator.guardrails import run_deterministic_guardrails
from orchestrator.observability import MODEL_PRICING, Trace, Timer
from orchestrator.rag import Retriever, load_definitions, format_context


# --------------------------------------------------------------------------
# Result types — everything the UI needs to render one pipeline run.
# --------------------------------------------------------------------------

@dataclass
class StageLog:
    """A blow-by-blow record of what each stage of the pipeline did."""

    plan: list[str] = field(default_factory=list)
    sub_answers: list[str] = field(default_factory=list)
    rag_citations: list[str] = field(default_factory=list)
    gate_log: list[tuple] = field(default_factory=list)
    critic_verdict: str = ""
    critic_approved: bool = False
    n_revisions: int = 0
    guardrail_passed: bool = True
    guardrail_violations: list[str] = field(default_factory=list)
    # Tool calls the agents made, in order. Each item:
    #   {"agent": "data", "tool": "query_retail", "input": {...}}
    tool_calls: list[dict] = field(default_factory=list)
    # Per-agent token & cost breakdown, populated at the end of the run.
    # Each item: {"name", "n_runs", "input_tokens", "output_tokens",
    #             "cached_input_tokens", "cost_usd"}
    agent_breakdown: list[dict] = field(default_factory=list)


@dataclass
class PipelineResult:
    """The full outcome of one `run_pipeline()` call."""

    question: str
    answer: str
    trace: Trace
    stages: StageLog


# --------------------------------------------------------------------------
# The retail tool, exposed over an in-process MCP server (Phases 2 + 6).
# --------------------------------------------------------------------------

@tool(
    "query_retail",
    description=(
        "Query the retail transactions dataset to return the top N entries "
        "ranked by a metric. Use this for any 'top N' question about products, "
        "countries, or customers. Arguments: year (optional), country "
        "(optional - OMIT to include all countries), top_n (default 10), "
        "group_by (one of 'StockCode', 'Country', 'Customer ID'), metric (one "
        "of 'revenue', 'Quantity'). Returns a list of dicts, one row each."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "year":     {"type": "integer", "description": "Calendar year filter, e.g. 2011"},
            "country":  {"type": "string",  "description": "Optional country filter. OMIT to include all countries."},
            "top_n":    {"type": "integer", "description": "How many rows to return"},
            "group_by": {"type": "string",  "description": "'StockCode', 'Country', or 'Customer ID'"},
            "metric":   {"type": "string",  "description": "'revenue' or 'Quantity'"},
        },
        "required": [],
    },
)
async def query_retail_tool(args):
    """Adapter: the agent passes args as a dict; we call the pandas function."""
    rows = tools.query_retail(
        year=args.get("year"),
        country=args.get("country"),
        top_n=args.get("top_n", 10),
        group_by=args.get("group_by", "StockCode"),
        metric=args.get("metric", "revenue"),
    )
    return {"content": [{"type": "text", "text": json.dumps(rows, default=str)}]}


def make_retail_server():
    """Build a fresh in-process MCP server exposing the retail tool."""
    return create_sdk_mcp_server(
        name="retail",
        version="1.0.0",
        tools=[query_retail_tool],
    )


# --------------------------------------------------------------------------
# Human-in-the-loop approval gate (Phase 7).
# --------------------------------------------------------------------------

def make_approval_gate(max_top_n: int, gate_log: list):
    """Build a `can_use_tool` callback that blocks oversized queries.

    The SDK calls the returned function before every tool use. A query for
    more rows than `max_top_n` is treated as too expensive and denied; every
    decision is appended to `gate_log` so the UI can show what happened.
    """

    async def approval_gate(tool_name, tool_input, context):
        top_n = tool_input.get("top_n", 10)

        if top_n > max_top_n:
            gate_log.append(("DENY", tool_name, top_n))
            return PermissionResultDeny(
                message=(
                    f"top_n={top_n} exceeds the limit of {max_top_n} - "
                    "blocked by the approval gate"
                )
            )

        gate_log.append(("ALLOW", tool_name, top_n))
        return PermissionResultAllow()

    return approval_gate


# --------------------------------------------------------------------------
# Agent system prompts.
# --------------------------------------------------------------------------

PLANNER_PROMPT = (
    "You are the planning agent for a retail analytics system. Every question "
    "you receive is about a retail transactions dataset (sales by country, "
    "product, customer, and year). Break the incoming question into the "
    "smallest set of independent, self-contained sub-questions needed to "
    "answer it - the next agent sees each sub-question alone, with no other "
    "context. If the question is already simple, return it unchanged as a "
    "one-item list. Never ask for clarification - always produce a plan. "
    'Reply with ONLY a JSON list of strings, e.g. ["sub-question 1", '
    '"sub-question 2"].'
)

DATA_PROMPT = (
    "You are a retail data analyst. Answer the question you are given using "
    "the query_retail tool. Call the tool with the correct filters, then "
    "present the result as a clean markdown table. Only filter by country if "
    "the question explicitly names a country. Report ONLY the figures the "
    "tool returns - do NOT add your own percentages, calculations, or extra "
    "commentary, and never invent numbers. If a tool call is blocked by the "
    "approval gate, tell the user plainly that the request was denied and why."
)

CRITIC_PROMPT = (
    "You are a critic agent. Your sole job is to review the DataAgent's "
    "answer to a question. Start your reply with a single bare word - APPROVE "
    "or REJECT - as the very first word, with no quotes or punctuation before "
    "it. If you REJECT, follow it with one sentence explaining what is wrong. "
    "Check: does the answer actually address the question, are any numbers "
    "implausible, and is anything required missing? This is UK-dominated "
    "retail data, so a top-countries answer that omits the United Kingdom is "
    "almost certainly wrong."
)

JUDGE_PROMPT = (
    "You are a quality-control judge for a retail analytics assistant. You "
    "are given a QUESTION and an ANSWER. Decide whether the answer is good "
    "enough to show the user. Reply with APPROVE or REJECT as the very first "
    "word, with no quotes or punctuation before it. If you REJECT, follow it "
    "with one sentence explaining what is wrong. Check that the answer "
    "actually addresses the question, is not empty or evasive, and contains "
    "no self-contradictory or clearly implausible claims. The figures come "
    "from a trusted internal data tool, so do NOT demand source citations - "
    "an uncited table of numbers is fine."
)


# --------------------------------------------------------------------------
# RAG retriever — built once, reused across calls.
# --------------------------------------------------------------------------

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Load (or return the cached) TF-IDF retriever over the definitions."""
    global _retriever
    if _retriever is None:
        documents = load_definitions()
        _retriever = Retriever(documents)
    return _retriever


def _retrieve_context(question: str, min_score: float = 0.3):
    """Retrieve relevant definitions for a question.

    Returns (context_text, cited_ids). If nothing scores above `min_score`
    the question does not touch a business term, so we retrieve nothing -
    injecting low-relevance text would only add noise. The 0.3 bar is tuned
    to the definitions corpus: a genuine term match (e.g. "churn" -> Churn)
    scores ~0.35-0.45, while a generic "top N revenue" question peaks ~0.28.
    """
    retriever = get_retriever()
    scored = retriever.retrieve(question, k=3)

    relevant = []
    for doc, score in scored:
        if score >= min_score:
            relevant.append((doc, score))

    if not relevant:
        return "", []

    context_text = format_context(relevant)
    cited_ids = []
    for doc, score in relevant:
        cited_ids.append(doc.doc_id)
    return context_text, cited_ids


# --------------------------------------------------------------------------
# The pipeline.
# --------------------------------------------------------------------------

def _agent_breakdown(all_runs, model: str) -> list[dict]:
    """Group AgentRuns by agent name and compute per-agent token + cost totals.

    Returns one row per agent, sorted by cost descending, so the UI can show
    "where the money went" in a single glance.
    """
    pricing = MODEL_PRICING.get(model, {})

    totals = defaultdict(lambda: {
        "n_runs": 0, "input": 0, "output": 0, "cached": 0,
    })
    for run in all_runs:
        t = totals[run.name]
        t["n_runs"] += 1
        t["input"] += run.input_tokens
        t["output"] += run.output_tokens
        t["cached"] += run.cached_input_tokens

    rows = []
    for agent_name, t in totals.items():
        cost = (
            t["input"]  / 1_000_000 * pricing.get("input", 0.0)
            + t["output"] / 1_000_000 * pricing.get("output", 0.0)
            + t["cached"] / 1_000_000 * pricing.get("cached_input", 0.0)
        )
        rows.append({
            "name":                agent_name,
            "n_runs":              t["n_runs"],
            "input_tokens":        t["input"],
            "output_tokens":       t["output"],
            "cached_input_tokens": t["cached"],
            "cost_usd":            round(cost, 6),
        })

    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return rows


def _combine(plan: list[str], sub_answers: list[str]) -> str:
    """Combine per-sub-task answers into one final answer."""
    if len(sub_answers) == 1:
        return sub_answers[0]

    parts = []
    for i in range(len(sub_answers)):
        if i < len(plan):
            heading = plan[i]
        else:
            heading = f"Step {i + 1}"
        parts.append(f"### {heading}\n{sub_answers[i]}")
    return "\n\n".join(parts)


def _emit(on_event, name: str, **payload):
    """Safely call the UI event callback — never let it break the pipeline."""
    if on_event is None:
        return
    try:
        on_event(name, payload)
    except Exception:
        pass


async def run_pipeline(
    question: str,
    model: str,
    memory_context: str = "",
    max_top_n: int = 100,
    max_revisions: int = 1,
    use_rag: bool = True,
    use_guardrails: bool = True,
    on_event: Callable[[str, dict], None] | None = None,
) -> PipelineResult:
    """Run the full plan -> retrieve -> delegate -> review -> guard flow.

    Parameters
    ----------
    question        the user's question
    model           model id, e.g. "claude-haiku-4-5-20251001"
    memory_context  rendered conversation memory to prepend (may be empty)
    max_top_n       the approval gate blocks queries larger than this
    max_revisions   how many times the DataAgent may revise after a reject
    use_rag         whether to retrieve and inject business definitions
    use_guardrails  whether to run the guardrail layer on the final answer
    """
    stages = StageLog()
    trace = Trace(model=model, question=question)
    gate_log: list = []

    # Every agent run, so token usage can be rolled into the one Trace.
    all_runs = []

    # The retail MCP server + the gated DataAgent options.
    retail_server = make_retail_server()
    approval_gate = make_approval_gate(max_top_n, gate_log)

    data_options = ClaudeAgentOptions(
        model=model,
        system_prompt=DATA_PROMPT,
        mcp_servers={"retail": retail_server},
        can_use_tool=approval_gate,
        max_turns=5,
    )
    planner_options = ClaudeAgentOptions(
        model=model,
        system_prompt=PLANNER_PROMPT,
        max_turns=2,
    )
    critic_options = ClaudeAgentOptions(
        model=model,
        system_prompt=CRITIC_PROMPT,
        max_turns=2,
    )
    judge_options = ClaudeAgentOptions(
        model=model,
        system_prompt=JUDGE_PROMPT,
        max_turns=2,
    )

    # Per-agent tool-call relay: every tool the agent invokes lands in
    # stages.tool_calls *and* fires a "tool.use" event so the UI can paint
    # chips in real time.
    def _tool_relay(agent_label: str):
        def _cb(tool_name: str, tool_input: dict):
            stages.tool_calls.append(
                {"agent": agent_label, "tool": tool_name, "input": tool_input}
            )
            _emit(on_event, "tool.use",
                  agent=agent_label, tool=tool_name, input=tool_input)
        return _cb

    with Timer() as timer:
        # --- Stage 1: PLAN ------------------------------------------------
        # Memory context goes to the planner so follow-up questions resolve.
        if memory_context.strip() != "":
            planner_input = (
                f"{memory_context}\n\nCURRENT QUESTION: {question}"
            )
        else:
            planner_input = question

        _emit(on_event, "plan.start")
        planner_run = await run_agent(
            "PlannerAgent", planner_input, planner_options,
            on_tool_use=_tool_relay("planner"),
        )
        all_runs.append(planner_run)

        plan = parse_plan(planner_run.answer)
        if not plan:
            # Planner produced nothing parseable - treat it as a single task.
            plan = [question]
        stages.plan = plan
        _emit(on_event, "plan.done", n_steps=len(plan), plan=plan)

        # --- Stage 2 + 3: RETRIEVE and DELEGATE ---------------------------
        sub_answers = []
        all_citations = []
        if use_rag:
            _emit(on_event, "retrieve.start")
        _emit(on_event, "delegate.start", n_steps=len(plan))
        for i, step in enumerate(plan):
            if use_rag:
                context_text, cited_ids = _retrieve_context(step)
            else:
                context_text, cited_ids = "", []

            for doc_id in cited_ids:
                if doc_id not in all_citations:
                    all_citations.append(doc_id)

            if context_text != "":
                data_prompt = (
                    f"{context_text}\n\n"
                    "Use the reference material above where it is relevant, "
                    f"then answer this question:\n{step}"
                )
            else:
                data_prompt = step

            _emit(on_event, "step.start", step_idx=i + 1, total_steps=len(plan), step=step)
            data_run = await run_agent(
                "DataAgent", data_prompt, data_options,
                on_tool_use=_tool_relay("data"),
            )
            all_runs.append(data_run)
            sub_answers.append(data_run.answer)
            _emit(on_event, "step.done", step_idx=i + 1, total_steps=len(plan))

        if use_rag:
            _emit(on_event, "retrieve.done", citations=all_citations)
        _emit(on_event, "delegate.done", n_steps=len(plan))

        stages.sub_answers = sub_answers
        stages.rag_citations = all_citations

        answer = _combine(plan, sub_answers)

        # --- Stage 4: REVIEW (reflection loop) ----------------------------
        approved = False
        for attempt in range(max_revisions + 1):
            _emit(on_event, "critic.start", attempt=attempt + 1)
            critic_prompt = f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer}"
            critic_run = await run_agent(
                "CriticAgent", critic_prompt, critic_options,
                on_tool_use=_tool_relay("critic"),
            )
            all_runs.append(critic_run)

            approved, reason = parse_verdict(critic_run.answer)
            stages.critic_verdict = reason
            stages.critic_approved = approved
            _emit(on_event, "critic.done", approved=approved, reason=reason, attempt=attempt + 1)

            if approved or attempt == max_revisions:
                break

            # Rejected - re-delegate to the DataAgent with the feedback.
            _emit(on_event, "revise.start", reason=reason)
            revision_prompt = (
                f"{question}\n\n"
                f"PREVIOUS ANSWER:\n{answer}\n\n"
                "A reviewer rejected the previous answer. Their feedback:\n"
                f"{reason}\n\n"
                "Read the feedback, fix the specific problem, and give a "
                "corrected answer."
            )
            data_run = await run_agent(
                "DataAgent", revision_prompt, data_options,
                on_tool_use=_tool_relay("data"),
            )
            all_runs.append(data_run)
            answer = data_run.answer
            stages.n_revisions += 1
            _emit(on_event, "revise.done", revision_idx=stages.n_revisions)

        # --- Stage 5: GUARD ----------------------------------------------
        if use_guardrails:
            _emit(on_event, "guard.start")
            det = run_deterministic_guardrails(answer)
            violations = list(det.violations)

            judge_prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
            judge_run = await run_agent(
                "JudgeAgent", judge_prompt, judge_options,
                on_tool_use=_tool_relay("judge"),
            )
            all_runs.append(judge_run)

            judge_approved, judge_reason = parse_verdict(judge_run.answer)
            if not judge_approved:
                violations.append(f"LLM judge rejected the answer: {judge_reason}")

            stages.guardrail_passed = (len(violations) == 0)
            stages.guardrail_violations = violations
            _emit(on_event, "guard.done", passed=stages.guardrail_passed, violations=violations)
        else:
            stages.guardrail_passed = True
            stages.guardrail_violations = []

    # --- Stage 6: RECORD --------------------------------------------------
    for run in all_runs:
        trace.input_tokens += run.input_tokens
        trace.output_tokens += run.output_tokens
        trace.cached_input_tokens += run.cached_input_tokens
        trace.n_tool_calls += run.n_tool_calls

    trace.n_delegations = len(all_runs)
    trace.latency_ms = timer.elapsed_ms
    trace.answer = answer
    trace.passed = stages.guardrail_passed and stages.critic_approved
    trace.compute_cost()

    stages.gate_log = gate_log
    stages.agent_breakdown = _agent_breakdown(all_runs, model)
    _emit(on_event, "pipeline.done", latency_ms=trace.latency_ms, cost_usd=trace.cost_usd)

    return PipelineResult(
        question=question,
        answer=answer,
        trace=trace,
        stages=stages,
    )
