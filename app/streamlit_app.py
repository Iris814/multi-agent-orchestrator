"""
Phase 10 capstone — the Streamlit web app.

A polished, animated UI that shows the multi-agent pipeline working in real
time: a hero header, a live agent panel where each agent lights up as it
runs, a graphviz diagram of the pipeline, sample question chips, and a
recorded conversation history.

All agent logic lives in app/pipeline.py; this file only handles
presentation, layout, and the live event-driven status updates.

Run from the project root:

    streamlit run app/streamlit_app.py
"""

import asyncio
import html
import json
import os
import sys
import time
from pathlib import Path

# --- Anchor to the project root so every relative path resolves ----------
ROOT = Path(__file__).resolve().parent
while not (ROOT / "data" / "retail.parquet").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.pipeline import run_pipeline
from orchestrator.evals import EvalRow, load_golden_set, score_answer, summarize
from orchestrator.memory import LongTermMemory, ShortTermMemory


# --- Constants ------------------------------------------------------------

MODELS = {
    "Haiku 4.5 — fast & cheap (dev)": "claude-haiku-4-5-20251001",
    "Sonnet 4.6 — stronger (prod)":   "claude-sonnet-4-6",
}
TRACES_PATH = "traces/traces.jsonl"

SAMPLE_QUESTIONS = [
    "Top 5 countries by revenue in 2011",
    "Compare top 3 countries — 2010 vs 2011",
    "What does 'churned' mean in our analysis?",
    "Top 3 products by quantity sold in 2011",
]

# The five agents shown in the live panel + pipeline graph.
AGENT_KEYS = ["planner", "retriever", "data", "critic", "judge"]


# --- Page config + CSS ----------------------------------------------------

st.set_page_config(
    page_title="Retail Analytics Orchestrator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* hide streamlit chrome for a more app-like feel */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* hero header */
    .hero {
        background: linear-gradient(120deg, #4338ca 0%, #6d28d9 45%, #be185d 100%);
        padding: 1.8rem 2rem;
        border-radius: 18px;
        margin: 0 0 1.4rem 0;
        color: white;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 70%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.1rem; font-weight: 700; margin: 0; color: white; line-height: 1.2;
    }
    .hero p {
        font-size: 0.95rem; opacity: 0.92; margin: 0.5rem 0 0 0; color: white;
        max-width: 720px;
    }
    .hero-pills { margin-top: 0.9rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .hero-pill {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.32);
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.76rem;
        color: white;
        backdrop-filter: blur(4px);
    }

    /* agent cards row */
    .agent-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
        margin: 0.4rem 0 0.5rem 0;
    }
    .agent-card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.85rem 0.7rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(.4,0,.2,1);
        position: relative;
    }
    .agent-card-active {
        border-color: #6366f1;
        background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%);
        animation: pulse 1.8s ease-in-out infinite;
        transform: translateY(-2px);
    }
    .agent-card-done {
        border-color: #10b981;
        background: linear-gradient(180deg, #ffffff 0%, #ecfdf5 100%);
    }
    .agent-card-failed {
        border-color: #f59e0b;
        background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%);
    }
    .agent-card-skipped { opacity: 0.45; }
    .agent-emoji { font-size: 1.7rem; line-height: 1; }
    .agent-name {
        font-weight: 600; font-size: 0.88rem; color: #111827; margin-top: 0.3rem;
    }
    .agent-badge {
        font-size: 0.72rem; margin-top: 0.18rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 0.2rem;
    }
    .agent-card-idle    .agent-badge { color: #6b7280; }
    .agent-card-active  .agent-badge { color: #4338ca; }
    .agent-card-done    .agent-badge { color: #047857; }
    .agent-card-failed  .agent-badge { color: #b45309; }
    .agent-card-skipped .agent-badge { color: #9ca3af; }
    .agent-card-active .agent-badge::before {
        content: ""; width: 8px; height: 8px; border-radius: 50%;
        background: #6366f1; animation: blink 1s ease-in-out infinite;
    }
    .agent-message {
        font-size: 0.72rem; color: #6b7280;
        margin-top: 0.4rem; min-height: 2.1em; line-height: 1.3;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.22); }
        50%      { box-shadow: 0 0 0 10px rgba(99, 102, 241, 0.06); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.35; transform: scale(0.85); }
    }

    /* sample chips */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border-radius: 999px !important;
        padding: 0.45rem 0.9rem !important;
        transition: all 0.15s !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        border-color: #6366f1 !important;
        color: #4338ca !important;
        background: #eef2ff !important;
    }

    /* result card */
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin: 0.5rem 0;
    }

    /* event timeline */
    .timeline {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-size: 0.78rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .timeline-row {
        display: flex; gap: 0.6rem;
        padding: 0.2rem 0;
        border-bottom: 1px dashed #e5e7eb;
    }
    .timeline-row:last-child { border-bottom: none; }
    .timeline-time { color: #9ca3af; flex-shrink: 0; width: 3.4em; }
    .timeline-stage { color: #4338ca; font-weight: 600; flex-shrink: 0; width: 8em; }
    .timeline-msg { color: #374151; }

    /* tool-call chips */
    .tool-chips {
        display: flex; flex-wrap: wrap; gap: 0.4rem;
        margin: 0.25rem 0 0.5rem 0;
    }
    .tool-chip {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #fef3c7;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
        font-family: ui-monospace, 'SF Mono', Monaco, monospace;
        font-size: 0.75rem;
        color: #78350f;
        animation: chip-in 0.25s ease-out;
    }
    .tool-chip-fade {
        background: #ffffff;
        border-color: #e5e7eb;
        color: #6b7280;
    }
    .tool-chip-agent {
        font-weight: 600;
        color: #b45309;
    }
    .tool-chip-name {
        font-weight: 600;
    }
    .tool-chip-args {
        color: #92400e;
        opacity: 0.85;
    }
    @keyframes chip-in {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* per-agent breakdown table */
    .breakdown { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .breakdown th, .breakdown td {
        text-align: left; padding: 0.5rem 0.7rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .breakdown th {
        background: #f8fafc; color: #475569;
        font-weight: 600; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .breakdown tr:last-child td { border-bottom: none; }
    .breakdown td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .breakdown .cost-bar {
        display: inline-block; height: 6px; border-radius: 3px;
        background: linear-gradient(90deg, #6366f1, #a78bfa);
        margin-right: 0.4rem; vertical-align: middle;
    }

    /* section heading */
    .section-h {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        font-weight: 600;
        margin: 1rem 0 0.4rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Session state --------------------------------------------------------

if "short_term" not in st.session_state:
    st.session_state.short_term = ShortTermMemory(max_turns=6)
if "history" not in st.session_state:
    # Each item: PipelineResult, newest last.
    st.session_state.history = []
if "queued_question" not in st.session_state:
    st.session_state.queued_question = None

long_term = LongTermMemory()


def run_async(coro):
    return asyncio.run(coro)


# --- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model = MODELS[model_label]

    max_top_n = st.slider(
        "Approval gate · max rows",
        min_value=10, max_value=500, value=100, step=10,
        help="Phase 7 — the human-in-the-loop gate denies queries asking "
             "for more rows than this.",
    )
    max_revisions = st.slider(
        "Max critic revisions",
        min_value=0, max_value=2, value=1,
        help="Phase 2 — how many times the DataAgent may revise after the "
             "critic rejects.",
    )
    use_rag = st.toggle(
        "RAG · inject definitions", value=True,
        help="Phase 5 — retrieve from data/definitions.md and ground answers.",
    )
    use_guardrails = st.toggle(
        "Guardrails on final answer", value=True,
        help="Phase 8 — deterministic checks + an LLM judge before display.",
    )

    st.divider()

    has_auth = bool(
        os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    )
    if has_auth:
        st.success("Auth token loaded")
    else:
        st.error("No auth token in `.env`")

    st.divider()

    st.markdown("### 🧠 Long-term memory")
    facts = long_term.recall()
    if facts:
        for fact in facts:
            st.caption(f"• {fact}")
    else:
        st.caption("_No saved facts yet._")
    cols = st.columns(2)
    if cols[0].button("🗑️ Clear LTM", use_container_width=True) and facts:
        long_term.path.unlink(missing_ok=True)
        st.rerun()
    if cols[1].button("🧹 New chat", use_container_width=True):
        st.session_state.short_term = ShortTermMemory(max_turns=6)
        st.session_state.history = []
        st.rerun()


# --- Hero header ----------------------------------------------------------

st.markdown(
    """
<div class="hero">
    <h1>📊 Retail Analytics Orchestrator</h1>
    <p>Watch a multi-agent pipeline answer your question in real time —
    a Planner splits it, a DataAgent queries the warehouse, a Critic
    reviews the result, and Guardrails screen the output. Every run is
    traced, costed, and ready for evals.</p>
    <div class="hero-pills">
        <span class="hero-pill">⚡ Plan → Retrieve → Delegate → Reflect → Guard</span>
        <span class="hero-pill">🔍 Per-run trace</span>
        <span class="hero-pill">📐 Eval-driven</span>
        <span class="hero-pill">🛠️ Claude Agent SDK</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# --- Tabs -----------------------------------------------------------------

tab_ask, tab_arch, tab_history, tab_evals = st.tabs(
    ["💬 Ask", "🏗️ Architecture", "📈 Run history", "✅ Evals"]
)


# --------------------------------------------------------------------------
# Helpers — agent state, panel rendering, pipeline graph
# --------------------------------------------------------------------------

def fresh_agents_state(use_rag: bool, use_guardrails: bool) -> dict:
    """Initial state for the five agent cards before a run starts."""
    return {
        "planner": {
            "emoji": "🧭", "name": "Planner",
            "status": "idle",
            "message": "Splits questions into sub-tasks",
        },
        "retriever": {
            "emoji": "🔎", "name": "Retriever",
            "status": "idle" if use_rag else "skipped",
            "message": "RAG over definitions" if use_rag else "Disabled",
        },
        "data": {
            "emoji": "🛒", "name": "DataAgent",
            "status": "idle",
            "message": "Queries the retail tool",
        },
        "critic": {
            "emoji": "🕵️", "name": "Critic",
            "status": "idle",
            "message": "Reviews each answer",
        },
        "judge": {
            "emoji": "🛡️", "name": "Judge",
            "status": "idle" if use_guardrails else "skipped",
            "message": "Guardrails layer" if use_guardrails else "Disabled",
        },
    }


def render_agent_row(state: dict) -> str:
    """Render the 5-agent card row as HTML.

    Output is intentionally single-line with no whitespace between tags —
    Streamlit's CommonMark renderer treats a blank line inside an HTML
    block as a block boundary, after which 4-space-indented content
    becomes a code block. So no newlines, no indentation.
    """
    badges = {
        "idle":    "Idle",
        "active":  "Running",
        "done":    "✓ Done",
        "failed":  "⚠ Issue",
        "skipped": "— Skipped",
    }
    cards = []
    for key in AGENT_KEYS:
        agent = state[key]
        status = agent["status"]
        cards.append(
            f'<div class="agent-card agent-card-{status}">'
            f'<div class="agent-emoji">{agent["emoji"]}</div>'
            f'<div class="agent-name">{html.escape(agent["name"])}</div>'
            f'<div class="agent-badge">{badges[status]}</div>'
            f'<div class="agent-message">{html.escape(agent["message"])}</div>'
            f'</div>'
        )
    return f'<div class="agent-row">{"".join(cards)}</div>'


def render_architecture_svg() -> str:
    """A 3-tier system architecture diagram that accurately reflects the
    real runtime flow of the orchestrator.

    Tier 1 (top)    — Interface:    User · Memory · Final Answer
    Tier 2 (middle) — Orchestrator: 5 sub-agents in sequence + Trace log
                                    (Planner → Retriever → DataAgent →
                                    Critic → Judge), with the
                                    revision-loop arc above
    Tier 3 (bottom) — Tools layer:  Approval Gate (HITL) → MCP server →
                                    Retail Parquet
                                    (the gate sits between DataAgent and
                                    MCP — this is the human-in-the-loop)

    Connectors are labeled with the protocol/action that's actually
    happening across that edge ("question", "context", "tool call",
    "✓ allow", "reads", "⟲ on REJECT", "stream", "log").

    Pure SVG; static (no animation); built on one line with no blank lines
    so Streamlit's CommonMark renderer treats it as a single HTML block.
    """
    W, H = 1240, 740

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;border-radius:22px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
    )

    # ---- defs: gradients, filters, arrow markers, font/style classes ----
    parts.append(
        '<defs>'
        '<linearGradient id="bg-arch" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#f0fdf4"/>'
        '<stop offset="55%" stop-color="#f5f3ff"/>'
        '<stop offset="100%" stop-color="#fdf2f8"/>'
        '</linearGradient>'
        '<linearGradient id="band-orch" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#ede9fe" stop-opacity="0.7"/>'
        '<stop offset="100%" stop-color="#fce7f3" stop-opacity="0.55"/>'
        '</linearGradient>'
        '<linearGradient id="band-tools" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#dbeafe" stop-opacity="0.55"/>'
        '<stop offset="100%" stop-color="#dcfce7" stop-opacity="0.45"/>'
        '</linearGradient>'
        '<radialGradient id="halo-data" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#86efac" stop-opacity="0.6"/>'
        '<stop offset="70%" stop-color="#86efac" stop-opacity="0.0"/>'
        '</radialGradient>'
        '<radialGradient id="halo-gate" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#fde68a" stop-opacity="0.7"/>'
        '<stop offset="70%" stop-color="#fde68a" stop-opacity="0.0"/>'
        '</radialGradient>'
        '<filter id="card-shadow-arch" x="-30%" y="-30%" width="160%" height="160%">'
        '<feDropShadow dx="0" dy="5" stdDeviation="10" '
        'flood-color="#0f172a" flood-opacity="0.07"/>'
        '</filter>'
        '<marker id="arch-arr" markerWidth="9" markerHeight="9" '
        'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#64748b"/>'
        '</marker>'
        '<marker id="arch-arr-amber" markerWidth="9" markerHeight="9" '
        'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#d97706"/>'
        '</marker>'
        '<marker id="arch-arr-green" markerWidth="9" markerHeight="9" '
        'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L0,6 L8,3 z" fill="#059669"/>'
        '</marker>'
        '<style>'
        ".arch-title{font-family:Georgia,'Times New Roman',serif;"
        "font-weight:600;fill:#1e293b;letter-spacing:0.01em;}"
        ".arch-tier{font-family:-apple-system,'Segoe UI',system-ui,sans-serif;"
        "font-size:11px;font-weight:700;fill:#94a3b8;letter-spacing:0.18em;}"
        ".arch-label{font-family:-apple-system,'Segoe UI',system-ui,sans-serif;"
        "font-weight:600;fill:#0f172a;}"
        ".arch-sub{font-family:-apple-system,'Segoe UI',system-ui,sans-serif;"
        "font-size:10.5px;fill:#64748b;}"
        ".arch-edge-label{font-family:-apple-system,'Segoe UI',system-ui,sans-serif;"
        "font-size:11px;font-weight:600;fill:#475569;}"
        ".arch-card{fill:white;stroke:#e2e8f0;stroke-width:1;}"
        ".arch-card-focal{fill:white;stroke:#10b981;stroke-width:1.5;}"
        ".arch-card-gate{fill:white;stroke:#f59e0b;stroke-width:1.5;}"
        ".arch-mono{font-family:ui-monospace,'SF Mono',Menlo,monospace;"
        "font-size:10px;fill:#94a3b8;}"
        '</style>'
        '</defs>'
    )

    # ---- pastel background ----
    parts.append(f'<rect width="{W}" height="{H}" rx="22" fill="url(#bg-arch)"/>')

    # ---- title + subtitle ----
    parts.append(
        '<text x="620" y="48" text-anchor="middle" font-size="26" '
        'class="arch-title">The Orchestrator — System Architecture</text>'
        '<text x="620" y="72" text-anchor="middle" font-size="12.5" '
        'class="arch-sub" opacity="0.85">'
        'How a single user question flows through plan · retrieve · '
        'delegate · reflect · guard</text>'
    )

    # ---- tier labels (left rail) ----
    parts.append(
        '<text x="20" y="140" class="arch-tier">INTERFACE</text>'
        '<text x="20" y="340" class="arch-tier">ORCHESTRATOR</text>'
        '<text x="20" y="610" class="arch-tier">TOOLS &amp; DATA</text>'
    )

    # ---- orchestrator band (drawn before its contents) ----
    parts.append(
        '<rect x="40" y="280" width="1160" height="220" rx="22" '
        'fill="url(#band-orch)" stroke="#e2e8f0" stroke-width="1" '
        'stroke-dasharray="0"/>'
        '<text x="64" y="306" class="arch-mono">orchestrate( plan → retrieve '
        '→ delegate → reflect → guard )</text>'
    )

    # ---- tool-layer band (subtle background grouping) ----
    parts.append(
        '<rect x="380" y="565" width="660" height="120" rx="18" '
        'fill="url(#band-tools)" stroke="#e2e8f0" stroke-width="1"/>'
    )

    # ============== Connectors (drawn before cards) ==============

    # 1. User → Orchestrator band (vertical, with "question" label).
    parts.append(
        '<path d="M 155 195 L 155 280" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
    )
    # 2. Memory ↔ Orchestrator band (bidirectional, two close arrows).
    parts.append(
        '<path d="M 612 195 L 612 280" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
        '<path d="M 628 280 L 628 195" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
    )
    # 3. Orchestrator band → Final Answer (up, "stream" label).
    parts.append(
        '<path d="M 1085 280 L 1085 195" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
    )

    # 4–7. Chain inside the orchestrator: Planner → Retriever → DataAgent →
    # Critic → Judge → Trace (all at y=400, the vertical center of the
    # agent cards). The last edge into Trace is dashed because Trace is
    # observability, not part of the answer path.
    chain_y = 400
    parts.append(
        f'<path d="M 254 {chain_y} L 280 {chain_y}" stroke="#64748b" '
        'stroke-width="1.5" fill="none" marker-end="url(#arch-arr)"/>'
        f'<path d="M 439 {chain_y} L 465 {chain_y}" stroke="#64748b" '
        'stroke-width="1.5" fill="none" marker-end="url(#arch-arr)"/>'
        f'<path d="M 650 {chain_y} L 680 {chain_y}" stroke="#64748b" '
        'stroke-width="1.5" fill="none" marker-end="url(#arch-arr)"/>'
        f'<path d="M 845 {chain_y} L 875 {chain_y}" stroke="#64748b" '
        'stroke-width="1.5" fill="none" marker-end="url(#arch-arr)"/>'
        f'<path d="M 1040 {chain_y} L 1070 {chain_y}" stroke="#94a3b8" '
        'stroke-width="1.5" stroke-dasharray="4,3" fill="none" '
        'marker-end="url(#arch-arr)"/>'
    )

    # 8. Revision arc (Critic top → DataAgent top, amber dashed).
    # Critic top center: (760, 340). DataAgent top center: (557, 340).
    parts.append(
        '<path d="M 760 340 Q 658 290 557 340" stroke="#d97706" '
        'stroke-width="2" stroke-dasharray="6,4" fill="none" '
        'marker-end="url(#arch-arr-amber)"/>'
    )

    # 9. DataAgent → HITL Gate (down arrow with label "tool call").
    parts.append(
        '<path d="M 557 466 L 557 565" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
    )

    # 10. HITL Gate → MCP Server (right with green allow arrow).
    parts.append(
        '<path d="M 652 625 L 700 625" stroke="#059669" stroke-width="2" '
        'fill="none" marker-end="url(#arch-arr-green)"/>'
    )
    # And a small "deny" dashed arrow curving back up from gate to DataAgent.
    parts.append(
        '<path d="M 462 595 Q 380 530 452 466" stroke="#d97706" '
        'stroke-width="1.5" stroke-dasharray="4,3" fill="none" '
        'marker-end="url(#arch-arr-amber)"/>'
    )

    # 11. MCP Server → Retail Parquet (right, "reads").
    parts.append(
        '<path d="M 870 625 L 905 625" stroke="#64748b" stroke-width="1.5" '
        'fill="none" marker-end="url(#arch-arr)"/>'
    )

    # ============== Connector text labels ==============
    parts.append(
        # tier 1 labels
        '<text x="170" y="240" class="arch-edge-label">question</text>'
        '<text x="486" y="240" class="arch-edge-label">context</text>'
        '<text x="640" y="240" class="arch-edge-label">save</text>'
        '<text x="1100" y="240" class="arch-edge-label">stream</text>'
        # judge → trace
        '<text x="1015" y="386" class="arch-edge-label">log</text>'
        # revision arc label
        '<text x="658" y="282" text-anchor="middle" font-size="11.5" '
        'font-weight="700" fill="#b45309">⟲ on REJECT</text>'
        # DataAgent down arrow
        '<text x="572" y="520" class="arch-edge-label">tool call</text>'
        # gate → mcp
        '<text x="676" y="614" text-anchor="middle" font-size="10.5" '
        'font-weight="700" fill="#047857">✓ allow</text>'
        # deny path
        '<text x="395" y="528" text-anchor="middle" font-size="10.5" '
        'font-weight="700" fill="#b45309">✗ deny</text>'
        # mcp → parquet
        '<text x="887" y="614" text-anchor="middle" class="arch-edge-label">reads</text>'
    )

    # ============== Tier 1 — Interface cards ==============
    # User
    parts.append(
        '<g transform="translate(80 95)">'
        '<rect width="150" height="100" rx="18" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="75" y="48" text-anchor="middle" font-size="30">👤</text>'
        '<text x="75" y="78" text-anchor="middle" font-size="14" '
        'class="arch-label">User</text>'
        '</g>'
    )
    # Memory (with subtitle)
    parts.append(
        '<g transform="translate(545 95)">'
        '<rect width="150" height="100" rx="18" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="75" y="42" text-anchor="middle" font-size="26">🧠</text>'
        '<text x="75" y="69" text-anchor="middle" font-size="13" '
        'class="arch-label">Memory</text>'
        '<text x="75" y="86" text-anchor="middle" class="arch-sub">'
        'short-term · long-term</text>'
        '</g>'
    )
    # Final Answer
    parts.append(
        '<g transform="translate(1010 95)">'
        '<rect width="150" height="100" rx="18" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="75" y="48" text-anchor="middle" font-size="30">📝</text>'
        '<text x="75" y="78" text-anchor="middle" font-size="14" '
        'class="arch-label">Final Answer</text>'
        '</g>'
    )

    # ============== Tier 2 — Orchestrator: 5 sub-agents + Trace ==============
    # Generic card helper inline — define x positions of each agent card top-left.
    # Vertical: y=340, h=120 → center at y=400.
    # Planner
    parts.append(
        '<g transform="translate(90 340)">'
        '<rect width="164" height="120" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="82" y="42" text-anchor="middle" font-size="26">🧭</text>'
        '<text x="82" y="73" text-anchor="middle" font-size="13" '
        'class="arch-label">Planner</text>'
        '<text x="82" y="92" text-anchor="middle" class="arch-sub">'
        'decompose question</text>'
        '<text x="82" y="108" text-anchor="middle" class="arch-mono">'
        'PlannerAgent</text>'
        '</g>'
    )
    # Retriever
    parts.append(
        '<g transform="translate(280 340)">'
        '<rect width="159" height="120" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="79" y="42" text-anchor="middle" font-size="26">🔎</text>'
        '<text x="79" y="73" text-anchor="middle" font-size="13" '
        'class="arch-label">Retriever</text>'
        '<text x="79" y="92" text-anchor="middle" class="arch-sub">'
        'TF-IDF RAG · cite</text>'
        '<text x="79" y="108" text-anchor="middle" class="arch-mono">'
        'rag.py</text>'
        '</g>'
    )
    # DataAgent (focal — green border + halo)
    parts.append(
        '<g transform="translate(465 340)">'
        '<circle cx="92" cy="60" r="55" fill="url(#halo-data)"/>'
        '<rect width="185" height="120" rx="16" class="arch-card-focal" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="92" y="42" text-anchor="middle" font-size="28">🛒</text>'
        '<text x="92" y="73" text-anchor="middle" font-size="14" '
        'class="arch-label" font-weight="700">DataAgent</text>'
        '<text x="92" y="92" text-anchor="middle" class="arch-sub">'
        'calls the MCP tool</text>'
        '<text x="92" y="108" text-anchor="middle" class="arch-mono">'
        'DATA_PROMPT + tool</text>'
        '</g>'
    )
    # Critic
    parts.append(
        '<g transform="translate(680 340)">'
        '<rect width="165" height="120" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="82" y="42" text-anchor="middle" font-size="26">🕵️</text>'
        '<text x="82" y="73" text-anchor="middle" font-size="13" '
        'class="arch-label">Critic</text>'
        '<text x="82" y="92" text-anchor="middle" class="arch-sub">'
        'reflection · APPROVE / REJECT</text>'
        '<text x="82" y="108" text-anchor="middle" class="arch-mono">'
        'CriticAgent</text>'
        '</g>'
    )
    # Judge
    parts.append(
        '<g transform="translate(875 340)">'
        '<rect width="165" height="120" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="82" y="42" text-anchor="middle" font-size="26">🛡️</text>'
        '<text x="82" y="73" text-anchor="middle" font-size="13" '
        'class="arch-label">Judge</text>'
        '<text x="82" y="92" text-anchor="middle" class="arch-sub">'
        'guardrails · LLM judge</text>'
        '<text x="82" y="108" text-anchor="middle" class="arch-mono">'
        'JudgeAgent</text>'
        '</g>'
    )
    # Trace log card (right edge of orchestrator band)
    parts.append(
        '<g transform="translate(1070 340)">'
        '<rect width="120" height="120" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)" stroke-dasharray="4,3"/>'
        '<text x="60" y="42" text-anchor="middle" font-size="26">📊</text>'
        '<text x="60" y="73" text-anchor="middle" font-size="13" '
        'class="arch-label">Trace</text>'
        '<text x="60" y="92" text-anchor="middle" class="arch-sub">'
        'observability</text>'
        '<text x="60" y="108" text-anchor="middle" class="arch-mono">'
        'traces.jsonl</text>'
        '</g>'
    )

    # ============== Tier 3 — Tools & Data ==============
    # Approval Gate — explicit human-in-the-loop (amber border + halo + 👤)
    parts.append(
        '<g transform="translate(462 575)">'
        '<circle cx="95" cy="50" r="48" fill="url(#halo-gate)"/>'
        '<rect width="190" height="100" rx="16" class="arch-card-gate" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="95" y="40" text-anchor="middle" font-size="22">👤 🚦</text>'
        '<text x="95" y="68" text-anchor="middle" font-size="13" '
        'class="arch-label" font-weight="700">Approval Gate</text>'
        '<text x="95" y="86" text-anchor="middle" class="arch-sub">'
        'human-in-the-loop · can_use_tool</text>'
        '</g>'
    )
    # MCP Server
    parts.append(
        '<g transform="translate(700 575)">'
        '<rect width="170" height="100" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="85" y="40" text-anchor="middle" font-size="24">🔌</text>'
        '<text x="85" y="68" text-anchor="middle" font-size="13" '
        'class="arch-label">MCP Server</text>'
        '<text x="85" y="86" text-anchor="middle" class="arch-mono">'
        'retail_server.py</text>'
        '</g>'
    )
    # Retail Parquet (the data)
    parts.append(
        '<g transform="translate(905 575)">'
        '<rect width="130" height="100" rx="16" class="arch-card" '
        'filter="url(#card-shadow-arch)"/>'
        '<text x="65" y="40" text-anchor="middle" font-size="24">💾</text>'
        '<text x="65" y="68" text-anchor="middle" font-size="13" '
        'class="arch-label">Retail</text>'
        '<text x="65" y="86" text-anchor="middle" class="arch-mono">'
        'retail.parquet</text>'
        '</g>'
    )

    parts.append('</svg>')
    return "".join(parts)


def render_pipeline_svg(state: dict) -> str:
    """Render the pipeline as an animated SVG flow chart.

    Five rounded nodes in a row, connected by arrows. The active node gets
    a pulsing indigo ring + blinking status dot; done nodes get a green
    checkmark badge; failed nodes get an amber warning. The edge feeding
    the active node animates a dashed "data flowing" pattern. A curved
    revision-loop arc above the row lights up amber when the critic rejects.
    """
    palette = {
        "idle":    {"border": "#cbd5e1", "fill": "#f8fafc", "text": "#64748b"},
        "active":  {"border": "#6366f1", "fill": "#eef2ff", "text": "#4338ca"},
        "done":    {"border": "#10b981", "fill": "#ecfdf5", "text": "#047857"},
        "failed":  {"border": "#f59e0b", "fill": "#fffbeb", "text": "#b45309"},
        "skipped": {"border": "#e5e7eb", "fill": "#fafafa", "text": "#9ca3af"},
    }
    emoji_map = {
        "planner": "🧭", "retriever": "🔎", "data": "🛒",
        "critic":  "🕵️", "judge": "🛡️",
    }
    name_map = {
        "planner":   "Planner",
        "retriever": "Retriever",
        "data":      "DataAgent",
        "critic":    "Critic",
        "judge":     "Judge",
    }

    # Layout (SVG coords) — 5 nodes in a row.
    NODE_W, NODE_H = 168, 92
    NODE_Y = 110
    # left edge of each node:
    NODE_X = [20, 220, 420, 620, 820]
    VIEW_W = NODE_X[-1] + NODE_W + 20   # 1008
    VIEW_H = 250

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {VIEW_W} {VIEW_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%; height:auto; background:transparent;">'
    )

    # Defs — arrowhead markers, drop shadow, and the keyframes used below.
    parts.append("""
        <defs>
            <marker id="arr" markerWidth="10" markerHeight="10" refX="9" refY="3"
                    orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/>
            </marker>
            <marker id="arr-active" markerWidth="10" markerHeight="10" refX="9" refY="3"
                    orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0 L0,6 L9,3 z" fill="#6366f1"/>
            </marker>
            <marker id="arr-amber" markerWidth="10" markerHeight="10" refX="9" refY="3"
                    orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0 L0,6 L9,3 z" fill="#f59e0b"/>
            </marker>
            <marker id="arr-green" markerWidth="10" markerHeight="10" refX="9" refY="3"
                    orient="auto" markerUnits="userSpaceOnUse">
                <path d="M0,0 L0,6 L9,3 z" fill="#10b981"/>
            </marker>
            <filter id="node-shadow" x="-10%" y="-10%" width="120%" height="130%">
                <feDropShadow dx="0" dy="2" stdDeviation="3"
                              flood-color="#0f172a" flood-opacity="0.08"/>
            </filter>
            <style>
                @keyframes pp-pulse {
                    0%, 100% { opacity: 0.45; stroke-width: 2; }
                    50%      { opacity: 0.05; stroke-width: 10; }
                }
                @keyframes pp-blink {
                    0%, 100% { opacity: 1; transform: scale(1); }
                    50%      { opacity: 0.3; transform: scale(1.4); }
                }
                @keyframes pp-flow {
                    to { stroke-dashoffset: -28; }
                }
                .pp-pulse-ring { animation: pp-pulse 1.6s ease-in-out infinite; }
                .pp-blink-dot {
                    animation: pp-blink 1s ease-in-out infinite;
                    transform-origin: center;
                    transform-box: fill-box;
                }
                .pp-edge-flow { animation: pp-flow 0.9s linear infinite; }
                .pp-node-label {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                }
            </style>
        </defs>
    """)

    # ----- Edges (drawn first so nodes overlay them) -----
    for i in range(len(AGENT_KEYS) - 1):
        x1 = NODE_X[i] + NODE_W       # right edge of node i
        x2 = NODE_X[i + 1]            # left edge of node i+1
        y  = NODE_Y + NODE_H / 2

        next_status = state[AGENT_KEYS[i + 1]]["status"]
        this_status = state[AGENT_KEYS[i]]["status"]
        is_active_edge = (next_status == "active")
        is_done_edge   = this_status in ("done", "skipped") and next_status in ("done", "skipped", "active", "failed")

        if is_active_edge:
            color, dash, cls, marker = "#6366f1", "6,4", "pp-edge-flow", "arr-active"
        elif is_done_edge:
            color, dash, cls, marker = "#10b981", "none", "", "arr-green"
        else:
            color, dash, cls, marker = "#cbd5e1", "none", "", "arr"

        parts.append(
            f'<line x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" '
            f'stroke-dasharray="{dash}" class="{cls}" '
            f'marker-end="url(#{marker})" />'
        )

    # ----- Revision-loop arc (Critic → DataAgent) -----
    crit_top_x = NODE_X[3] + NODE_W / 2   # center top of Critic
    data_top_x = NODE_X[2] + NODE_W / 2   # center top of DataAgent
    is_revising = state["critic"]["status"] == "failed"

    arc_color = "#f59e0b" if is_revising else "#e2e8f0"
    arc_marker = "arr-amber" if is_revising else "arr"
    arc_cls    = "pp-edge-flow" if is_revising else ""

    # Arc goes UP from Critic top, curves over to DataAgent top.
    arc_mid_x = (crit_top_x + data_top_x) / 2
    arc_mid_y = NODE_Y - 70
    parts.append(
        f'<path d="M {crit_top_x:.0f},{NODE_Y} '
        f'Q {arc_mid_x:.0f},{arc_mid_y} {data_top_x:.0f},{NODE_Y}" '
        f'stroke="{arc_color}" stroke-width="2" stroke-dasharray="6,4" '
        f'fill="none" class="{arc_cls}" marker-end="url(#{arc_marker})"/>'
    )
    # Label on the arc — bright when revising, muted otherwise.
    label_color = "#b45309" if is_revising else "#94a3b8"
    parts.append(
        f'<text x="{arc_mid_x:.0f}" y="{arc_mid_y - 4}" '
        f'text-anchor="middle" font-size="11" font-weight="600" '
        f'fill="{label_color}" class="pp-node-label">'
        f'{"⟲ revising" if is_revising else "⟲ on reject"}'
        f'</text>'
    )

    # ----- Nodes -----
    for i, key in enumerate(AGENT_KEYS):
        agent = state[key]
        status = agent["status"]
        p = palette[status]
        x = NODE_X[i]

        # Outer pulsing ring for active node.
        if status == "active":
            parts.append(
                f'<rect x="{x - 5}" y="{NODE_Y - 5}" '
                f'width="{NODE_W + 10}" height="{NODE_H + 10}" '
                f'rx="16" fill="none" stroke="#6366f1" '
                f'class="pp-pulse-ring"/>'
            )

        # The main node rectangle.
        parts.append(
            f'<rect x="{x}" y="{NODE_Y}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="13" fill="{p["fill"]}" stroke="{p["border"]}" stroke-width="2" '
            f'filter="url(#node-shadow)"/>'
        )

        # Emoji.
        parts.append(
            f'<text x="{x + NODE_W / 2:.0f}" y="{NODE_Y + 32}" '
            f'text-anchor="middle" font-size="24">{emoji_map[key]}</text>'
        )
        # Agent name.
        parts.append(
            f'<text x="{x + NODE_W / 2:.0f}" y="{NODE_Y + 58}" '
            f'text-anchor="middle" font-size="13" font-weight="700" '
            f'fill="#111827" class="pp-node-label">{name_map[key]}</text>'
        )
        # Status message (truncated).
        msg = agent["message"] or ""
        if len(msg) > 26:
            msg = msg[:25] + "…"
        parts.append(
            f'<text x="{x + NODE_W / 2:.0f}" y="{NODE_Y + 78}" '
            f'text-anchor="middle" font-size="11" '
            f'fill="{p["text"]}" class="pp-node-label">'
            f'{html.escape(msg)}</text>'
        )

        # Status badge — top-right corner.
        badge_x = x + NODE_W - 14
        badge_y = NODE_Y + 14
        if status == "done":
            parts.append(
                f'<circle cx="{badge_x}" cy="{badge_y}" r="10" fill="#10b981"/>'
                f'<text x="{badge_x}" y="{badge_y + 4}" text-anchor="middle" '
                f'font-size="12" font-weight="700" fill="white">✓</text>'
            )
        elif status == "failed":
            parts.append(
                f'<circle cx="{badge_x}" cy="{badge_y}" r="10" fill="#f59e0b"/>'
                f'<text x="{badge_x}" y="{badge_y + 4}" text-anchor="middle" '
                f'font-size="13" font-weight="700" fill="white">!</text>'
            )
        elif status == "active":
            parts.append(
                f'<circle cx="{badge_x}" cy="{badge_y}" r="6" fill="#6366f1" '
                f'class="pp-blink-dot"/>'
            )
        elif status == "skipped":
            parts.append(
                f'<circle cx="{badge_x}" cy="{badge_y}" r="9" fill="#e5e7eb" '
                f'stroke="#cbd5e1" stroke-width="1"/>'
                f'<text x="{badge_x}" y="{badge_y + 4}" text-anchor="middle" '
                f'font-size="13" font-weight="700" fill="#9ca3af">−</text>'
            )

    # Anchor labels at the very left and right of the row.
    parts.append(
        f'<text x="0" y="{NODE_Y + NODE_H + 28}" font-size="11" '
        f'fill="#94a3b8" class="pp-node-label">↳ user question</text>'
    )
    parts.append(
        f'<text x="{VIEW_W}" y="{NODE_Y + NODE_H + 28}" font-size="11" '
        f'fill="#94a3b8" text-anchor="end" class="pp-node-label">'
        f'final answer ↦</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def render_timeline(events: list, start_time: float) -> str:
    """Render the live event log as a scrolling timeline."""
    rows = []
    for stage, msg, t in events[-12:]:
        rows.append(
            f'<div class="timeline-row">'
            f'<span class="timeline-time">{t - start_time:5.1f}s</span>'
            f'<span class="timeline-stage">{html.escape(stage)}</span>'
            f'<span class="timeline-msg">{html.escape(msg)}</span>'
            f"</div>"
        )
    return f'<div class="timeline">{"".join(rows)}</div>'


def _format_tool_args(args: dict, max_len: int = 60) -> str:
    """Render a tool's kwargs as `k=v, k=v` truncated to max_len."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            v_repr = f'"{v}"'
        else:
            v_repr = str(v)
        parts.append(f"{k}={v_repr}")
    s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def render_tool_chips(tool_calls: list) -> str:
    """Render the running list of tool calls as animated chips.

    The most recent chip is highlighted; older ones fade to a neutral style.
    """
    if not tool_calls:
        return (
            '<div class="tool-chips">'
            '<span style="color: #9ca3af; font-size: 0.78rem;">'
            "No tools called yet"
            "</span></div>"
        )

    chips = []
    last = len(tool_calls) - 1
    for i, call in enumerate(tool_calls):
        agent = call.get("agent", "?")
        tool = call.get("tool", "?")
        # Strip the MCP prefix for readability — "mcp__retail__query_retail"
        # becomes just "query_retail".
        if "__" in tool:
            tool = tool.rsplit("__", 1)[-1]
        args = _format_tool_args(call.get("input") or {})

        cls = "tool-chip" if i == last else "tool-chip tool-chip-fade"
        chips.append(
            f'<span class="{cls}">'
            f'<span class="tool-chip-agent">{html.escape(agent)}</span>'
            f'→<span class="tool-chip-name">{html.escape(tool)}</span>'
            f'<span class="tool-chip-args">({html.escape(args)})</span>'
            "</span>"
        )
    return f'<div class="tool-chips">{"".join(chips)}</div>'


def render_agent_breakdown(breakdown: list) -> str:
    """Render the per-agent token + cost breakdown as a styled HTML table."""
    if not breakdown:
        return ""

    max_cost = max((row["cost_usd"] for row in breakdown), default=0.0) or 1.0
    body = []
    for row in breakdown:
        bar_width_px = int(60 * row["cost_usd"] / max_cost)
        body.append(
            f"<tr>"
            f"<td><strong>{html.escape(row['name'])}</strong></td>"
            f"<td class='num'>{row['n_runs']}</td>"
            f"<td class='num'>{row['input_tokens']:,}</td>"
            f"<td class='num'>{row['cached_input_tokens']:,}</td>"
            f"<td class='num'>{row['output_tokens']:,}</td>"
            f"<td class='num'>"
            f"<span class='cost-bar' style='width:{bar_width_px}px'></span>"
            f"${row['cost_usd']:.4f}</td>"
            f"</tr>"
        )
    return (
        "<table class='breakdown'>"
        "<thead><tr>"
        "<th>Agent</th><th>Runs</th><th>Input</th>"
        "<th>Cached</th><th>Output</th><th>Cost</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


# --------------------------------------------------------------------------
# Render past results (no live panel — just the final card)
# --------------------------------------------------------------------------

def render_past_result(result):
    """Render a completed PipelineResult in a polished card."""
    stages = result.stages
    trace = result.trace

    # Answer body in a card.
    st.markdown(result.answer)

    # Guardrail verdict.
    if stages.guardrail_passed:
        st.success("✅ Guardrails passed — safe to ship")
    else:
        st.error("⚠️ Guardrails flagged this answer:")
        for v in stages.guardrail_violations:
            st.write(f"• {v}")

    # Metric tiles.
    cols = st.columns(5)
    cols[0].metric("Latency",     f"{trace.latency_ms / 1000:.1f}s")
    cols[1].metric("Cost",        f"${trace.cost_usd:.4f}")
    cols[2].metric("Delegations", trace.n_delegations)
    cols[3].metric("Tool calls",  trace.n_tool_calls)
    cols[4].metric("Cache hit",   f"{trace.cache_hit_rate:.0%}")

    # Per-agent cost breakdown — surfaces "where the money went".
    if stages.agent_breakdown:
        st.markdown(
            '<div class="section-h">💰 Where the cost went</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_agent_breakdown(stages.agent_breakdown),
                    unsafe_allow_html=True)

    with st.expander("🔬 Pipeline detail — how this answer was built"):
        st.markdown("**1. Plan** — the question split into sub-tasks:")
        if stages.plan:
            for i, step in enumerate(stages.plan, start=1):
                st.write(f"{i}. {step}")
        else:
            st.caption("_No plan produced._")

        st.markdown("**2. RAG retrieval** — business definitions injected:")
        if stages.rag_citations:
            st.write(", ".join(f"`{c}`" for c in stages.rag_citations))
        else:
            st.caption("_No business terms touched._")

        st.markdown("**3. Tool calls** — every invocation the agents made:")
        if stages.tool_calls:
            for call in stages.tool_calls:
                agent = call.get("agent", "?")
                tool = call.get("tool", "?")
                if "__" in tool:
                    tool = tool.rsplit("__", 1)[-1]
                args = _format_tool_args(call.get("input") or {}, max_len=120)
                st.markdown(
                    f"- **{agent}** → `{tool}({args})`"
                )
        else:
            st.caption("_No tool calls in this run._")

        st.markdown("**4. Approval gate** — decisions per call:")
        if stages.gate_log:
            for decision, tool_name, top_n in stages.gate_log:
                emoji = "✅" if decision == "ALLOW" else "🚫"
                st.write(f"{emoji} {decision}: `{tool_name}` (top_n={top_n})")
        else:
            st.caption("_The agent made no tool calls._")

        st.markdown("**5. Critic review** — reflection verdict:")
        if stages.critic_approved:
            st.write("✅ APPROVED")
        else:
            st.write("❌ REJECTED")
        if stages.critic_verdict:
            st.caption(stages.critic_verdict)
        st.write(f"Revisions made: **{stages.n_revisions}**")

        st.markdown("**6. Token usage (totals):**")
        st.write(
            f"Input: **{trace.input_tokens:,}**  ·  "
            f"Cached input: **{trace.cached_input_tokens:,}**  ·  "
            f"Output: **{trace.output_tokens:,}**"
        )


# ==========================================================================
# Tab 1 — Ask
# ==========================================================================

with tab_ask:
    # Sample question chips, only when chat is empty.
    if not st.session_state.history:
        st.markdown('<div class="section-h">Try one of these</div>', unsafe_allow_html=True)
        chip_cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if chip_cols[i].button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.queued_question = q
                st.rerun()

    # Past conversation turns.
    for past in st.session_state.history:
        with st.chat_message("user"):
            st.write(past.question)
        with st.chat_message("assistant"):
            render_past_result(past)

    # Pull queued chip question OR the chat input.
    question = st.session_state.queued_question or st.chat_input(
        "Ask about the retail data — e.g. 'top 5 countries by revenue in 2011'"
    )
    if st.session_state.queued_question:
        st.session_state.queued_question = None

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if not has_auth:
                st.error("Add an auth token to `.env` first.")
            else:
                st.markdown(
                    '<div class="section-h">🤖 Agents at work</div>',
                    unsafe_allow_html=True,
                )

                # --- Build the placeholders the events will write into.
                agents_state = fresh_agents_state(use_rag, use_guardrails)
                event_log = []
                tool_calls = []   # live list, mutated by on_event
                start_time = time.perf_counter()

                graph_box = st.empty()
                cards_box = st.empty()

                st.markdown(
                    '<div class="section-h">🛠️ Tool calls (live)</div>',
                    unsafe_allow_html=True,
                )
                chips_box = st.empty()

                col_t, col_s = st.columns([3, 2])
                with col_t:
                    st.markdown(
                        '<div class="section-h">Event timeline</div>',
                        unsafe_allow_html=True,
                    )
                    timeline_box = st.empty()
                with col_s:
                    st.markdown(
                        '<div class="section-h">🔍 Pipeline state</div>',
                        unsafe_allow_html=True,
                    )
                    state_box = st.empty()

                # Initial render.
                graph_box.markdown(
                    render_pipeline_svg(agents_state), unsafe_allow_html=True,
                )
                cards_box.markdown(
                    render_agent_row(agents_state), unsafe_allow_html=True,
                )
                chips_box.markdown(
                    render_tool_chips(tool_calls), unsafe_allow_html=True,
                )
                timeline_box.markdown(
                    render_timeline(event_log, start_time), unsafe_allow_html=True,
                )
                state_box.json({"status": "waiting for events…"}, expanded=False)
                # Track the latest pipeline state so the inspector can show it.
                live_state = {"plan": [], "citations": [], "sub_answers_count": 0,
                              "n_revisions": 0, "critic": None, "guardrails": None}

                # --- The event callback that paints the live UI.
                def on_event(name: str, data: dict):
                    s = agents_state  # short alias
                    now = time.perf_counter()
                    msg = ""

                    if name == "plan.start":
                        s["planner"]["status"] = "active"
                        s["planner"]["message"] = "Analyzing the question…"
                        msg = "Splitting the question into sub-tasks"
                    elif name == "plan.done":
                        s["planner"]["status"] = "done"
                        n = data.get("n_steps", 1)
                        s["planner"]["message"] = f"{n} sub-task{'s' if n != 1 else ''}"
                        msg = f"Plan ready: {n} sub-task{'s' if n != 1 else ''}"
                        live_state["plan"] = data.get("plan") or []
                    elif name == "retrieve.start":
                        if s["retriever"]["status"] != "skipped":
                            s["retriever"]["status"] = "active"
                            s["retriever"]["message"] = "Searching definitions…"
                        msg = "Retrieving business definitions"
                    elif name == "retrieve.done":
                        if s["retriever"]["status"] != "skipped":
                            cited = data.get("citations") or []
                            s["retriever"]["status"] = "done"
                            s["retriever"]["message"] = (
                                f"Cited {', '.join(cited)}" if cited
                                else "No matching definitions"
                            )
                            msg = f"Retrieval done — cited {len(cited)} doc(s)"
                            live_state["citations"] = cited
                    elif name == "delegate.start":
                        s["data"]["status"] = "active"
                        s["data"]["message"] = "Querying retail dataset…"
                        msg = "DataAgent starting"
                    elif name == "step.start":
                        idx = data.get("step_idx", 1)
                        total = data.get("total_steps", 1)
                        s["data"]["message"] = f"Sub-task {idx}/{total}: querying…"
                        msg = f"Sub-task {idx}/{total}"
                    elif name == "step.done":
                        idx = data.get("step_idx", 1)
                        total = data.get("total_steps", 1)
                        msg = f"Sub-task {idx}/{total} answered"
                        live_state["sub_answers_count"] = idx
                    elif name == "tool.use":
                        agent = data.get("agent", "?")
                        tool = data.get("tool", "?")
                        # Trim the MCP namespace prefix for display.
                        display_tool = tool.rsplit("__", 1)[-1] if "__" in tool else tool
                        tool_calls.append({
                            "agent": agent, "tool": tool,
                            "input": data.get("input") or {},
                        })
                        args = _format_tool_args(data.get("input") or {}, max_len=80)
                        msg = f"{agent} → {display_tool}({args})"
                    elif name == "delegate.done":
                        n = data.get("n_steps", 1)
                        s["data"]["status"] = "done"
                        s["data"]["message"] = f"Answered {n} sub-task(s)"
                        msg = f"DataAgent done ({n} run(s))"
                    elif name == "critic.start":
                        s["critic"]["status"] = "active"
                        attempt = data.get("attempt", 1)
                        s["critic"]["message"] = f"Reviewing (attempt {attempt})…"
                        msg = f"Critic reviewing (attempt {attempt})"
                    elif name == "critic.done":
                        approved = data.get("approved", False)
                        reason = data.get("reason", "")
                        if approved:
                            s["critic"]["status"] = "done"
                            s["critic"]["message"] = "✓ Approved"
                            msg = "Critic approved"
                        else:
                            s["critic"]["status"] = "failed"
                            s["critic"]["message"] = "✗ Rejected — revising"
                            msg = "Critic rejected"
                        live_state["critic"] = {"approved": approved, "reason": reason}
                    elif name == "revise.start":
                        s["data"]["status"] = "active"
                        s["data"]["message"] = "Revising after critic feedback…"
                        msg = "DataAgent revising"
                    elif name == "revise.done":
                        s["data"]["status"] = "done"
                        s["data"]["message"] = "Revised answer ready"
                        msg = "Revision complete"
                        # Critic will run again — reset critic so its next
                        # active state shows correctly.
                        s["critic"]["status"] = "idle"
                        live_state["n_revisions"] = data.get("revision_idx", live_state["n_revisions"] + 1)
                    elif name == "guard.start":
                        if s["judge"]["status"] != "skipped":
                            s["judge"]["status"] = "active"
                            s["judge"]["message"] = "Running checks + LLM judge…"
                        msg = "Guardrails running"
                    elif name == "guard.done":
                        passed = data.get("passed", True)
                        violations = data.get("violations") or []
                        if s["judge"]["status"] != "skipped":
                            if passed:
                                s["judge"]["status"] = "done"
                                s["judge"]["message"] = "✓ Passed every check"
                            else:
                                s["judge"]["status"] = "failed"
                                s["judge"]["message"] = f"{len(violations)} violation(s)"
                            msg = (
                                "Guardrails passed" if passed
                                else f"Guardrails flagged ({len(violations)})"
                            )
                        live_state["guardrails"] = {"passed": passed, "violations": violations}
                    elif name == "pipeline.done":
                        msg = (
                            f"Pipeline complete · "
                            f"{data.get('latency_ms', 0) / 1000:.1f}s · "
                            f"${data.get('cost_usd', 0):.4f}"
                        )

                    if msg:
                        event_log.append((name, msg, now))

                    # Re-render every live element.
                    graph_box.markdown(
                        render_pipeline_svg(agents_state),
                        unsafe_allow_html=True,
                    )
                    cards_box.markdown(
                        render_agent_row(agents_state), unsafe_allow_html=True,
                    )
                    chips_box.markdown(
                        render_tool_chips(tool_calls), unsafe_allow_html=True,
                    )
                    timeline_box.markdown(
                        render_timeline(event_log, start_time),
                        unsafe_allow_html=True,
                    )
                    state_box.json({
                        "plan":              live_state["plan"],
                        "rag_citations":     live_state["citations"],
                        "sub_answers_done":  live_state["sub_answers_count"],
                        "tool_calls_so_far": len(tool_calls),
                        "n_revisions":       live_state["n_revisions"],
                        "critic":            live_state["critic"],
                        "guardrails":        live_state["guardrails"],
                    }, expanded=False)

                memory_context = st.session_state.short_term.as_context()
                result = None
                try:
                    result = run_async(
                        run_pipeline(
                            question=question,
                            model=model,
                            memory_context=memory_context,
                            max_top_n=max_top_n,
                            max_revisions=max_revisions,
                            use_rag=use_rag,
                            use_guardrails=use_guardrails,
                            on_event=on_event,
                        )
                    )
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")

                if result is not None:
                    st.markdown(
                        '<div class="section-h">📝 Final answer</div>',
                        unsafe_allow_html=True,
                    )
                    render_past_result(result)

                    # Persist for the chat history + traces log.
                    st.session_state.short_term.add("user", question)
                    st.session_state.short_term.add("assistant", result.answer)
                    st.session_state.history.append(result)
                    result.trace.append_jsonl(TRACES_PATH)


# ==========================================================================
# Tab 2 — Architecture (static overview)
# ==========================================================================

with tab_arch:
    st.markdown(render_architecture_svg(), unsafe_allow_html=True)

    st.markdown(
        '<div class="section-h" style="margin-top:1.6rem">'
        "Reading the diagram (top to bottom)</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    with cols[0]:
        st.markdown("**Tier 1 — Interface**")
        st.caption(
            "**👤 User** sends a question; the **🧠 Memory** layer prepends "
            "the short-term conversation buffer and any long-term facts; the "
            "**📝 Final Answer** is streamed back. The orchestrator never "
            "talks to the user directly except through these three nodes."
        )

    with cols[1]:
        st.markdown("**Tier 2 — Orchestrator**")
        st.caption(
            "Five specialists run in sequence — **🧭 Planner** decomposes, "
            "**🔎 Retriever** does TF-IDF RAG over the definitions, "
            "**🛒 DataAgent** (focal) calls the MCP tool, **🕵️ Critic** "
            "reflects (and can ⟲ **REJECT** back to the DataAgent), "
            "**🛡️ Judge** runs the guardrail layer. Every run is appended to "
            "**📊 Trace** (`traces.jsonl`) for evals + observability."
        )

    with cols[2]:
        st.markdown("**Tier 3 — Tools & Data**")
        st.caption(
            "When DataAgent calls a tool, the request goes through the "
            "**👤🚦 Approval Gate** — the **human-in-the-loop** checkpoint "
            "that can ✗ **deny** oversized queries before they run. "
            "Approved calls hit the **🔌 MCP Server** (`retail_server.py`), "
            "which reads from **💾 retail.parquet** and returns rows."
        )

    st.markdown(
        '<div class="section-h" style="margin-top:1.6rem">'
        "Patterns wired through the stack</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
| Pattern | Where it lives | Phase |
|---|---|---|
| Orchestrator–worker | `app/pipeline.py` · `orchestrator/agents.py` | 2 |
| Reflection (LLM critic + revision loop) | `CRITIC_PROMPT` + revise step | 2, 8 |
| Planning & decomposition | `PlannerAgent` + `parse_plan` | 3 |
| Hierarchical memory (short + long term) | `orchestrator/memory.py` | 4 |
| RAG with citations | `orchestrator/rag.py` · TF-IDF | 5 |
| MCP (in-process + stdio) | `mcp_servers/retail_server.py` | 6 |
| Human-in-the-loop approval gate | `make_approval_gate` · `can_use_tool` | 7 |
| Deterministic + LLM-judge guardrails | `orchestrator/guardrails.py` | 8 |
| Eval-driven development | `orchestrator/evals.py` + `tests/golden.json` | 9 |
| Cost & latency observability | `orchestrator/observability.py` · `Trace` | 1+ |
        """
    )


# ==========================================================================
# Tab 3 — Run history (reads traces.jsonl)
# ==========================================================================

with tab_history:
    st.markdown(
        '<div class="section-h">Every recorded run</div>', unsafe_allow_html=True,
    )
    st.caption(
        f"Each agent run appends a structured Trace to `{TRACES_PATH}` — "
        "the Phase 1+ observability layer that makes Phase 9 evals possible."
    )

    trace_file = Path(TRACES_PATH)
    if not trace_file.exists():
        st.info("No traces yet — ask a question on the **Ask** tab to begin.")
    else:
        rows = []
        for line in trace_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not rows:
            st.info("Trace file is empty.")
        else:
            traces_df = pd.DataFrame(rows)

            cols = st.columns(4)
            cols[0].metric("Total runs",   len(traces_df))
            cols[1].metric("Avg latency",  f"{traces_df['latency_ms'].mean() / 1000:.1f}s")
            cols[2].metric("Avg cost",     f"${traces_df['cost_usd'].mean():.4f}")
            cols[3].metric("Total cost",   f"${traces_df['cost_usd'].sum():.4f}")

            show_cols = [
                "question", "model", "latency_ms", "cost_usd",
                "n_delegations", "n_tool_calls", "passed",
            ]
            present = [c for c in show_cols if c in traces_df.columns]
            st.dataframe(traces_df[present], width="stretch", hide_index=True)


# ==========================================================================
# Tab 3 — Evals (run the golden set)
# ==========================================================================

with tab_evals:
    st.markdown(
        '<div class="section-h">Golden-set evaluation</div>', unsafe_allow_html=True,
    )
    st.caption(
        "Run the agent against a fixed list of questions with known-correct "
        "answers, then score each one (Phase 9). Substring scoring — the "
        "expected facts must all appear in the answer."
    )

    try:
        golden = load_golden_set()
    except Exception as exc:
        golden = []
        st.error(f"Could not load the golden set: {exc}")

    if golden:
        st.write(f"Golden set: **{len(golden)} cases**")

        if not has_auth:
            st.warning("Add an auth token to `.env` to run the eval.")
        elif st.button("▶ Run eval", type="primary"):
            eval_rows = []
            progress = st.progress(0.0, text="Running golden set…")

            for index, case in enumerate(golden):
                try:
                    result = run_async(
                        run_pipeline(
                            question=case.question,
                            model=model,
                            max_top_n=max_top_n,
                            max_revisions=max_revisions,
                            use_rag=use_rag,
                            use_guardrails=use_guardrails,
                        )
                    )
                    missing = score_answer(result.answer, case.expected_substrings)
                    eval_rows.append(EvalRow(
                        case_id=case.case_id,
                        passed=(missing == []),
                        missing=missing,
                        latency_ms=result.trace.latency_ms,
                        cost_usd=result.trace.cost_usd,
                    ))
                except Exception as exc:
                    eval_rows.append(EvalRow(
                        case_id=case.case_id,
                        passed=False,
                        missing=[f"error: {exc}"],
                        latency_ms=0,
                        cost_usd=0.0,
                    ))
                progress.progress(
                    (index + 1) / len(golden),
                    text=f"Case {index + 1}/{len(golden)} — {case.case_id}",
                )

            progress.empty()
            summary = summarize(eval_rows)

            cols = st.columns(3)
            cols[0].metric("Pass rate",   f"{summary['pass_rate']:.0%}")
            cols[1].metric("Avg latency", f"{summary['avg_latency_ms'] / 1000:.1f}s")
            cols[2].metric("Total cost",  f"${summary['total_cost_usd']:.4f}")

            detail = []
            for row in eval_rows:
                detail.append({
                    "case":      row.case_id,
                    "passed":    "✅" if row.passed else "❌",
                    "missing":   ", ".join(row.missing),
                    "latency_s": round(row.latency_ms / 1000, 1),
                    "cost_usd":  row.cost_usd,
                })
            st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
