"""
streamlit_app.py — Demo UI for the Self-Healing RAG pipeline.

Run with:
    streamlit run app/streamlit_app.py

Design system: "code dark + run green" — deep slate canvas, slate cards, a single
green accent, Fira Code / Fira Sans type, SVG icons (no emoji), soft depth.
Surfaces not just the answer but the full self-healing trace: the critic's verdict,
its reasoning, how many times the query was rewritten, the sources, and the exact
context chunks the answer was grounded in.
"""
import os
import sys

# Make ``src`` importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st  # noqa: E402

from src.graph.graph import answer_query  # noqa: E402

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🔁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG = "#0F172A"
SURFACE = "#1E293B"
SURFACE_2 = "#172033"
BORDER = "#334155"
ACCENT = "#22C55E"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"

VERDICT_STYLES = {
    "grounded": ("#22C55E", "rgba(34,197,94,0.12)"),
    "hallucinated": ("#EF4444", "rgba(239,68,68,0.12)"),
    "insufficient": ("#F59E0B", "rgba(245,158,11,0.12)"),
}

EXAMPLES = [
    "What does the RAGAS faithfulness metric measure?",
    "What does LangGraph enable that a LangChain chain cannot?",
    "What is the difference between dense and sparse retrieval?",
    "What embedding dimension does all-MiniLM-L6-v2 produce?",
]


# ── SVG icon set (Lucide-style, inline so they inherit currentColor) ──────────
def icon(name: str, size: int = 18) -> str:
    paths = {
        "refresh": '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 '
        '2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        "doc": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v5h5"/>',
        "loop": '<path d="M17 2.1 21 6l-3.9 3.9"/><path d="M3 12V9a3 3 0 0 1 3-3h15"/>'
        '<path d="M7 21.9 3 18l3.9-3.9"/><path d="M21 12v3a3 3 0 0 1-3 3H3"/>',
        "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 '
        '2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>'
    )


# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {{
  --bg: {BG}; --surface: {SURFACE}; --border: {BORDER};
  --accent: {ACCENT}; --text: {TEXT}; --muted: {MUTED};
}}

.stApp {{
  background:
    radial-gradient(900px 500px at 50% -10%, rgba(34,197,94,0.10), transparent 60%),
    radial-gradient(700px 400px at 100% 0%, rgba(56,189,248,0.06), transparent 55%),
    {BG};
  font-family: 'Fira Sans', system-ui, sans-serif;
}}

/* hide default streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer {{ display: none; }}
.block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 820px; }}

code, .mono {{ font-family: 'Fira Code', monospace; }}

/* ── hero ────────────────────────────────────────────── */
.hero {{ text-align: center; margin-bottom: 1.75rem; }}
.hero .badge {{
  display: inline-flex; align-items: center; gap: .5rem;
  font-family: 'Fira Code', monospace; font-size: .72rem; letter-spacing: .06em;
  text-transform: uppercase; color: {ACCENT};
  background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.30);
  padding: .35rem .8rem; border-radius: 999px; margin-bottom: 1rem;
}}
.hero h1 {{
  font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin: 0 0 .6rem;
  background: linear-gradient(180deg, #FFFFFF, #B6C2D4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero p {{ color: {MUTED}; font-size: 1.02rem; max-width: 560px; margin: 0 auto; line-height: 1.6; }}

.flow {{
  display: flex; flex-wrap: wrap; justify-content: center; gap: .4rem;
  margin: 1.1rem 0 .25rem; font-family: 'Fira Code', monospace; font-size: .76rem;
}}
.flow span {{ color: {TEXT}; background: {SURFACE}; border: 1px solid {BORDER};
  padding: .28rem .6rem; border-radius: 8px; }}
.flow .arrow {{ color: {MUTED}; border: none; background: none; padding: .28rem .1rem; }}

/* ── section label ───────────────────────────────────── */
.label {{
  display: flex; align-items: center; gap: .5rem; color: {MUTED};
  font-family: 'Fira Code', monospace; font-size: .74rem; letter-spacing: .08em;
  text-transform: uppercase; margin: .2rem 0 .6rem;
}}
.label svg {{ color: {ACCENT}; }}

/* ── cards ───────────────────────────────────────────── */
/* Radius scale (locked): cards 16px · sub-panels 12px · chips 8px · buttons 10px · pills full. */
@keyframes riseIn {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
.card {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
  padding: 1.3rem 1.4rem; box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  /* Motivated motion: the card rises in to acknowledge the pipeline produced output. */
  animation: riseIn .45s cubic-bezier(0.16, 1, 0.3, 1) both;
}}
.answer-card {{ font-size: 1.06rem; line-height: 1.7; color: {TEXT}; }}

.pill {{
  display: inline-flex; align-items: center; gap: .45rem; font-weight: 600;
  font-size: .82rem; padding: .35rem .75rem; border-radius: 999px; margin-bottom: .9rem;
}}

/* ── stat grid ───────────────────────────────────────── */
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin: .25rem 0 1rem; }}
.stat {{
  background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: .85rem 1rem; text-align: center;
}}
.stat .k {{ font-family: 'Fira Code', monospace; font-size: .68rem; letter-spacing: .06em;
  text-transform: uppercase; color: {MUTED}; }}
.stat .v {{ font-family: 'Fira Code', monospace; font-size: 1.25rem; font-weight: 600;
  color: {TEXT}; margin-top: .25rem; }}

.reason {{ color: {MUTED}; font-size: .92rem; line-height: 1.6;
  border-left: 2px solid {BORDER}; padding-left: .85rem; margin: .2rem 0 1rem; }}
.reason b {{ color: {TEXT}; }}

.src {{ display: inline-flex; align-items: center; gap: .4rem; font-family: 'Fira Code', monospace;
  font-size: .78rem; color: {TEXT}; background: {SURFACE_2}; border: 1px solid {BORDER};
  padding: .3rem .6rem; border-radius: 8px; margin: 0 .4rem .4rem 0; }}
.src svg {{ color: {ACCENT}; }}

.ctx {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT};
  border-radius: 10px; padding: .8rem .95rem; margin-bottom: .6rem;
  color: #CBD5E1; font-size: .9rem; line-height: 1.6; }}
.ctx .n {{ font-family: 'Fira Code', monospace; color: {ACCENT}; font-weight: 600; margin-right: .4rem; }}

/* ── inputs & buttons ────────────────────────────────── */
.stTextInput input, .stTextArea textarea {{
  background: {SURFACE} !important; border: 1px solid {BORDER} !important;
  border-radius: 12px !important; color: {TEXT} !important; font-size: 1rem !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: {ACCENT} !important; box-shadow: 0 0 0 3px rgba(34,197,94,0.18) !important;
}}
div.stButton > button {{
  border-radius: 10px; border: 1px solid {BORDER}; background: {SURFACE}; color: {TEXT};
  font-family: 'Fira Code', monospace; font-size: .82rem; font-weight: 500;
  transition: border-color .2s, color .2s, background .2s; cursor: pointer;
}}
div.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; background: {SURFACE_2}; }}
div.stButton > button[kind="primary"] {{
  background: {ACCENT}; border: 1px solid {ACCENT}; color: #06210F; font-weight: 700;
}}
div.stButton > button[kind="primary"]:hover {{ background: #1FB055; border-color: #1FB055; color: #06210F; }}

@media (max-width: 640px) {{ .hero h1 {{ font-size: 2rem; }} .stats {{ grid-template-columns: 1fr; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <div class="badge">{icon('loop', 14)} Self-Healing RAG</div>
  <h1>Answers that grade themselves.</h1>
  <p>Retrieves, reranks, generates, then <b>critiques its own answer</b>, rewriting
  the query on hallucination and refusing when context is thin.</p>
  <div class="flow">
    <span>Retrieve</span><span class="arrow">→</span>
    <span>Rerank</span><span class="arrow">→</span>
    <span>Generate</span><span class="arrow">→</span>
    <span>Critique</span><span class="arrow">↻</span>
    <span>Heal</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Input + example chips ─────────────────────────────────────────────────────
st.session_state.setdefault("question", "")


def _set_q(q: str) -> None:
    st.session_state.question = q


st.markdown(f'<div class="label">{icon("spark", 14)} Try an example</div>', unsafe_allow_html=True)
chip_cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    chip_cols[i % 2].button(
        example, key=f"ex_{i}", on_click=_set_q, args=(example,), use_container_width=True
    )

st.text_input(
    "Ask a question",
    key="question",
    placeholder="Ask anything about the indexed documents…",
    label_visibility="collapsed",
)
ask = st.button("Run the pipeline", type="primary", use_container_width=True)


# ── Run ───────────────────────────────────────────────────────────────────────
def render_result(result: dict) -> None:
    grounded = result["grounded"]
    color, bg = (ACCENT, "rgba(34,197,94,0.12)") if grounded else (
        "#F59E0B",
        "rgba(245,158,11,0.12)",
    )
    status_icon = icon("check") if grounded else icon("alert")
    status_text = "Grounded answer" if grounded else "Refused — could not ground"

    st.markdown(f'<div class="label">{icon("check", 14)} Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card">
  <span class="pill" style="color:{color}; background:{bg};">{status_icon} {status_text}</span>
  <div class="answer-card">{_escape(result["answer"])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Self-healing trace
    v = (result["critique"] or "n/a").lower()
    vcolor, vbg = VERDICT_STYLES.get(v, (MUTED, "rgba(148,163,184,0.12)"))

    st.markdown(
        f'<div class="label" style="margin-top:1.4rem">{icon("loop", 14)} Self-healing trace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="card">
  <div class="stats">
    <div class="stat"><div class="k">Verdict</div>
      <div class="v" style="color:{vcolor}">{v.upper()}</div></div>
    <div class="stat"><div class="k">Retries</div>
      <div class="v">{result["retries"]}</div></div>
    <div class="stat"><div class="k">Sources</div>
      <div class="v">{len(result["sources"])}</div></div>
  </div>
  <div class="reason"><b>Critic reasoning:</b> {_escape(result["critique_reason"] or "—")}</div>
  {_sources_html(result["sources"])}
  {_contexts_html(result["contexts"])}
</div>
""",
        unsafe_allow_html=True,
    )


def _escape(text: str) -> str:
    return (text or "").replace("<", "&lt;").replace(">", "&gt;")


def _sources_html(sources: list) -> str:
    if not sources:
        return ""
    chips = "".join(f'<span class="src">{icon("doc", 13)}{_escape(s)}</span>' for s in sources)
    return f'<div style="margin-bottom:.9rem">{chips}</div>'


def _contexts_html(contexts: list) -> str:
    if not contexts:
        return ""
    blocks = ""
    for i, ctx in enumerate(contexts, start=1):
        preview = _escape(ctx[:420] + ("…" if len(ctx) > 420 else ""))
        blocks += f'<div class="ctx"><span class="n">[{i}]</span>{preview}</div>'
    return (
        f'<div class="label" style="margin-top:.4rem">{icon("search", 13)} '
        f"Retrieved context</div>{blocks}"
    )


if ask and st.session_state.question.strip():
    try:
        with st.spinner("Retrieving, reranking, generating, then self-critiquing…"):
            result = answer_query(st.session_state.question)
        render_result(result)
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f"""
<div class="card" style="border-color:#EF4444">
  <span class="pill" style="color:#EF4444; background:rgba(239,68,68,0.12)">{icon('alert')} Pipeline error</span>
  <div class="answer-card mono" style="font-size:.9rem; color:#FCA5A5">{_escape(str(exc))}</div>
  <div class="reason" style="margin-top:.8rem">Is Qdrant running (<code>make qdrant</code>),
  the index built (<code>make ingest</code>), and are your API keys set in <code>.env</code>?</div>
</div>
""",
            unsafe_allow_html=True,
        )
elif ask:
    st.markdown(
        f'<div class="reason">{icon("alert", 14)} Enter a question or pick an example above.</div>',
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="text-align:center; margin-top:2.5rem; color:{MUTED}; font-family:'Fira Code',monospace; font-size:.74rem; line-height:1.9">
  Built with LangGraph, Qdrant, Cohere Rerank and RAGAS<br>
  <a href="https://github.com/fahadcs321" style="color:{ACCENT}; text-decoration:none">github.com/fahadcs321</a>
</div>
""",
    unsafe_allow_html=True,
)
