"""LLM-based signal discovery: workflow image, context-window data (200/500/1000) with side-by-side URLs."""
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from seminar.data import (
    DIR_CONTEXT_200,
    DIR_CONTEXT_500,
    DIR_CONTEXT_1000,
    PATH_SIGNALS_200,
    PATH_SIGNALS_500,
    PATH_SIGNALS_1000,
    REPO_ROOT,
)

CONTEXT_DISCOVERY = "discovery_set.json"
SIGNALS_PATHS = {"200": PATH_SIGNALS_200, "500": PATH_SIGNALS_500, "1000": PATH_SIGNALS_1000}
ROUNDS_PER_EXPERIMENT = 12


def _load_signals_jsonl(path) -> list[dict]:
    """Load one JSONL file; each line is {round, response}."""
    if not path or not path.exists():
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except Exception:
        return []
    return out


def _split_experiments(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """First 12 lines = Experiment 1, next 12 lines = Experiment 2 (by line order)."""
    exp1 = rows[:ROUNDS_PER_EXPERIMENT]
    exp2 = rows[ROUNDS_PER_EXPERIMENT : ROUNDS_PER_EXPERIMENT * 2]
    return exp1, exp2


def _render_one_round(row: dict) -> None:
    """Render one round's summary, signals, and caveats (for comparison view)."""
    resp = row.get("response") or {}
    st.caption(resp.get("summary", "")[:400] + ("..." if len(resp.get("summary", "")) > 400 else ""))
    for sig in (resp.get("signals") or [])[:3]:
        st.markdown(f"- **{sig.get('name', '—')}**: {sig.get('description', '')[:150]}...")
    caveats = resp.get("caveats") or []
    if caveats:
        st.caption("Caveats: " + "; ".join(c[:80] for c in caveats[:2]))


def _load_entries(context_dir: Path) -> list:
    """Load entries from context_*/discovery_set.json."""
    path = context_dir / CONTEXT_DISCOVERY
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data.get("entries", []) if isinstance(data, dict) else []


PAIRS_PER_ROUND = 6  # 3 reprint + 3 non_reprint per round


def _entries_for_round(entries: list, round_1based: int) -> list:
    """Return the 6 entries (GT pairs) used for this round."""
    start = (round_1based - 1) * PAIRS_PER_ROUND
    return entries[start : start + PAIRS_PER_ROUND]


def _pair_display_url(entry: dict, side: str) -> str | None:
    """Build ECCO URL for one side (src/dst) from entry; use precomputed _url or trs_start/end."""
    url_key = "src_url" if side == "src" else "dst_url"
    if entry.get(url_key):
        return entry[url_key]
    doc_id = entry.get("src_doc_id" if side == "src" else "dst_doc_id")
    start_key = "src_trs_start" if side == "src" else "dst_trs_start"
    end_key = "src_trs_end" if side == "src" else "dst_trs_end"
    start_val = entry.get(start_key)
    end_val = entry.get(end_key)
    if doc_id is not None and start_val is not None and end_val is not None:
        return f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}&offsetStart={start_val}&offsetEnd={end_val}&isAlreadyOctavified=0"
    return None


def _render_pair_viewer(entries: list, window_label: str, session_key: str):
    """One pair at a time: Prev/Next, then side-by-side src_url and dst_url."""
    if not entries:
        st.info(f"No entries for **{window_label}**.")
        return
    key_idx = f"{session_key}_idx"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    idx = max(0, min(st.session_state[key_idx], len(entries) - 1))
    st.session_state[key_idx] = idx
    pair = entries[idx]
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ Previous", key=f"{session_key}_prev") and idx > 0:
            st.session_state[key_idx] = idx - 1
            st.rerun()
    with col_info:
        st.caption(f"**{window_label}** — Pair **{idx + 1}** of **{len(entries)}** · {pair.get('reuse_type', '')} · src=`{pair.get('src_doc_id')}` dst=`{pair.get('dst_doc_id')}`")
    with col_next:
        if st.button("Next ▶", key=f"{session_key}_next") and idx < len(entries) - 1:
            st.session_state[key_idx] = idx + 1
            st.rerun()
    src_url = pair.get("src_url")
    dst_url = pair.get("dst_url")
    if not src_url and pair.get("src_doc_id") and pair.get("src_trs_start") is not None and pair.get("src_trs_end") is not None:
        src_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={pair['src_doc_id']}&offsetStart={pair['src_trs_start']}&offsetEnd={pair['src_trs_end']}&isAlreadyOctavified=0"
    if not dst_url and pair.get("dst_doc_id") and pair.get("dst_trs_start") is not None and pair.get("dst_trs_end") is not None:
        dst_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={pair['dst_doc_id']}&offsetStart={pair['dst_trs_start']}&offsetEnd={pair['dst_trs_end']}&isAlreadyOctavified=0"
    if src_url or dst_url:
        st.markdown("**Links:** " + (f"[Source (ECCO)]({src_url})" if src_url else "—") + " · " + (f"[Destination (ECCO)]({dst_url})" if dst_url else "—"))
        if src_url and dst_url:
            col_src, col_dst = st.columns(2)
            with col_src:
                st.caption("Source")
                components.iframe(src_url, height=420)
            with col_dst:
                st.caption("Destination")
                components.iframe(dst_url, height=420)
    else:
        st.warning("No src_url/dst_url or offset fields in this entry.")


def render():
    st.header("LLM-based Signal Discovery", divider="rainbow")
    st.markdown(
        "The naive-signal stage showed that **length** is useful for candidate reduction but not sufficient. "
        "We use an LLM to discover more informative **textual and structural signals**."
    )
    st.markdown("**Model**: Qwen3-30B-Instruct. **Context windows**: 200, 500, 1000 characters.")
    with st.expander("LLM settings (system prompt & query template)", expanded=False):
        st.markdown("**System prompt**")
        st.code(
            '''You are an expert research assistant for textual reuse analysis.

You will be given two groups of reuse pairs:
- Group A: reprint pairs
- Group B: non-reprint pairs

Your task is not to classify the pairs, but to identify 3–5 major signals that distinguish reprint from non-reprint across the examples.

You must focus on cross-sample patterns rather than pair-specific observations.

You must return a single valid JSON object only.
Do not write any explanation outside the JSON.
Do not use markdown.''',
            language="text",
        )
        st.markdown("**Query template** (placeholders: `{formatted_reprint_pairs}`, `{formatted_nonreprint_pairs}`)")
        st.code(
            r"""This is a signal discovery task for textual reuse.

You are given:
- Group A: 3 reprint pairs
- Group B: 3 non-reprint pairs

Each pair includes a source text and a destination text.
The reused span is marked with [REUSE_START] and [REUSE_END].

Your task:
- identify 3–5 major signals that distinguish reprint from non-reprint across these examples
- focus on cross-sample patterns
- include short textual evidence whenever possible
- indicate whether each signal overlaps with naive signals, refines them, is new, or is uncertain
- focus on textual and discourse properties rather than OCR noise or typographic corruption

Do not classify the pairs directly.
Return one JSON object only.

Group A — Reprint pairs
{formatted_reprint_pairs}

Group B — Non-reprint pairs
{formatted_nonreprint_pairs}

Output format:
{
  "summary": "...",
  "signals": [
    {
      "name": "...",
      "description": "...",
      "stability": "stable | sometimes | occasional",
      "relation_to_naive_signals": "overlaps_with_naive | refinement_of_naive | new | uncertain",
      "supporting_examples": [
        {
          "type": "reprint | non_reprint",
          "quote_or_observation": "..."
        }
      ]
    }
  ],
  "caveats": ["...", "..."]
}""",
            language="text",
        )
        st.caption("Helper: `coarse_reuse_type(reuse_type)` maps fine-grained reuse_type to a coarse label for the prompt (e.g. Unknown if empty).")
    st.markdown("---")

    st.markdown("### Workflow")
    workflow_image = REPO_ROOT / "image" / "signals_discovery_llm.png"
    if workflow_image.exists():
        st.image(str(workflow_image), use_container_width=True)
    else:
        st.caption(f"Image not found: `{workflow_image}`")

    st.markdown("---")
    st.markdown("### Context-window data (200 / 500 / 1000 char)")
    st.caption("Discovery set entries for the selected window. Default: 500; switch to 200 or 1000 to compare.")
    context_option = st.radio(
        "Context window",
        options=["500", "200", "1000"],
        index=0,
        horizontal=True,
        key="llm_context_window",
    )
    context_dirs = {"200": DIR_CONTEXT_200, "500": DIR_CONTEXT_500, "1000": DIR_CONTEXT_1000}
    context_dir = context_dirs[context_option]
    entries = _load_entries(context_dir)
    _render_pair_viewer(entries, f"{context_option} char", "llm_ctx")

    st.markdown("---")
    st.markdown("### Experiment results (signals discovery)")
    st.caption(
        "Two runs per context: Experiment 1 = no OCR mentioned in prompt, GT in order; "
        "Experiment 2 = OCR in prompt or different GT order. Each run has up to 12 rounds."
    )
    sig_context = st.radio(
        "Context window (results)",
        options=["200", "500", "1000"],
        index=1,
        horizontal=True,
        key="llm_signals_context",
    )
    path = SIGNALS_PATHS.get(sig_context)
    all_rows = _load_signals_jsonl(path) if path else []
    exp1_rows, exp2_rows = _split_experiments(all_rows)

    exp_choice = st.radio(
        "Experiment",
        options=["Experiment 1 (no OCR in prompt, GT in order)", "Experiment 2 (OCR in prompt / different GT)"],
        index=0,
        horizontal=True,
        key="llm_exp_choice",
    )
    exp_rows = exp1_rows if "Experiment 1" in exp_choice else exp2_rows
    round_options = list(range(1, len(exp_rows) + 1)) if exp_rows else [1]
    round_idx = st.selectbox("Round", options=round_options, index=0, key="llm_round")
    if exp_rows and round_idx:
        context_dirs = {"200": DIR_CONTEXT_200, "500": DIR_CONTEXT_500, "1000": DIR_CONTEXT_1000}
        discovery_entries = _load_entries(context_dirs.get(sig_context, Path()))
        round_pairs = _entries_for_round(discovery_entries, round_idx)

        st.markdown("**Text pairs used in this round (GT → ECCO)**")
        st.caption(f"Each round uses {PAIRS_PER_ROUND} pairs (3 reprint + 3 non-reprint). Showing {len(round_pairs)} pair(s) below.")
        if round_pairs:
            for i, entry in enumerate(round_pairs):
                role = entry.get("_role") or entry.get("reuse_type") or "—"
                src_url = _pair_display_url(entry, "src")
                dst_url = _pair_display_url(entry, "dst")
                st.markdown(f"**Pair {i + 1}** ({role})")
                if src_url or dst_url:
                    col_src, col_dst = st.columns(2)
                    with col_src:
                        st.caption(f"[Source (ECCO)]({src_url})" if src_url else "Source —")
                        if src_url:
                            components.iframe(src_url, height=280)
                    with col_dst:
                        st.caption(f"[Destination (ECCO)]({dst_url})" if dst_url else "Destination —")
                        if dst_url:
                            components.iframe(dst_url, height=280)
                else:
                    st.caption("No URL for this pair.")
        else:
            st.caption("No discovery-set entries for this round (check context window and round index).")

        st.markdown("---")
        st.markdown("**LLM output for this round**")
        row = exp_rows[round_idx - 1]
        resp = row.get("response") or {}
        st.markdown("**Summary**")
        st.write(resp.get("summary", ""))
        signals = resp.get("signals") or []
        for i, sig in enumerate(signals):
            with st.expander(f"Signal: {sig.get('name', '—')}"):
                st.write(sig.get("description", ""))
                st.caption(f"Stability: {sig.get('stability', '—')} · Relation: {sig.get('relation_to_naive_signals', '—')}")
                for ex in sig.get("supporting_examples") or []:
                    text = ex.get("quote_or_observation", "")
                    st.markdown(f"- **{ex.get('type', '')}**: {text[:400] + ('...' if len(text) > 400 else '')}")
        caveats = resp.get("caveats") or []
        if caveats:
            st.markdown("**Caveats**")
            for c in caveats:
                st.markdown(f"- {c}")

    compare_exp = st.checkbox("Compare same round across experiments", key="llm_compare_exp")
    if compare_exp and exp1_rows and exp2_rows and round_idx:
        c1, c2 = st.columns(2)
        i = min(round_idx - 1, len(exp1_rows) - 1)
        j = min(round_idx - 1, len(exp2_rows) - 1)
        r1, r2 = exp1_rows[i], exp2_rows[j]
        with c1:
            st.markdown("**Experiment 1**")
            _render_one_round(r1)
        with c2:
            st.markdown("**Experiment 2**")
            _render_one_round(r2)

    compare_ctx = st.checkbox("Compare same round across context windows (200 / 500 / 1000)", key="llm_compare_ctx")
    if compare_ctx and round_idx:
        cols = st.columns(3)
        for col, (ctx, path_key) in zip(cols, list(SIGNALS_PATHS.items())):
            with col:
                st.markdown(f"**Context {ctx}**")
                rows = _load_signals_jsonl(path_key) if path_key else []
                e1, e2 = _split_experiments(rows)
                exp_rows_ctx = e1 if "Experiment 1" in exp_choice else e2
                if round_idx <= len(exp_rows_ctx):
                    _render_one_round(exp_rows_ctx[round_idx - 1])
                else:
                    st.caption("No data for this round.")

