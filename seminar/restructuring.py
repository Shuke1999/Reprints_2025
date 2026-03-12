"""Dataset restructuring: split diagram image, Group A/B/Hard pair viewer with side-by-side URLs."""
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from seminar.data import (
    DISCOVERY_COUNTS,
    MAIN_EVAL_COUNTS,
    HARD_EVAL_COUNTS,
    DISCOVERY_BY_SRC,
    MAIN_EVAL_BY_SRC,
    HARD_EVAL_BY_SRC,
    GROUP_B_REPRINTS,
    GROUP_B_ENTRIES,
    HARD_NEGATIVES_PAIRS,
    HARD_NEGATIVES_ENTRIES,
    PATH_GT_ORIGIN,
    DIR_GT_SPLITS,
    REPO_ROOT,
)

ECCO_BASE = "https://onko-sivu.2.rahtiapp.fi/ecco"


def _build_url(doc_id: str, start: int, end: int) -> str:
    return f"{ECCO_BASE}?docId={doc_id}&offsetStart={start}&offsetEnd={end}&isAlreadyOctavified=0"


def _load_gt_offset_lookup():
    """Load gt_offset_origin and return dict (src_doc_id, dst_doc_id) -> {src_trs_start, src_trs_end, dst_trs_start, dst_trs_end}."""
    if not PATH_GT_ORIGIN.exists():
        return {}
    try:
        with open(PATH_GT_ORIGIN, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    lookup = {}
    for rec in data:
        key = (str(rec.get("src_doc_id")), str(rec.get("dst_doc_id")))
        if key in lookup:
            continue
        lookup[key] = {
            "src_trs_start": rec.get("src_trs_start"),
            "src_trs_end": rec.get("src_trs_end"),
            "dst_trs_start": rec.get("dst_trs_start"),
            "dst_trs_end": rec.get("dst_trs_end"),
        }
    return lookup


def _pair_urls(pair: dict, gt_lookup: dict) -> tuple[str | None, str | None]:
    """Return (src_url, dst_url) for a pair from pair_ids; use gt_lookup for offsets."""
    src_id = pair.get("src_doc_id")
    dst_id = pair.get("dst_doc_id")
    if not src_id or not dst_id:
        return None, None
    key = (str(src_id), str(dst_id))
    offsets = gt_lookup.get(key)
    if not offsets or offsets["src_trs_start"] is None or offsets["src_trs_end"] is None or offsets["dst_trs_start"] is None or offsets["dst_trs_end"] is None:
        return None, None
    src_url = _build_url(src_id, offsets["src_trs_start"], offsets["src_trs_end"])
    dst_url = _build_url(dst_id, offsets["dst_trs_start"], offsets["dst_trs_end"])
    return src_url, dst_url


def _render_pair_viewer(pairs: list, gt_lookup: dict, section_label: str, session_key_prefix: str):
    """One pair at a time: Prev/Next, then side-by-side src_url and dst_url (links + iframes)."""
    if not pairs:
        st.info(f"No pairs in **{section_label}**.")
        return
    key_idx = f"{session_key_prefix}_idx"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    idx = max(0, min(st.session_state[key_idx], len(pairs) - 1))
    st.session_state[key_idx] = idx
    pair = pairs[idx]
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ Previous", key=f"{session_key_prefix}_prev") and idx > 0:
            st.session_state[key_idx] = idx - 1
            st.rerun()
    with col_info:
        st.caption(f"**{section_label}** — Pair **{idx + 1}** of **{len(pairs)}** · src=`{pair.get('src_doc_id')}` dst=`{pair.get('dst_doc_id')}`")
    with col_next:
        if st.button("Next ▶", key=f"{session_key_prefix}_next") and idx < len(pairs) - 1:
            st.session_state[key_idx] = idx + 1
            st.rerun()
    src_url, dst_url = _pair_urls(pair, gt_lookup)
    if src_url and dst_url:
        st.markdown("**Links:** " + f"[Source (ECCO)]({src_url})" + " · " + f"[Destination (ECCO)]({dst_url})")
        col_src, col_dst = st.columns(2)
        with col_src:
            st.caption("Source")
            components.iframe(src_url, height=420)
        with col_dst:
            st.caption("Destination")
            components.iframe(dst_url, height=420)
    else:
        st.warning("This pair has no offset data in ground truth; cannot build ECCO links. src_doc_id / dst_doc_id only.")


def render():
    st.header("Dataset Restructuring", divider="rainbow")
    st.markdown(
        "Since naive signals were not enough for detection, we restructured the GT data to support "
        "**LLM-assisted signal discovery**. We used a **length-only baseline** to separate easier (Group A) "
        "and harder (Group B) reprint cases."
    )
    st.markdown("---")

    st.markdown("### Split diagram")
    gt_split_image = REPO_ROOT / "image" / "gt_split.png"
    if gt_split_image.exists():
        st.image(str(gt_split_image), use_container_width=True)
    else:
        st.caption(f"Image not found: `{gt_split_image}`")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Discovery set")
        for k, v in DISCOVERY_COUNTS.items():
            st.markdown(f"- {k}: **{v}**")
        with st.expander("By src_doc_id"):
            for src, count in sorted(DISCOVERY_BY_SRC.items()):
                st.text(f"{src}: {count}")
    with c2:
        st.markdown("#### Main evaluation set")
        for k, v in MAIN_EVAL_COUNTS.items():
            st.markdown(f"- {k}: **{v}**")
        with st.expander("By src_doc_id"):
            for src, count in sorted(MAIN_EVAL_BY_SRC.items()):
                st.text(f"{src}: {count}")
    with c3:
        st.markdown("#### Hard evaluation set")
        for k, v in HARD_EVAL_COUNTS.items():
            st.markdown(f"- {k}: **{v}**")
        with st.expander("By src_doc_id"):
            for src, count in sorted(HARD_EVAL_BY_SRC.items()):
                st.text(f"{src}: {count}")

    st.markdown("---")
    st.markdown("### Group B and hard negatives")
    st.markdown(f"- **Group B reprints**: {GROUP_B_REPRINTS} pairs ({GROUP_B_ENTRIES} entries) – reprints missed by the baseline.")
    st.markdown(f"- **Hard negatives** (baseline false positives): {HARD_NEGATIVES_PAIRS} pairs ({HARD_NEGATIVES_ENTRIES} entries).")

    st.markdown("---")
    st.markdown("### Example pairs: Group A, Group B, Hard negatives")
    gt_lookup = _load_gt_offset_lookup()
    group_a_path = DIR_GT_SPLITS / "group_a_reprints.json"
    group_b_path = DIR_GT_SPLITS / "group_b_reprints.json"
    hard_eval_path = DIR_GT_SPLITS / "hard_eval_set.json"

    def load_pair_ids(path: Path) -> list:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        return data.get("pair_ids", data) if isinstance(data, dict) else (data if isinstance(data, list) else [])

    group_a_pairs = load_pair_ids(group_a_path)
    group_b_pairs = load_pair_ids(group_b_path)
    hard_pairs = load_pair_ids(hard_eval_path)

    tab_a, tab_b, tab_hard = st.tabs(["Group A (reprints caught by baseline)", "Group B (reprints missed)", "Hard negatives (false positives)"])
    with tab_a:
        _render_pair_viewer(group_a_pairs, gt_lookup, "Group A", "restruct_group_a")
    with tab_b:
        _render_pair_viewer(group_b_pairs, gt_lookup, "Group B", "restruct_group_b")
    with tab_hard:
        _render_pair_viewer(hard_pairs, gt_lookup, "Hard negatives", "restruct_hard")
