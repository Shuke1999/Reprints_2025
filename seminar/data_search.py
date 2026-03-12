"""Data & Search: GT preview, filters, derived data entry."""
import json
import streamlit as st
from pathlib import Path

from seminar.data import (
    PATH_GT_ORIGIN,
    PATH_HUME_GT_LIST,
    PATH_TOPK_BY_FEATURE,
    DIR_GT_SPLITS,
    DIR_CONTEXT_200,
    PATH_SIGNALS_200,
    PATH_SIGNALS_500,
    PATH_SIGNALS_1000,
    DATA_DIR,
)


def _safe_load_json(path: Path, default=None):
    if default is None:
        default = []
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def render():
    st.header("Data & Search", divider="rainbow")
    st.markdown("Full experimental data and ground truth for self-service lookup and verification.")

    st.markdown("### Ground truth")
    if PATH_GT_ORIGIN.exists():
        st.caption(f"`{PATH_GT_ORIGIN.relative_to(DATA_DIR)}`")
        data, err = _safe_load_json(PATH_GT_ORIGIN, default={})
        if err:
            st.warning(err)
        elif isinstance(data, list):
            st.metric("Records", len(data))
            with st.expander("Preview (first 5)"):
                st.json(data[:5] if len(data) > 5 else data)
        elif isinstance(data, dict):
            st.json(data)
    else:
        st.info(f"GT file not found at {PATH_GT_ORIGIN}")

    st.markdown("### Hume ECCO ID list")
    if PATH_HUME_GT_LIST.exists():
        with open(PATH_HUME_GT_LIST, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        st.caption(f"`{PATH_HUME_GT_LIST.relative_to(DATA_DIR)}` – {len(lines)} IDs")
        st.text(", ".join(lines[:20]) + (" ..." if len(lines) > 20 else ""))
    else:
        st.info(f"File not found: {PATH_HUME_GT_LIST}")

    st.markdown("### Top-k by feature (example pairs)")
    if PATH_TOPK_BY_FEATURE.exists():
        data, err = _safe_load_json(PATH_TOPK_BY_FEATURE, default={})
        if err:
            st.warning(err)
        elif isinstance(data, dict):
            st.caption(f"Features: {list(data.keys())[:8]} ...")
            feat = st.selectbox("Feature", options=list(data.keys()), key="topk_feat")
            if feat and isinstance(data[feat], dict):
                for key in ["reprint_top_20", "non_reprint_top_20"]:
                    if key in data[feat]:
                        with st.expander(f"{feat} – {key}"):
                            st.json(data[feat][key][:3])
    else:
        st.info(f"File not found: {PATH_TOPK_BY_FEATURE}")

    st.markdown("### Derived / split data")
    st.caption("GT splits and context sets live under `data/data0309/`. Use file browser or downstream tools for full search.")
    for label, d in [
        ("GT splits", DIR_GT_SPLITS),
        ("Context 200", DIR_CONTEXT_200),
    ]:
        st.text(f"{label}: {d} (exists: {d.exists()})")

    st.markdown("### LLM discovery outputs")
    for label, p in [
        ("signals_discovery_200", PATH_SIGNALS_200),
        ("signals_discovery_500", PATH_SIGNALS_500),
        ("signals_discovery_1000", PATH_SIGNALS_1000),
    ]:
        st.caption(f"{label}: {p} (exists: {p.exists()})")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            st.text(f"Lines (rounds): {len(lines)}")
