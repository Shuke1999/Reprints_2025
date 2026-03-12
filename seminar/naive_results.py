"""Naive signal results: feature table, threshold tabs, topk by feature viewer, summary, takeaway."""
import json
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from seminar.data import (
    FEATURE_COMPARISON_ROWS,
    THRESHOLD_SCANS,
    PATH_TOPK_BY_FEATURE,
)


def render():
    st.header("Naive Signal Results", divider="rainbow")
    st.caption("Positive = reprint | Negative = non-reprint")

    st.markdown("### Feature comparison: positive vs negative")
    df = pd.DataFrame(FEATURE_COMPARISON_ROWS)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Threshold scanning")
    st.markdown(
        "**TPR** = True Positive Rate (hit rate on reprints) | **FPR** = False Positive Rate. "
        "**Separability Proxy** = TPR / (TPR + FPR). For shift_std, smaller threshold is better (is_reverse=True)."
    )
    feat_names = list(THRESHOLD_SCANS.keys())
    tabs = st.tabs(feat_names[:6])
    for idx, feat_name in enumerate(feat_names[:6]):
        data = THRESHOLD_SCANS[feat_name]
        with tabs[idx]:
            st.caption(data["direction"])
            st.dataframe(
                pd.DataFrame(
                    data["rows"],
                    columns=["Threshold", "TPR", "FPR", "Separability Proxy"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    if len(THRESHOLD_SCANS) > 6:
        with st.expander("More features (threshold scan)"):
            for feat_name in feat_names[6:]:
                data = THRESHOLD_SCANS[feat_name]
                st.markdown(f"**{feat_name}** – {data['direction']}")
                st.dataframe(
                    pd.DataFrame(
                        data["rows"],
                        columns=["Threshold", "TPR", "FPR", "Separability Proxy"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    col_main, col_side = st.columns([2, 1])
    with col_side:
        st.markdown("#### Summary")
        st.success("**Length** is the only signal with clear discriminative power.")
        st.warning("**Density** is weak; only marginal benefit at high thresholds.")
        st.info("**Fragmentation, span ratio, coverage, alignment**: ineffective for separation.")
    with col_main:
        st.markdown("#### Key finding")
        st.markdown("- `reuse_length_dst` median: **1695** (reprint) vs **219** (non-reprint)")
        st.markdown("- Length thresholds: e.g. dst ≥ 1679 → TPR 0.506, FPR 0.024; dst ≥ 3519 → TPR 0.21, FPR 0")

    st.markdown("---")
    st.markdown("### Takeaway")
    st.markdown("> Naive signals do not solve reprint detection. Length is useful for **candidate reduction**, but not sufficient for reliable classification.")

    st.markdown("---")
    st.markdown("### Top-k by feature: example pairs")
    if not PATH_TOPK_BY_FEATURE.exists():
        st.warning(f"`{PATH_TOPK_BY_FEATURE}` not found.")
    else:
        try:
            with open(PATH_TOPK_BY_FEATURE, "r", encoding="utf-8") as f:
                topk_data = json.load(f)
        except Exception as e:
            st.error(f"Failed to load topk_by_feature: {e}")
            topk_data = {}
        if topk_data:
            feature_names = list(topk_data.keys())
            feat = st.selectbox("Feature", options=feature_names, key="topk_feature")
            list_type = st.radio("List", options=["reprint_top_20", "non_reprint_top_20"], horizontal=True, key="topk_list_type")
            by_feat = topk_data.get(feat) or {}
            items = by_feat.get(list_type) or []
            if not items:
                st.info(f"No entries in **{feat}** → **{list_type}**.")
            else:
                session_key = f"topk_idx_{feat}_{list_type}"
                if session_key not in st.session_state:
                    st.session_state[session_key] = 0
                idx = max(0, min(st.session_state[session_key], len(items) - 1))
                st.session_state[session_key] = idx
                pair = items[idx]
                col_prev, col_info, col_next = st.columns([1, 2, 1])
                with col_prev:
                    prev_clicked = st.button("◀ Previous pair", key=f"topk_prev_{feat}_{list_type}")
                    if prev_clicked and idx > 0:
                        st.session_state[session_key] = idx - 1
                        st.rerun()
                with col_info:
                    st.caption(f"Pair **{idx + 1}** of **{len(items)}** · feature_value = {pair.get('feature_value')} · {pair.get('reuse_type', '')}")
                with col_next:
                    next_clicked = st.button("Next pair ▶", key=f"topk_next_{feat}_{list_type}")
                    if next_clicked and idx < len(items) - 1:
                        st.session_state[session_key] = idx + 1
                        st.rerun()
                src_url = pair.get("src_url")
                dst_url = pair.get("dst_url")
                if not src_url and pair.get("src_doc_id") and pair.get("src_start_offset") is not None and pair.get("src_end_offset") is not None:
                    src_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={pair['src_doc_id']}&offsetStart={pair['src_start_offset']}&offsetEnd={pair['src_end_offset']}&isAlreadyOctavified=0"
                if not dst_url and pair.get("dst_doc_id") and pair.get("dst_start_offset") is not None and pair.get("dst_end_offset") is not None:
                    dst_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={pair['dst_doc_id']}&offsetStart={pair['dst_start_offset']}&offsetEnd={pair['dst_end_offset']}&isAlreadyOctavified=0"
                if src_url or dst_url:
                    st.markdown("**Links:** " + (f"[Source (ECCO)]({src_url})" if src_url else "—") + " · " + (f"[Destination (ECCO)]({dst_url})" if dst_url else "—"))
                    if src_url and dst_url:
                        col_src, col_dst = st.columns(2)
                        with col_src:
                            st.caption("Source")
                            components.iframe(src_url, height=480)
                        with col_dst:
                            st.caption("Destination")
                            components.iframe(dst_url, height=480)
                else:
                    st.warning("No src_url/dst_url or offset fields in this entry.")
        else:
            st.info("topk_by_feature.json is empty or has no features.")
