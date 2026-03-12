"""Naive signal definitions: cards + workflow image + example placeholders."""
import streamlit as st

from seminar.data import REPO_ROOT


def _signal_card(title: str, definition: str, formula: str = ""):
    with st.expander(title, expanded=False):
        st.markdown(definition)
        if formula:
            st.code(formula, language="text")


def render():
    st.header("Naive Signal Analysis", divider="rainbow")
    st.markdown(
        "Before building a more complex detection framework, we first tested whether simple, "
        "directly measurable signals can separate reprints from non-reprints."
    )
    st.markdown("---")

    st.markdown("### Naive signals definition")
    _signal_card(
        "**Length-related** – reuse_length_dst",
        "Length of the reuse span on the destination side (ground truth), in characters.",
        "dst_trs_end − dst_trs_start",
    )
    _signal_card(
        "**Length-related** – reuse_length_src",
        "Length of the reuse span on the source side.",
        "src_trs_end − src_trs_start",
    )
    _signal_card(
        "**Fragmentation** – num_fragments",
        "Number of raw reuse fragments detected for a given (src_doc_id, dst_doc_id) pair.",
    )
    _signal_card(
        "**Span expansion** – span_ratio_src / span_ratio_dst",
        "Ratio between the total span covered by raw fragments and the ground-truth span (source or destination).",
    )
    _signal_card(
        "**Coverage** – overlap_ratio_src / overlap_ratio_dst",
        "Proportion of the ground-truth span that is covered by raw fragments.",
    )
    _signal_card(
        "**Propagation density** – pair_reuse_density",
        "Number of distinct destination texts that reuse the same source segment.",
    )
    _signal_card(
        "**Alignment stability** – fragmentation_gap_rate, shift_stability, shift_std, shift_iqr",
        "Gaps between fragments, consistency of alignment shift across fragments; lower shift_std / shift_iqr = more stable alignment.",
    )

    st.markdown("---")
    st.markdown("### Workflow")
    workflow_image = REPO_ROOT / "image" / "naive_signals.png"
    if workflow_image.exists():
        st.image(str(workflow_image), use_container_width=True)
    else:
        st.caption(f"Workflow image not found: `{workflow_image}`")
