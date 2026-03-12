"""Project overview page: research goal, data, GT stats, workflow placeholder."""
import streamlit as st
import streamlit.components.v1 as components

from seminar.data import (
    DATA_DIR,
    GT_POSITIVE_PAIRS,
    GT_NEGATIVE_PAIRS,
    GT_ATTRIBUTED_REPRINT,
    GT_UNATTRIBUTED_REPRINT,
    GT_QUOTED,
    GT_CRIBBED,
    RAW_INTERVALS,
    RAW_UNIQUE_PAIRS,
    HUME_ECCO_IDS,
)


def render():
    st.header("Project Overview", divider="rainbow")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Research goal")
        st.markdown(
            "We aim to identify **reprint events** from raw text reuse data and eventually apply "
            "the method to **ECCO books** and **newspapers**. The current work starts from the "
            "**Hume case** as a pilot dataset and develops the workflow in stages."
        )
        st.markdown("### Why Hume?")
        st.markdown(
            "Because annotated data are limited, we work with a subset that contains all currently "
            "annotated material: **17 ECCO IDs**, and only **Hume → others** reuse."
        )
        st.markdown("### Data used")
        st.markdown("**ECCO (Eighteenth Century Collections Online)** — A digital archive of books published in Great Britain and its territories between 1701 and 1800. The collection comprises over 180,000 titles and more than 32 million pages, covering English and other European languages. It is maintained by Gale and supports full-text search.")
        st.markdown("**Burney Collection** — A digitized set of 17th- and 18th-century newspapers and news materials originally gathered by the scholar Charles Burney (1757–1817), now held by the British Library. It includes over 1,270 newspaper titles (nearly 1 million pages), from London dailies and provincial papers to Irish, Scottish, and colonial imprints, digitized by Gale in partnership with the British Library.")
        st.markdown("We do **not** include Nichols (published before Hume's birth).")
        st.caption(f"Hume subset: **{HUME_ECCO_IDS} ECCO IDs**, Hume → others only.")

    with col_right:
        st.markdown("### Dataset statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Positive (reprint) pairs", GT_POSITIVE_PAIRS)
            st.metric("Negative (non-reprint) pairs", GT_NEGATIVE_PAIRS)
            st.caption("Attributed Reprint: 130 | Unattributed: 46")
            st.caption("Quoted: 746 | Cribbed: 212 | Other: 3")
        with c2:
            st.metric("Raw intervals (ECCO–ECCO)", f"{RAW_INTERVALS:,}")
            st.metric("Unique pairs", f"{RAW_UNIQUE_PAIRS:,}")
        st.caption("Raw reuse and ground truth are aligned in the same offset coordinate system.")

    st.markdown("---")
    st.markdown("### Workflow")
    miro_url = "https://miro.com/app/board/uXjVG2dqZus=/"
    st.markdown(f"**Workflow board:** [Open in Miro (new tab)]({miro_url})")
    # Embed Miro board; if Miro blocks iframe (X-Frame-Options), the frame may be blank — use the link above
    components.iframe(miro_url, height=500)
