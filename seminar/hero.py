"""Hero / title page for the seminar."""
import streamlit as st


def render():
    st.title("Identifying essay-length reprinting across eighteenth-century books and newspapers")
    st.caption("A case study of David Hume")
    st.markdown("---")
    st.markdown(
        "**From naive signals to LLM-assisted signal discovery for reprint detection**"
    )
    st.markdown(
        "This project aims to detect **reprint pairs** from raw textual reuse pairs and develop "
        "a signal-based framework that can later scale from the David Hume pilot case to the full "
        "ECCO and newspaper corpora."
    )
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Reprint detection**")
    with col2:
        st.info("**ECCO**")
    with col3:
        st.info("**Newspapers**")
