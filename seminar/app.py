"""
Seminar presentation app: single entry point with sidebar navigation.
All page content lives in separate modules under seminar/.
"""
import sys
from pathlib import Path

# Ensure repo root is on path when running as streamlit run seminar/app.py
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

from seminar import hero, overview, naive_signals, naive_results, restructuring
from seminar import llm_discovery, next_steps, data_search


PAGES = [
    ("Hero", hero.render),
    ("Project Overview", overview.render),
    ("Naive Signal Analysis", naive_signals.render),
    ("Naive Signal Results", naive_results.render),
    ("Dataset Restructuring", restructuring.render),
    ("LLM-based Signal Discovery", llm_discovery.render),
    ("Next-stage Proposal", next_steps.render),
    ("Data & Search", data_search.render),
]


def main():
    st.set_page_config(
        page_title="Reprint Detection – Seminar",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page_names = [p[0] for p in PAGES]
    page_fns = {p[0]: p[1] for p in PAGES}

    with st.sidebar:
        st.markdown("## Reprint Detection Seminar")
        st.caption("From naive signals to LLM-assisted discovery")
        selected = st.radio("Page", page_names, label_visibility="collapsed")

    if selected and selected in page_fns:
        page_fns[selected]()


if __name__ == "__main__":
    main()
