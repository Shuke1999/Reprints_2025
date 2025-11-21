import json
import os
from pathlib import Path

import streamlit as st


def _resolve_repo_root() -> Path:
    default_root = Path(__file__).resolve().parents[1]
    override = os.environ.get("REPRINTS_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return default_root


def _resolve_data_dir(repo_root: Path) -> Path:
    override = os.environ.get("REPRINTS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root / "data").resolve()


REPO_ROOT = _resolve_repo_root()
DATA_DIR = _resolve_data_dir(REPO_ROOT)
DERIVED_ECCO_DIR = DATA_DIR / "data_2011" / "derived-ecco"
DERIVED_NEWSPAPER_DIR = DATA_DIR / "data_2011" / "derived-newspaper"


def render_statistics_section(derived_dir: Path, data_type: str) -> None:
    summary_path = derived_dir / "hume_borrowed_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            st.markdown("#### Borrowed interval statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Interval groups", summary.get("borrowed_interval_groups", 0))
            with col2:
                st.metric("Total borrowed intervals", summary.get("borrowed_intervals", 0))
            with col3:
                st.metric(
                    "Outgoing records overlapping",
                    summary.get("outgoing_records_overlapping_borrowed", 0),
                )
        except Exception as exc:
            st.warning(f"Unable to load borrowed interval stats: {exc}")

    if data_type == "ecco":
        filter_stats_path = derived_dir / "hume_outgoing_ecco-ecco_original_only_stats.json"
    else:
        filter_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_original_only_stats.json"

    if filter_stats_path.exists():
        try:
            with open(filter_stats_path, "r") as f:
                filter_stats = json.load(f)
            st.markdown("#### Filter statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total records", filter_stats.get("total_records", 0))
            with col2:
                st.metric("Kept records", filter_stats.get("kept_records", 0))
            with col3:
                st.metric("Filtered records", filter_stats.get("filtered_records", 0))
        except Exception as exc:
            st.warning(f"Unable to load filter statistics: {exc}")

    if data_type == "ecco":
        merged_stats_path = derived_dir / "hume_outgoing_ecco-ecco_original_only_merged_stats.json"
    else:
        merged_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_original_only_merged_stats.json"

    if merged_stats_path.exists():
        try:
            with open(merged_stats_path, "r") as f:
                merged_stats = json.load(f)
            st.markdown("#### Merge statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Input records", merged_stats.get("input_records", 0))
            with col2:
                st.metric("Output blocks", merged_stats.get("output_blocks", 0))
            with col3:
                st.metric("Reduction", merged_stats.get("reduction", 0))
            with col4:
                st.metric("Reduction %", f"{merged_stats.get('reduction_percentage', 0):.2f}%")
        except Exception as exc:
            st.warning(f"Unable to load merge statistics: {exc}")


def render_data_workflow_page() -> None:
    st.header("Step 1: Data preprocess - Clean non-Hume Data", divider="rainbow")
    tab1, tab2 = st.tabs(["ECCO-ECCO", "ECCO-Newspaper"])

    with tab1:
        render_statistics_section(DERIVED_ECCO_DIR, "ecco")

    with tab2:
        render_statistics_section(DERIVED_NEWSPAPER_DIR, "newspaper")

    st.markdown("#### Data processing flowchart")
    flowchart = """
        digraph {
            rankdir=TB;
            node [shape=box, style=rounded];
            
        A [label="Raw data
    All reuse records", fillcolor="#e1f5ff", style="filled,rounded"];
        B [label="Identify borrowed intervals
    borrowed_intervals.json", fillcolor="#fff4e1", style="filled,rounded"];
        C [label="Check overlap with
    borrowed intervals", shape=diamond, fillcolor="#fff9e1", style="filled"];
        D [label="Mark as 'not from Hume'
    outgoing_overlaps.json", fillcolor="#ffe1e1", style="filled,rounded"];
        E [label="Keep as 'from Hume'", fillcolor="#e1ffe1", style="filled,rounded"];
        F [label="Filter out", fillcolor="#ffe1e1", style="filled,rounded"];
        G [label="Final data", fillcolor="#e1f5ff", style="filled,rounded"];
        H [label="Filtered records", fillcolor="#ffe1e1", style="filled,rounded"];
            
            A -> B;
            B -> C;
        C -> D [label="Yes"];
        C -> E [label="No"];
            D -> F;
            E -> G;
            F -> H;
        }
        """
    st.graphviz_chart(flowchart)

    st.markdown(
        """
**Process overview:**
1. **Raw data**: all reuse records extracted from the corpus
2. **Identify borrowed intervals**: detect sections in Hume documents that originate elsewhere
3. **Overlap check**: determine whether each reuse overlaps a borrowed interval
4. **Filter**: remove records overlapping borrowed intervals (not truly originating from Hume)
5. **Result**: keep only reuses that genuinely start from Hume

**Reference files:**
- `hume_borrowed_intervals.json`: borrowed interval details
- `hume_borrowed_summary.json`: summary statistics
- `hume_outgoing_overlaps.json`: overlaps with borrowed intervals
"""
    )


def main():
    st.set_page_config(page_title="Reuses of Hume – Workflow", layout="wide")
    st.title("Reuses of Hume – Data Workflow")
    render_data_workflow_page()


if __name__ == "__main__":
    main()

