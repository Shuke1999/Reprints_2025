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


def _render_block_stats(derived_dir: Path, dataset_label: str) -> None:
    """Render block-level statistics from all_reprint_pairs_stats.json."""
    stats_path = derived_dir / "all_reprint_pairs_stats.json"
    enriched_path = derived_dir / "all_reprint_pairs_enriched.json"

    if not stats_path.exists():
        st.info(f"No block statistics file found for {dataset_label}.")
        return

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as exc:
        st.warning(f"Unable to load block stats for {dataset_label}: {exc}")
        return

    total_pairs = stats.get("total_pairs", 0)
    overlap_stats = stats.get("overlap_ratio_stats", {})
    intersection_stats = stats.get("intersection_len_stats", {})

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total block pairs", total_pairs)
        if enriched_path.exists():
            size_mb = enriched_path.stat().st_size / (1024 * 1024)
            st.caption(f"`all_reprint_pairs_enriched.json` ({size_mb:.1f} MB)")
    with cols[1]:
        st.metric(
            "Overlap ratio range",
            f"{overlap_stats.get('min', 0):.2f} – {overlap_stats.get('max', 0):.2f}",
        )
        st.caption(f"mean ≈ {overlap_stats.get('mean', 0):.2f}")
    with cols[2]:
        st.metric(
            "Intersection length range",
            f"{intersection_stats.get('min', 0)} – {intersection_stats.get('max', 0)}",
        )
        st.caption(f"mean ≈ {intersection_stats.get('mean', 0):.0f}")
    with cols[3]:
        threshold = stats.get("min_block_length_used")
        st.metric("Min block length threshold", threshold or "N/A")
        note = stats.get("note")
        if note:
            st.caption(note)


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
    render_block_generation_section()


def render_block_generation_section() -> None:
    st.header("Step 2: Build enriched block pairs", divider="rainbow")
    st.markdown(
        """
We now turn the cleaned reuses into **enriched block pairs**—ready-made snippets that let us inspect
how one Hume essay spreads across destinations.

**What goes in**
- Filtered reuse fragments from `*_original_only_merged.json`
- Metadata bundles (`*_merged_with_urls.json`) with section headers, publication dates and Gale URLs
- The enrichment scripts in `derived-ecco/` and `derived-newspaper/`

**Pipeline at a glance**
1. **Find overlaps** between destination snippets of the same target essay → record `overlap_ratio`,
   `intersection_len`, and `min_block_length`.
2. **Rebuild URL offsets** so merged fragments open the exact text range in Gale (`src_trs_*`, `dst_trs_*`).
3. **Attach metadata** such as section headers, publication dates, and (for newspapers) `src_section_id`
   for image retrieval.
4. **Save to disk** as `all_reprint_pairs.json` + `all_reprint_pairs_enriched.json`, powering the Block
   Comparison and Network pages.
"""
    )

    stats_tab1, stats_tab2 = st.tabs(["ECCO-ECCO block stats", "ECCO-Newspaper block stats"])
    with stats_tab1:
        _render_block_stats(DERIVED_ECCO_DIR, "ECCO-ECCO")
    with stats_tab2:
        _render_block_stats(DERIVED_NEWSPAPER_DIR, "ECCO-Newspaper")

    with st.expander("Data dictionary (quick reference)", expanded=False):
        st.markdown(
            """
- `block_a` / `block_b`: source offsets (`src_trs_start/end`), destination offsets, lengths, fragment count, Gale URLs  
- `overlap_ratio`: how similar the two destination snippets are (0–1)  
- `min_block_length`: shorter snippet length, used for filtering noise  
- `src_section_id` (newspaper only): lookup key to fetch Nichols/Burney page images
"""
        )


def main():
    st.set_page_config(page_title="Reuses of Hume – Workflow", layout="wide")
    st.title("Reuses of Hume – Data Workflow")
    render_data_workflow_page()


if __name__ == "__main__":
    main()

