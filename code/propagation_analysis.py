"""Propagation analysis: Compare reuse patterns between ECCO and Newspaper.

This module provides analysis and visualization of how the same target essay
is reused differently in ECCO vs Newspaper, with annotations for reuse patterns.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from annotation import annotate_reuse, AnnotationType


def _resolve_repo_root() -> Path:
    """Resolve the project root (supports env override for Streamlit Cloud)."""
    default_root = Path(__file__).resolve().parents[1]
    override = os.environ.get("REPRINTS_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return default_root


def _resolve_data_dir(repo_root: Path) -> Path:
    """Resolve the data directory with optional environment override."""
    override = os.environ.get("REPRINTS_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root / "data").resolve()


REPO_ROOT = _resolve_repo_root()
DATA_DIR = _resolve_data_dir(REPO_ROOT)
DERIVED_ECCO_DIR = DATA_DIR / "data_2011" / "derived-ecco"
DERIVED_NEWSPAPER_DIR = DATA_DIR / "data_2011" / "derived-newspaper"

BLOCKS_DATA_FILES = {
    "ecco": DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json",
    "newspaper": DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json",
}

DST_METADATA_FILES = {
    "ecco": DERIVED_ECCO_DIR / "hume_outgoing_ecco-ecco_original_only_merged_with_urls.json",
    "newspaper": DERIVED_NEWSPAPER_DIR / "hume_outgoing_ecco-newspaper_original_only_merged_with_urls.json",
}


def _parse_year(date_str: str | None) -> int | None:
    """Parse year from date string."""
    if not date_str:
        return None
    try:
        year = int(date_str.split("-", 1)[0])
    except (ValueError, AttributeError):
        return None
    return year if 1400 <= year <= 1900 else None


def _calculate_essay_ratio(pair: dict[str, Any], metadata_cache: dict[str, dict]) -> float | None:
    """Calculate essay_ratio for a pair.
    
    essay_ratio = src_piece_length / (src_section_end - src_section_start)
    """
    src_doc_id = pair.get("src_doc_id")
    src_section_id = pair.get("src_section_id")
    data_type = pair.get("data_type", "ecco")
    
    if src_doc_id is None or src_section_id is None:
        return None
    
    # Get section boundaries from metadata
    metadata = metadata_cache.get(data_type, {})
    section_starts = metadata.get("src_section_starts", {})
    section_ends = metadata.get("src_section_ends", {})
    
    key = (str(src_doc_id), str(src_section_id))
    section_start = section_starts.get(key)
    section_end = section_ends.get(key)
    
    if section_start is None or section_end is None:
        return None
    
    section_span = section_end - section_start
    if section_span == 0:
        return None
    
    # Get src_piece_length from block_a or block_b
    block_a = pair.get("block_a", {})
    block_b = pair.get("block_b", {})
    src_piece_length = block_a.get("src_piece_length") or block_b.get("src_piece_length")
    
    if src_piece_length is None:
        return None
    
    return src_piece_length / section_span


def _load_metadata_cache() -> dict[str, dict]:
    """Load metadata cache with section boundaries."""
    cache: dict[str, dict] = {}
    
    for data_type, metadata_path in DST_METADATA_FILES.items():
        if not metadata_path.exists():
            continue
        
        section_starts: dict[tuple[str, str], int] = {}
        section_ends: dict[tuple[str, str], int] = {}
        dst_pub_dates: dict[str, str] = {}
        dst_pub_years: dict[str, int] = {}
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            
            for record in records:
                src_doc_id = record.get("src_doc_id")
                src_section_id = record.get("src_section_id")
                if src_doc_id is not None and src_section_id is not None:
                    key = (str(src_doc_id), str(src_section_id))
                    if key not in section_starts:
                        section_starts[key] = record.get("src_section_start")
                        section_ends[key] = record.get("src_section_end")
                
                dst_doc_id = record.get("dst_doc_id")
                dst_publication_date = record.get("dst_publication_date")
                if dst_doc_id and dst_publication_date:
                    doc_id_str = str(dst_doc_id)
                    dst_pub_dates[doc_id_str] = dst_publication_date
                    year = _parse_year(dst_publication_date)
                    if year is not None:
                        dst_pub_years[doc_id_str] = year
        
        except Exception as exc:
            st.warning(f"Unable to load metadata for {data_type}: {exc}")
        
        cache[data_type] = {
            "src_section_starts": section_starts,
            "src_section_ends": section_ends,
            "dst_pub_dates": dst_pub_dates,
            "dst_pub_years": dst_pub_years,
        }
    
    return cache


def _load_blocks_data(data_type: str) -> list[dict[str, Any]]:
    """Load blocks data for a given data type."""
    data_path = BLOCKS_DATA_FILES.get(data_type)
    if not data_path or not data_path.exists():
        return []
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Add data_type to each pair
        for pair in data:
            pair["data_type"] = data_type
        return data
    except Exception as exc:
        st.warning(f"Unable to load blocks data for {data_type}: {exc}")
        return []


def _process_reuse_data(
    blocks_data: list[dict[str, Any]],
    metadata_cache: dict[str, dict],
    min_essay_ratio: float = 0.0,
    max_essay_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    """Process reuse data and add annotations.
    
    Returns a list of processed reuse records with annotations.
    """
    processed: list[dict[str, Any]] = []
    
    for pair in blocks_data:
        # Calculate essay_ratio
        essay_ratio = _calculate_essay_ratio(pair, metadata_cache)
        if essay_ratio is None:
            continue
        
        # Filter by essay_ratio
        if not (min_essay_ratio <= essay_ratio <= max_essay_ratio):
            continue
        
        src_doc_id = pair.get("src_doc_id")
        src_section_id = pair.get("src_section_id")
        data_type = pair.get("data_type", "ecco")
        
        # Get metadata
        metadata = metadata_cache.get(data_type, {})
        dst_pub_dates = metadata.get("dst_pub_dates", {})
        dst_pub_years = metadata.get("dst_pub_years", {})
        
        # Process block_a and block_b
        for block_label, block in [("Block A", pair.get("block_a", {})), ("Block B", pair.get("block_b", {}))]:
            if not block:
                continue
            
            dst_doc_id = block.get("dst_doc_id")
            if not dst_doc_id:
                continue
            
            # Get publication info
            dst_pub_date = dst_pub_dates.get(str(dst_doc_id))
            dst_year = dst_pub_years.get(str(dst_doc_id))
            
            # Annotate reuse
            annotation = annotate_reuse(
                src_piece_length=block.get("src_piece_length"),
                dst_piece_length=block.get("dst_piece_length"),
                fragment_count=block.get("fragment_count"),
                intersection_len=pair.get("intersection_len"),
                min_block_length=pair.get("min_block_length"),
            )
            
            processed.append({
                "target_essay_doc_id": src_doc_id,
                "target_essay_section_id": src_section_id,
                "essay_ratio": essay_ratio,
                "dataset": "ECCO" if data_type == "ecco" else "Newspaper",
                "dst_doc_id": dst_doc_id,
                "dst_publication_date": dst_pub_date,
                "dst_year": dst_year,
                "src_piece_length": block.get("src_piece_length"),
                "dst_piece_length": block.get("dst_piece_length"),
                "fragment_count": block.get("fragment_count"),
                "annotation": annotation,
                "block_label": block_label,
            })
    
    return processed


def _create_visualizations(df: pd.DataFrame) -> None:
    """Create visualizations for propagation analysis."""
    if df.empty:
        st.info("No data available for visualization.")
        return
    
    # 1. Timeline visualization
    st.subheader("Timeline: Propagation Over Time")
    
    timeline_df = df[df["dst_year"].notna()].copy()
    if not timeline_df.empty:
        timeline_summary = (
            timeline_df.groupby(["dst_year", "dataset", "annotation"])
            .size()
            .reset_index(name="count")
        )
        
        fig_timeline = px.bar(
            timeline_summary,
            x="dst_year",
            y="count",
            color="annotation",
            facet_col="dataset",
            labels={
                "dst_year": "Year",
                "count": "Number of Reuses",
                "annotation": "Annotation Type",
            },
            title="Reuse Propagation Timeline by Dataset and Annotation Type",
            barmode="stack",
        )
        fig_timeline.update_layout(height=500)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No year information available for timeline visualization.")
    
    # 2. Annotation distribution comparison
    st.subheader("Annotation Distribution: ECCO vs Newspaper")
    
    annotation_summary = (
        df.groupby(["dataset", "annotation"])
        .size()
        .reset_index(name="count")
    )
    
    fig_dist = px.bar(
        annotation_summary,
        x="annotation",
        y="count",
        color="dataset",
        labels={
            "annotation": "Annotation Type",
            "count": "Number of Reuses",
            "dataset": "Dataset",
        },
        title="Annotation Type Distribution by Dataset",
        barmode="group",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # 3. Summary statistics
    st.subheader("Summary Statistics")
    
    summary_stats = (
        df.groupby(["dataset", "annotation"])
        .agg(
            count=("annotation", "size"),
            avg_src_length=("src_piece_length", "mean"),
            avg_dst_length=("dst_piece_length", "mean"),
        )
        .reset_index()
    )
    summary_stats["avg_src_length"] = summary_stats["avg_src_length"].round(0)
    summary_stats["avg_dst_length"] = summary_stats["avg_dst_length"].round(0)
    
    st.dataframe(summary_stats, use_container_width=True, hide_index=True)


# Main application
st.set_page_config(page_title="Propagation Analysis", layout="wide")
st.title("Reuse Propagation Analysis: ECCO vs Newspaper")

st.markdown(
    """
    This analysis compares how the same target essay is reused differently 
    in ECCO vs Newspaper, with annotations for reuse patterns:
    - **Deletion**: Text is shortened in reuse
    - **Concatenation**: Multiple source blocks are combined
    - **Paraphrase**: Text is rephrased (placeholder)
    - **Other**: Default category
    """
)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    essay_ratio_range = st.slider(
        "essay_ratio range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        help="Filter by essay_ratio (src_piece_length / section_span)",
    )
    
    dataset_filter = st.multiselect(
        "Datasets",
        options=["ECCO", "Newspaper"],
        default=["ECCO", "Newspaper"],
    )
    
    annotation_filter = st.multiselect(
        "Annotation Types",
        options=["deletion", "Concatenation", "paraphrase", "other"],
        default=["deletion", "Concatenation", "paraphrase", "other"],
    )

# Load and process data
with st.spinner("Loading data..."):
    metadata_cache = _load_metadata_cache()
    
    all_blocks_data: list[dict[str, Any]] = []
    for data_type in ["ecco", "newspaper"]:
        blocks_data = _load_blocks_data(data_type)
        all_blocks_data.extend(blocks_data)
    
    processed_data = _process_reuse_data(
        all_blocks_data,
        metadata_cache,
        min_essay_ratio=essay_ratio_range[0],
        max_essay_ratio=essay_ratio_range[1],
    )

# Convert to DataFrame
df = pd.DataFrame(processed_data)

# Apply filters
if not df.empty:
    df = df[df["dataset"].isin(dataset_filter)]
    df = df[df["annotation"].isin(annotation_filter)]

# Display results
if df.empty:
    st.warning("No data matches the current filters.")
else:
    st.success(f"Found {len(df)} reuse records matching the filters.")
    
    # Create visualizations
    _create_visualizations(df)
    
    # Raw data view (optional)
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
