import json
import os
from pathlib import Path

import streamlit as st


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


def _render_compact_stats(derived_dir: Path, data_type: str) -> None:
    """Render compact preprocessing statistics."""
    # Filter statistics (most important)
    if data_type == "ecco":
        filter_stats_path = derived_dir / "hume_outgoing_ecco-ecco_original_only_stats.json"
    else:
        filter_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_original_only_stats.json"
    
    if filter_stats_path.exists():
        try:
            with open(filter_stats_path, "r") as f:
                filter_stats = json.load(f)
            total = filter_stats.get("total_records", 0)
            kept = filter_stats.get("kept_records", 0)
            filtered = filter_stats.get("filtered_records", 0)
            st.metric("Total → Kept", f"{total:,} → {kept:,}", delta=f"-{filtered:,} filtered")
        except Exception:
            pass
    
    # Merge statistics
    if data_type == "ecco":
        merged_stats_path = derived_dir / "hume_outgoing_ecco-ecco_original_only_merged_stats.json"
    else:
        merged_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_original_only_merged_stats.json"
    
    if merged_stats_path.exists():
        try:
            with open(merged_stats_path, "r") as f:
                merged_stats = json.load(f)
            input_records = merged_stats.get("input_records", 0)
            output_blocks = merged_stats.get("output_blocks", 0)
            reduction_pct = merged_stats.get("reduction_percentage", 0)
            st.metric("Records → Blocks", f"{input_records:,} → {output_blocks:,}", delta=f"-{reduction_pct:.1f}%")
        except Exception:
            pass


def render_title_slide() -> None:
    """Render the title slide with motivation and research questions."""
    st.markdown("""
    <style>
    .title-main {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        line-height: 1.3;
    }
    .title-sub {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .question-item {
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .question-number {
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<div class="title-main">Identifying essay-length reprinting across eighteenth-century books and newspapers:<br>A case study of David Hume</div>', unsafe_allow_html=True)
    
    # Authors
    st.markdown('<div class="title-sub">Speaker: <strong>Ke Shu</strong><br>Supervisor: <strong>Mikko Tolonen, Eetu Mäkelä</strong></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Motivation section
    st.markdown('<div class="section-header"># Motivation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    We aim to build an **operational system** that, given existing **text reuse** detections between ECCO and historical newspapers, **identifies probable essay-length reprinting** (full or partial) and points us to the **matching ECCO source**. 
    
    The immediate challenge is **fragmentation**—OCR noise and alignment errors split long reprints into multiple short reuse pieces, depressing recall. Using **David Hume** as a controlled pilot, we address this by:
    
    - **(i) offset-based defragmentation** to merge proximate pieces
    - **(ii) clustering** reuse pieces that co-concentrate in the **same Hume essay** and the **same newspaper issue**
    - **(iii) earliest-source tracing** to filter non-Hume origins
    
    The goal is a **high-yield candidate list** of likely essay-length (partial/fullish) reprints that can be **quickly verified** on ECCO page views and then extended to other authors/themes. Some **noise is acceptable** at early stages; crude length heuristics are fine if they reliably surface dense clusters indicative of longer reprints.
    """)
    
    st.divider()
    
    # Research Questions section
    st.markdown('<div class="section-header"># Research Questions </div>', unsafe_allow_html=True)
    
    questions = [
        {
            "number": "1",
            "title": "Recovering length from fragments",
            "content": "How can we algorithmically reconstruct essay-length reprints from fragmented text reuse hits—via start-offset windows, overlap rules, and cluster density thresholds within a source essay and a target newspaper issue?"
        },
        {
            "number": "2",
            "title": "Signals of \"essay-length reprinting\"",
            "content": "Which operational metrics (e.g., number of reuse hits per window, cumulative token/line coverage within an essay, dispersion across pages/columns, co-occurrence within a newspaper issue) best indicate a probable essay-length (partial/full) reprint without manual inspection?"
        },
        {
            "number": "3",
            "title": "Earliest-source filtering for Hume",
            "content": "How effective is earliest-source tracing at keeping Hume-origin cases and excluding non-Hume borrowings, while maintaining recall under fragmentation?"
        },
        {
            "number": "4",
            "title": "Practical definition of the \"essay\" unit",
            "content": "What length/context thresholds make the essay a reliable unit for detection and verification across books and newspapers, and how should these be calibrated with expert adjudication (accepting some noise early on)?"
        },
        {
            "number": "5",
            "title": "Throughput and scalability beyond Hume",
            "content": "To what extent does the system reduce the candidate space to the expected hundreds (at most) of original cases for Hume, and how readily can the same pipeline be ported to other authors/themes once validated?"
        }
    ]
    
    for q in questions:
        st.markdown(f"""
        <div class="question-item">
            <span class="question-number">Question {q['number']}: {q['title']}</span><br>
            {q['content']}
        </div>
        """, unsafe_allow_html=True)


def render_dataset_and_preprocessing() -> None:
    """Render Dataset Introduction and Data Preprocessing."""
    st.header("Dataset & Data Preprocessing", divider="blue")
    
    # Dataset Introduction
    st.markdown("""
    ## Dataset Introduction
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ECCO-Hume
        
        ECCO-Hume is a curated digital edition of David Hume's collected essays drawn from 
        Eighteenth Century Collections Online, providing book-side sources for matching 
        essay-length reprints.
        """)
    
    with col2:
        st.markdown("""
        ### Burney Newspapers Collection
        
        The Seventeenth and Eighteenth Century Burney Newspapers Collection (Burney) comprises 
        British newspapers from the 17th–18th centuries, supplying issue-level targets where 
        probable (partial/full) reprints surface.
        """)
    
    st.divider()
    
    # Initial Text Reuse Detection
    st.markdown("""
    ## Initial Text Reuse Detection
    
    We begin with **Hume as the source**, identifying all text reuse fragments related to his works 
    in both ECCO and newspapers. This detection is accomplished using an **optimized BLAST algorithm** 
    from prior research, which we do not detail here.
    
    The detection process yields two directions:
    
    - **Outgoing**: Others quoting/reprinting Hume's text (Hume → Others)
    - **Incoming**: Hume quoting/borrowing from others (Others → Hume)
    
    Since this study focuses on **the propagation of Hume's texts**, we concentrate on the 
    **outgoing direction**—tracking how Hume's original works were reprinted across ECCO books 
    and newspapers.
    """)
    
    st.divider()
    
    # Data Preprocessing
    st.markdown("""
    ## Data Preprocessing: Identify Original Hume Content
    
    ### Overview
    
    **Problem**: Some passages in Hume's works are borrowed from other authors. 
    We need to identify and filter out these "non-original" sections to ensure 
    our analysis focuses only on original Hume content.
    """)
    
    # Optimized compact flowchart
    flowchart_stage1 = """
    digraph Stage1 {
        rankdir=LR;
        node [shape=box, style=rounded, fontname="Arial", fontsize=10];
        edge [fontname="Arial", fontsize=9];
        
        RawData [label="Raw Reprint\nRecords", fillcolor="#e1f5ff", style="filled,rounded"];
        IdentifyBorrowed [label="Identify\nBorrowed\nPassages", fillcolor="#fff4e1", style="filled,rounded"];
        CheckOverlap [label="Check\nOverlap?", shape=diamond, fillcolor="#fff9e1", style="filled"];
        MarkNotHume [label="Mark as\nNon-Hume", fillcolor="#ffe1e1", style="filled,rounded"];
        KeepHume [label="Keep as\nHume Original", fillcolor="#e1ffe1", style="filled,rounded"];
        FilterOut [label="Filter\nOut", fillcolor="#ffe1e1", style="filled,rounded"];
        CleanedData [label="Cleaned\nData", fillcolor="#e1f5ff", style="filled,rounded"];
        
        RawData -> IdentifyBorrowed;
        IdentifyBorrowed -> CheckOverlap;
        CheckOverlap -> MarkNotHume [label="Yes"];
        CheckOverlap -> KeepHume [label="No"];
        MarkNotHume -> FilterOut;
        KeepHume -> CleanedData;
        FilterOut -> CleanedData [style=dashed, color=gray];
    }
    """
    st.graphviz_chart(flowchart_stage1)
    
    # Show compact statistics to demonstrate the effect of the process
    st.markdown("""
    ### Process Impact: Key Statistics
    """)
    
    # Compact display in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("**ECCO → ECCO**")
        _render_compact_stats(DERIVED_ECCO_DIR, "ecco")
    
    with col2:
        st.caption("**ECCO → Newspaper**")
        _render_compact_stats(DERIVED_NEWSPAPER_DIR, "newspaper")
    
    st.markdown("""
    <style>
    .step-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .step-detail {
        margin-left: 1.5rem;
        margin-bottom: 0.3rem;
        color: #555;
    }
    .step-key {
        font-weight: 600;
        color: #333;
    }
    </style>
    
    ### Process Steps
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-title">1. Raw Reprint Records</div>
    <div class="step-detail"><span class="step-key">Input:</span> All reprint records extracted from ECCO and newspaper databases</div>
    <div class="step-detail"><span class="step-key">Content:</span> Hume source location, reprint destination, text content</div>
    
    <div class="step-title">2. Identify Borrowed Passages</div>
    <div class="step-detail"><span class="step-key">Task:</span> Find sections in Hume documents that originate from other authors</div>
    <div class="step-detail"><span class="step-key">Method:</span> Text comparison to identify passages highly similar to earlier literature</div>
    <div class="step-detail"><span class="step-key">Output:</span> A list of "borrowed intervals" marking non-original content in Hume's works</div>
    
    <div class="step-title">3. Check Overlap</div>
    <div class="step-detail"><span class="step-key">Task:</span> Determine if each reprint record overlaps with borrowed passages</div>
    <div class="step-detail"><span class="step-key">Criteria:</span> If a reprint's location falls within a borrowed interval, it may not be original Hume content</div>
    
    <div class="step-title">4. Filter Processing</div>
    <div class="step-detail"><span class="step-key">Remove:</span> All reprint records overlapping with borrowed passages</div>
    <div class="step-detail"><span class="step-key">Keep:</span> Only reprints of original Hume content</div>
    
    <div class="step-title">5. Cleaned Data</div>
    <div class="step-detail"><span class="step-key">Output:</span> High-quality reprint record collection</div>
    <div class="step-detail"><span class="step-key">Characteristic:</span> Every record is guaranteed to be from original Hume content</div>
    """, unsafe_allow_html=True)


def render_stage2() -> None:
    """Render Stage 2: Build Analysis Units."""
    st.header("Stage 2: Build Analysis Units - Create Reprint Comparison Blocks", divider="green")
    
    st.markdown("""
    <style>
    .issue-card {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .issue-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 0.5rem;
    }
    .issue-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #dc3545;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .solution-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #007bff;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .summary-box {
        background-color: #e7f3ff;
        border: 2px solid #007bff;
        padding: 1.5rem;
        margin: 2rem 0;
        border-radius: 8px;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .metric-item {
        margin-left: 1.5rem;
        margin-top: 0.5rem;
        padding-left: 1rem;
        border-left: 3px solid #007bff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Issue 1
    st.markdown("""
    <div class="issue-card">
        <div class="issue-number">Issue 1</div>
        <div class="issue-title">Reprints of the same Hume essay are fragmented</div>
        <p>Making it difficult to compare them as a whole.</p>
        <div class="solution-title">How to solve:</div>
        <p>Merge adjacent/overlapping reprint fragments by <strong>"same source essay + start-offset windows"</strong> to generate continuous <strong>blocks</strong>, serving as basic units for subsequent analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issue 2
    st.markdown("""
    <div class="issue-card">
        <div class="issue-number">Issue 2</div>
        <div class="issue-title">The same source essay may be reprinted to multiple destinations</div>
        <p>Lacking a comparable framework.</p>
        <div class="solution-title">How to solve:</div>
        <p>Group by <strong>"source essay"</strong>, aggregate its <strong>destination blocks</strong> across different newspapers/books, and establish <strong>pair/cluster</strong> comparison relationships to compare how the same essay was reprinted to different destinations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issue 3
    st.markdown("""
    <div class="issue-card">
        <div class="issue-number">Issue 3</div>
        <div class="issue-title">Need quantifiable similarity and coverage metrics</div>
        <p>To compare reprint blocks across different destinations.</p>
        <div class="solution-title">How to solve:</div>
        <p>Calculate three core metrics:</p>
        <div class="metric-item"><strong>Overlap Ratio</strong>: Intersection character count ÷ Union character count of two destination blocks <em>(similarity)</em></div>
        <div class="metric-item"><strong>Reuse Ratio</strong>: Reprint block length ÷ Original source section length <em>(coverage)</em></div>
        <div class="metric-item"><strong>Intersection Length</strong>: Overlapping character count between two blocks <em>(absolute scale)</em></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Issue 4
    st.markdown("""
    <div class="issue-card">
        <div class="issue-number">Issue 4</div>
        <div class="issue-title">Lack traceable contextual metadata</div>
        <p>Making it difficult to filter and verify.</p>
        <div class="solution-title">How to solve:</div>
        <p>Enrich each block with metadata: <strong>section/headers, publication dates, page/column numbers, source and destination IDs, database URLs</strong>, facilitating traceability and manual verification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issue 5
    st.markdown("""
    <div class="issue-card">
        <div class="issue-number">Issue 5</div>
        <div class="issue-title">Analysis and visualization require a unified data structure</div>
        <div class="solution-title">How to solve:</div>
        <p>Output standardized <strong>"comparison block"</strong> data: each row = a pair (or cluster) of destination block comparisons, containing the above metrics and metadata; directly usable for visualization and statistical analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    st.markdown("""
    <div class="summary-box">
        <strong>Summary:</strong> Stage 2 integrates cleaned reprint records into comparable <strong>comparison blocks</strong>, enabling robust measurement of reprint overlap and coverage across multiple <strong>destinations</strong> for the <strong>same Hume essay</strong>, and providing standardized input for subsequent visualization and analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Stage 2 flowchart (horizontal layout)
    st.markdown("### Process Flow")
    flowchart_stage2 = """
    digraph Stage2 {
        rankdir=LR;
        node [shape=box, style=rounded, fontname="Arial", fontsize=9];
        edge [fontname="Arial", fontsize=8];
        
        CleanedData [label="Cleaned Data\nOnly original Hume\nreprint records", fillcolor="#e1f5ff", style="filled,rounded"];
        MergeFragments [label="Merge Fragments\nCombine multiple reprint fragments\nof the same Hume essay", fillcolor="#e1ffe1", style="filled,rounded"];
        FindOverlaps [label="Find Overlaps\nIdentify same essay reprinted\nto different destinations", fillcolor="#fff4e1", style="filled,rounded"];
        CalculateMetrics [label="Calculate Similarity Metrics\n• Overlap ratio: similarity between\n  two destination reprints\n• Reuse ratio: reprint length\n  divided by source section span\n• Intersection length: overlapping\n  text character count", fillcolor="#e1f5ff", style="filled,rounded"];
        AddMetadata [label="Add Metadata\n• Section headers\n• Publication dates\n• Database URLs\n• Image identifiers", fillcolor="#fff4e1", style="filled,rounded"];
        FinalBlocks [label="Final Analysis Units\nComplete comparison blocks\nwith all metrics and metadata", fillcolor="#e1f5ff", style="filled,rounded"];
        
        CleanedData -> MergeFragments;
        MergeFragments -> FindOverlaps;
        FindOverlaps -> CalculateMetrics;
        CalculateMetrics -> AddMetadata;
        AddMetadata -> FinalBlocks;
    }
    """
    st.graphviz_chart(flowchart_stage2)


def render_stage3() -> None:
    """Render Stage 3: Visualization & Analysis."""
    st.header("Stage 3: Visualization & Analysis", divider="violet")
    
    # Introduction
    st.info("""
    **What Stage 3 Does**
    
    Stage 3 is essentially a **demo tool** for people who want to use this data. 
    The goal is simple: help researchers quickly find interesting cases and see the details.
    
    Our research question is: **How do we identify long-segment text reuse?** 
    And more importantly: **What happens during transmission?** 
    Do people rewrite it? Do they cut parts out? Do they combine different sources?
    These are the questions we're trying to answer about how Hume's ideas spread.
    """)
    
    # Feature 1: Quick Location
    with st.expander("1. Quick Location: Find the Interesting Cases", expanded=True):
        st.markdown("""
        First, you need to **find where the long-segment reuses might be**. 
        We provide two ways to do this:
        """)
        
        st.markdown("#### Network View")
        st.markdown("""
        - You see a graph where each node is either a Hume essay or a newspaper/book
        - The lines between them show connections
        - **Thicker lines mean more text was reused**—these are the cases worth looking at
        """)
        st.info('**Example:** If you see Hume\'s "Of the Liberty of the Press" connected to a newspaper with a thick line, that\'s probably a long-segment reuse.')
        
        st.markdown("#### Timeline View")
        st.markdown("""
        - You can drag through time and see when reprints appeared
        - **Long-segment reuses often cluster together**—if you see multiple reprints in a short period, that's a signal
        """)
        st.info("**Example:** If something appears on June 12th, then June 15th, then June 22nd, that's likely a propagation chain.")
    
    # Feature 2: Detailed Comparison
    with st.expander("2. Detailed Comparison: See What Changed", expanded=True):
        st.markdown("""
        Once you find an interesting case, you need to **see exactly what happened**.
        
        We show the original Hume text on the left, and the reprinted version on the right, side by side. 
        You can see the actual images from ECCO and the newspapers.
        
        The key question is: **How was it changed?**
        """)
        
        st.markdown("""
        - **Was it rewritten?** We highlight where words were substituted or new text was inserted
        - **Was it cut?** We mark the sections that were deleted
        - **Was it patched together?** If a newspaper page combines text from multiple Hume essays, we show that
        """)
        
        st.info("**Example:** You might see that 1,200 characters match the original, but there were 3 substitutions and 2 big deletions. And on the same page, there's also 180 characters from another Hume essay—that's a patchwork reprint.")
    
    # Feature 3: Metrics
    with st.expander("3. Key Numbers: What to Look For", expanded=True):
        st.markdown("""
        We calculate several metrics to help you decide what's worth investigating:
        """)
        
        st.markdown("""
        - **Coverage (or Reuse Ratio)**: How much of the original was reused? If it's above 0.4, that's probably a long-segment reuse
        - **Intersection Length**: How many characters actually match? If it's over 1,000, that's worth a closer look
        - **Rewrite Intensity**: What percentage was changed? For example, 30% deletion, 12% substitution
        - **Patchwork Score**: How many different sources are combined? If it's 2 or more, that's a patchwork case
        """)
        
        st.info("**Example:** If you see Coverage=0.52, Intersection=1,240 characters, Deletion=28%, and Patchwork=2 sources, that's a high-priority case for review.")
    
    # Feature 4: Research Workflow
    with st.expander("4. From Finding to Evidence", expanded=True):
        st.markdown("""
        The final step is to **turn your findings into research evidence**.
        
        You can bookmark interesting cases, save screenshots, export the data with all the metrics and links. 
        Everything is traceable—you can go back and verify later.
        
        You can also filter by these metrics. For example, find all cases with Coverage≥0.4 and Patchwork≥1, 
        and you get a list of candidates for further study.
        """)
        
        st.info("**Example:** You might filter and find 27 cases that have high coverage and are patchworks. These become your evidence for studying how deletions and combinations affect Hume's arguments.")
    
    # How it answers research questions
    st.markdown("### How This Answers Our Research Questions")
    st.markdown("""
    - **Finding long-segment reuses**: The network and timeline help you locate them quickly, and the coverage and intersection metrics confirm they're long segments
    - **Identifying changes**: The side-by-side comparison shows you exactly what was rewritten, deleted, or patched together
    - **Studying propagation**: The timeline shows you the rhythm of how ideas spread, and the network shows you the paths they took
    """)
    
    # Summary
    st.success("""
    **In short:** Stage 3 connects "where are the long-segment reuses" with "what exactly changed". 
    You use the network and timeline to find interesting cases, then use the detailed comparison 
    to see the rewriting, deletion, and patchwork. Finally, you document everything as evidence 
    for studying how Hume's ideas changed as they spread.
    """)
    
    # Stage 3 flowchart (horizontal layout)
    st.markdown("### Process Flow")
    flowchart_stage3 = """
    digraph Stage3 {
        rankdir=LR;
        node [shape=box, style=rounded, fontname="Arial", fontsize=9];
        edge [fontname="Arial", fontsize=8];
        
        FinalBlocks [label="Comparison Blocks\n(from Stage 2)", fillcolor="#e1f5ff", style="filled,rounded"];
        QuickLocation [label="Quick Location\nNetwork & Timeline\nIdentify suspicious\nlong segments", fillcolor="#e1f5ff", style="filled,rounded"];
        DetailedCompare [label="Detailed Comparison\nSide-by-side view\nRewrite/Deletion/Patchwork\nidentification", fillcolor="#fff4e1", style="filled,rounded"];
        MetricsAnalysis [label="Metrics Analysis\nCoverage, Intersection\nRewrite Intensity\nPatchwork Score", fillcolor="#e1ffe1", style="filled,rounded"];
        EvidenceExport [label="Evidence Export\nBookmarks & Notes\nReproducible findings", fillcolor="#f0e8f8", style="filled,rounded"];
        
        FinalBlocks -> QuickLocation;
        QuickLocation -> DetailedCompare;
        DetailedCompare -> MetricsAnalysis;
        MetricsAnalysis -> EvidenceExport;
    }
    """
    st.graphviz_chart(flowchart_stage3)
    
    st.markdown("---")
    
    # Interactive Demo Links
    st.markdown("### Interactive Demo Applications")
    st.markdown("""
    Try out these interactive tools to explore the data:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Network & Propagation Analysis**
        - [Network & Propagation](https://reprints-network.streamlit.app/) - Explore reprint propagation networks, temporal trends by decade, and cross-dataset comparison
        - *Features*: Network view, target essay propagation analysis, timeline visualization
        """)
    
    with col2:
        st.markdown("""
        **Text Comparison & Analysis**
        - [Block Pairs Comparison](https://reprints-data.streamlit.app/) - View all reprint blocks by target essay with side-by-side text display
        - [Reuse Ratio Explorer](https://reprints2025-ratio.streamlit.app/) - Filter and analyze by reuse ratio, view statistics per essay
        - *Features*: ECCO page and newspaper image viewing, text comparison, search functionality
        """)


def render_findings() -> None:
    """Render Findings section."""
    st.header("Findings", divider="rainbow")
    
    st.markdown("""
    ### Key Findings
    
    This section presents the main findings from our analysis of essay-length reprinting 
    of David Hume's works across eighteenth-century books and newspapers.
    """)
    
    # Add your findings content here
    st.info("💡 **Note:** Add your findings and results here.")


def main():
    st.set_page_config(
        page_title="Identifying Essay-Length Reprinting: A Case Study of David Hume",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide the default Streamlit menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Create tabs for the presentation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Title & Research Questions",
        "📊 Dataset & Preprocessing",
        "🟢 Stage 2: Build Analysis Units",
        "🟣 Stage 3: Visualization & Analysis",
        "📈 Findings"
    ])
    
    with tab1:
        render_title_slide()
    
    with tab2:
        render_dataset_and_preprocessing()
    
    with tab3:
        render_stage2()
    
    with tab4:
        render_stage3()
    
    with tab5:
        render_findings()


if __name__ == "__main__":
    main()
