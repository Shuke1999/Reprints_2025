import json
import os
import re
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path


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

DST_METADATA_FILES = {
    "ecco": DERIVED_ECCO_DIR / "hume_outgoing_ecco-ecco_original_only_merged_with_urls.json",
    "newspaper": DERIVED_NEWSPAPER_DIR / "hume_outgoing_ecco-newspaper_original_only_merged_with_urls.json",
}

_METADATA_CACHE: dict[str, dict[str, dict]] = {}

st.title("Reuses of Hume")

st.header("Step 1: Data preprocess - Clean non-Hume Data", divider="rainbow")

tab1, tab2 = st.tabs(["ECCO-ECCO", "ECCO-Newspaper"])

def render_statistics_section(derived_dir, data_type):
    """Render statistics section with unified data loading."""
    # Try to load borrowed summary stats
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
                    "outgoing records overlapping",
                    summary.get("outgoing_records_overlapping_borrowed", 0),
                )
        except Exception as e:
            st.warning(f"Unable to load borrowed interval stats: {e}")
    
    # Load filter statistics
    if data_type == "ecco":
        filter_stats_path = derived_dir / "hume_outgoing_ecco-ecco_original_only_stats.json"
    else:  # newspaper
        filter_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_from_hume_stats.json"
    
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
        except Exception as e:
            st.warning(f"Unable to load filter statistics: {e}")
    
    # Load merge statistics
    if data_type == "ecco":
        merged_stats_path = derived_dir / "hume_outgoing_ecco-ecco_hume_only_merged_stats.json"
    else:  # newspaper
        merged_stats_path = derived_dir / "hume_outgoing_ecco-newspaper_hume_only_merged_stats.json"
    
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
        except Exception as e:
            st.warning(f"Unable to load merge statistics: {e}")

with tab1:
    render_statistics_section(DERIVED_ECCO_DIR, "ecco")

with tab2:
    render_statistics_section(DERIVED_NEWSPAPER_DIR, "newspaper")

# Flowchart outside the tabs
st.markdown("#### Data processing flowchart")
flowchart = """
    digraph {
        rankdir=TB;
        node [shape=box, style=rounded];
        
    A [label="Raw data\nAll reuse records", fillcolor="#e1f5ff", style="filled,rounded"];
    B [label="Identify borrowed intervals\nborrowed_intervals.json", fillcolor="#fff4e1", style="filled,rounded"];
    C [label="Check overlap with\nborrowed intervals", shape=diamond, fillcolor="#fff9e1", style="filled"];
    D [label="Mark as 'not from Hume'\noutgoing_overlaps.json", fillcolor="#ffe1e1", style="filled,rounded"];
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

st.markdown("""
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
""")

st.header("Block Pairs Comparison", divider="rainbow")

blocks_tab1, blocks_tab2 = st.tabs(["ECCO-ECCO Blocks", "ECCO-Newspaper Blocks"])

def _get_newspaper_image_urls(src_section_id: str, src_doc_id: str | None = None) -> list[str]:
    """Fetch newspaper image URLs from Gale (same logic as lehti_streamlit.py)."""
    if not src_section_id:
        return []
    
    COLLECTIONS = {
        'nichols': {
            'prodId': 'NICN',
            'prefix': '',
        },
        'burney': {
            'prodId': 'BBCN',
            'prefix': 'Z',
        },
    }
    
    if src_doc_id and str(src_doc_id).upper().startswith('W'):
        collection = 'burney'
    elif src_doc_id and str(src_doc_id).upper().startswith('N'):
        collection = 'nichols'
    else:
        collection = 'nichols'
    
    config = COLLECTIONS.get(collection, COLLECTIONS['nichols'])
    prod_id = config['prodId']
    prefix = config['prefix']
    
    doc_id_str = str(src_section_id)
    if collection == 'burney' and not doc_id_str.startswith(prefix):
        gale_doc_id = f"{prefix}{doc_id_str}"
    else:
        gale_doc_id = doc_id_str
    
    target = (
        f"https://go.gale.com/ps/retrieve.do?"
        f"docId=GALE%7C{requests.utils.quote(gale_doc_id)}"
        f"&prodId={prod_id}"
        f"&userGroupName=uhelsink"
        f"&aty=ip"
    )
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        response = requests.get(target, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch from Gale: {e}")
    
    match = (
        re.search(r'var\s+dviResponse\s*=\s*(\{[\s\S]*?\});', html) or
        re.search(r'dviResponse\s*=\s*(\{[\s\S]*?\});', html)
    )
    
    if not match:
        raise Exception('dviResponse object not found in retrieved HTML')
    
    obj_text = match.group(1)
    
    try:
        sanitized = re.sub(r',\s*}', '}', obj_text)
        sanitized = re.sub(r',\s*]', ']', sanitized)
        try:
            json_str = sanitized.replace("'", '"')
            dvi_response = json.loads(json_str)
        except:
            dvi_response = eval(sanitized)
    except Exception as e:
        raise Exception(f'Error parsing dviResponse: {e}')
    
    if not dvi_response or not isinstance(dvi_response.get('pageDocuments'), list):
        raise Exception('dviResponse.pageDocuments missing or not an array')
    
    image_list = dvi_response.get('imageList', [])
    if isinstance(image_list, list):
        current_article_images = [img for img in image_list if img.get('currentArticle')]
        if current_article_images:
            image_urls = []
            for image in current_article_images:
                image_id = image.get('recordId')
                if image_id:
                    separator = '&' if '?' in image_id else '?'
                    url = f"https://luna.gale.com/imgsrv/FastFetch/UBER2/{image_id}{separator}format=jpeg"
                    image_urls.append(url)
            return image_urls
    
    return []

def _render_preview(url: str | None, mode: str, label: str, height: int = 420) -> None:
    if not url:
        st.info(f"{label} has no URL")
        return
    try:
        if mode == "Try displaying image":
            st.image(url, caption=label, use_container_width=True)
        elif mode == "Embed webpage":
            components.iframe(url, height=height)
    except Exception as exc:
        st.warning(f"{label} preview failed: {exc} (link still available above)")

def _render_newspaper_preview(src_section_id: str | None, src_doc_id: str | None, mode: str, label: str) -> None:
    """Render newspaper preview (fetch images via src_section_id)."""
    if not src_section_id:
        st.info(f"{label} has no src_section_id")
        return
    
    collection_info = ""
    if src_doc_id:
        if str(src_doc_id).upper().startswith('W'):
            collection_info = " (Burney collection)"
        elif str(src_doc_id).upper().startswith('N'):
            collection_info = " (Nichols collection)"
    st.info(f"src_section_id: `{src_section_id}`{collection_info}")
    if src_doc_id:
        st.info(f"src_doc_id: `{src_doc_id}`")
    
    if mode != "Links only":
        try:
            with st.spinner(f"Fetching images for {label}..."):
                image_urls = _get_newspaper_image_urls(src_section_id, src_doc_id)
                if image_urls:
                    st.success(f"Fetched {len(image_urls)} images")
                    for idx, img_url in enumerate(image_urls):
                        st.image(img_url, caption=f"{label} - Image {idx + 1}", use_container_width=True)
                else:
                    st.warning(f"No images found for {label} (src_section_id: {src_section_id})")
        except Exception as exc:
            st.error(f"Failed to fetch images for {label}: {exc}")
            st.info(f"Verify src_section_id `{src_section_id}` and src_doc_id `{src_doc_id}`, or try again later")


def _extract_year_from_doc_id(doc_id: str | None) -> int | None:
    """Try to extract a 4-digit year from doc_id."""
    if not doc_id:
        return None
    match = re.search(r'_(\d{4})_', doc_id)
    if match:
        year = int(match.group(1))
        if 1400 <= year <= 1900:
            return year
    match = re.search(r'(\d{4})', doc_id)
    if match:
        year = int(match.group(1))
        if 1400 <= year <= 1900:
            return year
    return None


def _infer_collection_from_doc_id(doc_id: str | None) -> str:
    """Infer the newspaper collection from doc_id."""
    if not doc_id:
        return "unknown"
    upper = doc_id.upper()
    if upper.startswith("W"):
        return "burney"
    if upper.startswith("N"):
        return "nichols"
    return "unknown"


def _parse_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        year = int(date_str.split("-", 1)[0])
    except (ValueError, AttributeError):
        return None
    return year if 1400 <= year <= 1900 else None


def _get_metadata(data_type: str) -> dict[str, dict]:
    if data_type in _METADATA_CACHE:
        return _METADATA_CACHE[data_type]

    metadata_path = DST_METADATA_FILES.get(data_type)
    src_headers: dict[tuple[str, str], str | None] = {}
    src_pub_dates: dict[str, str] = {}
    src_pub_years: dict[str, int] = {}
    dst_pub_dates: dict[str, str] = {}
    dst_pub_years: dict[str, int] = {}

    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for record in records:
                # For outgoing: src is the target essay (Hume's original)
                src_doc_id = record.get("src_doc_id")
                src_section_id = record.get("src_section_id")
                if src_doc_id is not None and src_section_id is not None:
                    key = (str(src_doc_id), str(src_section_id))
                    header = record.get("src_section_header")
                    if header and key not in src_headers:
                        src_headers[key] = header

                src_publication_date = record.get("src_publication_date")
                if src_doc_id and src_publication_date:
                    doc_id_str = str(src_doc_id)
                    src_pub_dates.setdefault(doc_id_str, src_publication_date)
                    year = _parse_year(src_publication_date)
                    if year is not None:
                        src_pub_years.setdefault(doc_id_str, year)

                # dst is the destination (where it's reprinted to)
                dst_doc_id = record.get("dst_doc_id")
                dst_publication_date = record.get("dst_publication_date")
                if dst_doc_id and dst_publication_date:
                    doc_id_str = str(dst_doc_id)
                    dst_pub_dates.setdefault(doc_id_str, dst_publication_date)
                    year = _parse_year(dst_publication_date)
                    if year is not None:
                        dst_pub_years.setdefault(doc_id_str, year)
        except Exception as exc:
            st.warning(f"Unable to load metadata for {data_type}: {exc}")
    else:
        st.warning(f"Metadata file for {data_type} not found: {metadata_path}")

    meta = {
        "src_headers": src_headers,
        "src_pub_dates": src_pub_dates,
        "src_pub_years": src_pub_years,
        "dst_pub_dates": dst_pub_dates,
        "dst_pub_years": dst_pub_years,
    }
    _METADATA_CACHE[data_type] = meta
    return meta


def _get_src_section_header(src_doc_id: str | None, src_section_id: str | int | None, data_type: str) -> str | None:
    """Get section header for target essay (src in outgoing data)."""
    if src_doc_id is None or src_section_id is None:
        return None
    headers = _get_metadata(data_type)["src_headers"]
    return headers.get((str(src_doc_id), str(src_section_id)))


def _ensure_src_headers(pairs: list[dict], data_type: str) -> None:
    """Ensure src_section_header is available for each pair (target essay)."""
    for pair in pairs:
        if pair.get("src_section_header"):
            continue
        header = _get_src_section_header(pair.get("src_doc_id"), pair.get("src_section_id"), data_type)
        if header:
            pair["src_section_header"] = header


def _get_src_publication_date(src_doc_id: str | None, data_type: str) -> str | None:
    if not src_doc_id:
        return None
    return _get_metadata(data_type)["src_pub_dates"].get(str(src_doc_id))


def _get_dst_publication_date(dst_doc_id: str | None, data_type: str) -> str | None:
    if not dst_doc_id:
        return None
    return _get_metadata(data_type)["dst_pub_dates"].get(str(dst_doc_id))


def _get_src_publication_year(src_doc_id: str | None, data_type: str) -> int | None:
    if not src_doc_id:
        return None
    return _get_metadata(data_type)["src_pub_years"].get(str(src_doc_id))


def _get_dst_publication_year(dst_doc_id: str | None, data_type: str) -> int | None:
    if not dst_doc_id:
        return None
    return _get_metadata(data_type)["dst_pub_years"].get(str(dst_doc_id))

def load_blocks_data(data_path: Path):
    """Load block data from a JSON file."""
    if not data_path.exists():
        return None
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Unable to load data: {e}")
        return None

def render_block_comparison(blocks_data, data_type: str):
    """Render the block comparison view."""
    if blocks_data is None:
        st.warning("Data file is missing or could not be loaded.")
        return
    
    st.info(f"Loaded {len(blocks_data)} block pair records.")
    
    overlap_ratios = [pair.get('overlap_ratio', 0) for pair in blocks_data]
    min_block_lengths = [pair.get('min_block_length', 0) for pair in blocks_data]
    
    min_overlap = min(overlap_ratios) if overlap_ratios else 0
    max_overlap = max(overlap_ratios) if overlap_ratios else 1
    min_length = min(min_block_lengths) if min_block_lengths else 0
    max_length = max(min_block_lengths) if min_block_lengths else 1000
    
    col1, col2 = st.columns(2)
    with col1:
        overlap_min = st.slider(
            "Minimum overlap_ratio",
            min_value=float(min_overlap),
            max_value=float(max_overlap),
            value=float(min_overlap),
            step=0.01,
            key=f"overlap_min_{data_type}"
        )
    with col2:
        overlap_max = st.slider(
            "Maximum overlap_ratio",
            min_value=float(min_overlap),
            max_value=float(max_overlap),
            value=float(max_overlap),
            step=0.01,
            key=f"overlap_max_{data_type}"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        length_min = st.slider(
            "Minimum min_block_length",
            min_value=int(min_length),
            max_value=int(max_length),
            value=int(min_length),
            step=1,
            key=f"length_min_{data_type}"
        )
    with col4:
        length_max = st.slider(
            "Maximum min_block_length",
            min_value=int(min_length),
            max_value=int(max_length),
            value=int(max_length),
            step=1,
            key=f"length_max_{data_type}"
        )
    
    filtered_data = [
        pair for pair in blocks_data
        if (overlap_min <= pair.get('overlap_ratio', 0) <= overlap_max and
            length_min <= pair.get('min_block_length', 0) <= length_max)
    ]
    
    st.success(f"{len(filtered_data)} records remain after filtering.")

    # Ensure src_section_header is available for each pair (target essay)
    _ensure_src_headers(filtered_data, data_type)
    
    essays_dict = {}
    for pair in filtered_data:
        src_doc_id = pair.get('src_doc_id')
        src_section_id = pair.get('src_section_id')
        essay_key = (src_doc_id, src_section_id)
        if essay_key not in essays_dict:
            essays_dict[essay_key] = []
        essays_dict[essay_key].append(pair)
    
    essay_options = [
        f"{doc_id} (Section {section_id}) - {len(pairs)} pairs"
        for (doc_id, section_id), pairs in sorted(essays_dict.items())
    ]
    
    if not essay_options:
        st.warning("No target essays matched the filters.")
        return
    
    selected_essay = st.selectbox(
        "Select target essay",
        options=essay_options,
        key=f"essay_selector_{data_type}"
    )
    
    selected_index = essay_options.index(selected_essay)
    selected_key = list(sorted(essays_dict.keys()))[selected_index]
    selected_pairs = essays_dict[selected_key]
    src_doc_id, src_section_id = selected_key
    src_section_header = _get_src_section_header(src_doc_id, src_section_id, data_type)
    src_publication_date = _get_src_publication_date(src_doc_id, data_type)
    src_publication_year = _get_src_publication_year(src_doc_id, data_type)
    
    st.info(f"**Target essay:** `{src_doc_id}` (Section `{src_section_id}`) - {len(selected_pairs)} block pairs")
    if src_section_header:
        st.caption(f"Section header: {src_section_header.strip()}")
    if src_publication_date:
        st.caption(f"Publication date: {src_publication_date}")
    elif src_publication_year:
        st.caption(f"Publication year: {src_publication_year}")
    
    all_src_starts = []
    all_src_ends = []
    for pair in selected_pairs:
        block_a = pair.get('block_a', {})
        block_b = pair.get('block_b', {})
        if block_a.get('src_trs_start') is not None:
            all_src_starts.append(block_a.get('src_trs_start'))
        if block_a.get('src_trs_end') is not None:
            all_src_ends.append(block_a.get('src_trs_end'))
        if block_b.get('src_trs_start') is not None:
            all_src_starts.append(block_b.get('src_trs_start'))
        if block_b.get('src_trs_end') is not None:
            all_src_ends.append(block_b.get('src_trs_end'))
    
    if all_src_starts and all_src_ends:
        essay_merged_start = min(all_src_starts)
        essay_merged_end = max(all_src_ends)
        first_pair = selected_pairs[0]
        block_a = first_pair.get('block_a', {})
        block_b = first_pair.get('block_b', {})
        url_template = block_a.get('src_trs_url') or block_b.get('src_trs_url')
        if url_template:
            doc_id_match = re.search(r'docId=([^&]+)', url_template)
            if doc_id_match:
                doc_id = doc_id_match.group(1)
                essay_merged_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}&offsetStart={essay_merged_start}&offsetEnd={essay_merged_end}"
                st.markdown(f"**Target essay full URL:** [{essay_merged_url}]({essay_merged_url})")
                # Note: src_section_url might not exist in outgoing data, check if available
                section_url_source = block_a.get('src_section_url') or block_b.get('src_section_url')
                if section_url_source:
                    urls = [u.strip() for u in section_url_source.split('\n') if u.strip()]
                    if urls:
                        st.markdown(f"**First page link:** [{urls[0]}]({urls[0]})")
    
    st.divider()
    
    preview_mode = st.radio(
        "URL preview mode",
        options=("Links only", "Embed webpage", "Try displaying image"),
        horizontal=True,
        key=f"preview_mode_{data_type}"
    )
    
    destination_entries: list[dict] = []

    for idx, pair in enumerate(selected_pairs):
        with st.expander(
            f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')} "
            f"(overlap: {pair.get('overlap_ratio', 0):.4f}, min_length: {pair.get('min_block_length', 0)})",
            expanded=False
        ):
            col_pair_info1, col_pair_info2 = st.columns(2)
            with col_pair_info1:
                st.markdown("**Pair details**")
                st.markdown(f"- overlap_ratio: `{pair.get('overlap_ratio', 0):.4f}`")
                st.markdown(f"- intersection_len: `{pair.get('intersection_len', 0)}`")
                st.markdown(f"- min_block_length: `{pair.get('min_block_length', 0)}`")
            
            with col_pair_info2:
                st.markdown("**Destination documents**")
                st.markdown(f"- dst_doc_id_a: `{pair.get('dst_doc_id_a', 'N/A')}`")
                st.markdown(f"- dst_doc_id_b: `{pair.get('dst_doc_id_b', 'N/A')}`")
            
            st.divider()
            
            st.markdown("### Target Essay -> Destination Blocks")
            col_block_a, col_block_b = st.columns(2)
            
            # Block A
            with col_block_a:
                if 'block_a' in pair:
                    st.markdown("#### Block A")
                    block_a = pair['block_a']

                    dst_a_pub_date = _get_dst_publication_date(block_a.get('dst_doc_id'), data_type)
                    destination_entries.append({
                        "dst_doc_id": block_a.get('dst_doc_id'),
                        "dst_publication_date": dst_a_pub_date,
                        "year": _get_dst_publication_year(block_a.get('dst_doc_id'), data_type) or _extract_year_from_doc_id(block_a.get('dst_doc_id')),
                        "block_label": "Block A",
                        "pair_index": idx + 1,
                        "pair_summary": f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')}",
                        "block": block_a,
                    })
                    
                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_a.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_a.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_a.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_a.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_a.get('src_piece_length', 'N/A')}`")
                    src_a_pub_date = _get_src_publication_date(block_a.get('src_doc_id'), data_type)
                    if src_a_pub_date:
                        st.markdown(f"- src_publication_date: `{src_a_pub_date}`")
                    if block_a.get('src_trs_url'):
                        st.markdown(f"- [src_trs_url]({block_a['src_trs_url']})")
                    
                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_a.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_a.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_a.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_a.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_a.get('fragment_count', 'N/A')}`")
                    dst_a_pub_date = _get_dst_publication_date(block_a.get('dst_doc_id'), data_type)
                    if dst_a_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_a_pub_date}`")
                    if block_a.get('dst_trs_url'):
                        st.markdown(f"- [dst_trs_url]({block_a['dst_trs_url']})")
            
            # Block B
            with col_block_b:
                if 'block_b' in pair:
                    st.markdown("#### Block B")
                    block_b = pair['block_b']

                    dst_b_pub_date = _get_dst_publication_date(block_b.get('dst_doc_id'), data_type)
                    destination_entries.append({
                        "dst_doc_id": block_b.get('dst_doc_id'),
                        "dst_publication_date": dst_b_pub_date,
                        "year": _get_dst_publication_year(block_b.get('dst_doc_id'), data_type) or _extract_year_from_doc_id(block_b.get('dst_doc_id')),
                        "block_label": "Block B",
                        "pair_index": idx + 1,
                        "pair_summary": f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')}",
                        "block": block_b,
                    })
                    
                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_b.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_b.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_b.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_b.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_b.get('src_piece_length', 'N/A')}`")
                    src_b_pub_date = _get_src_publication_date(block_b.get('src_doc_id'), data_type)
                    if src_b_pub_date:
                        st.markdown(f"- src_publication_date: `{src_b_pub_date}`")
                    if block_b.get('src_trs_url'):
                        st.markdown(f"- [src_trs_url]({block_b['src_trs_url']})")
                    
                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_b.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_b.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_b.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_b.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_b.get('fragment_count', 'N/A')}`")
                    dst_b_pub_date = _get_dst_publication_date(block_b.get('dst_doc_id'), data_type)
                    if dst_b_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_b_pub_date}`")
                    if block_b.get('dst_trs_url'):
                        st.markdown(f"- [dst_trs_url]({block_b['dst_trs_url']})")
            
            # Preview area
            if preview_mode != "Links only":
                st.divider()
                st.markdown("### Preview")
                
                # Target essay preview using merged range (src is the target essay)
                if 'block_a' in pair or 'block_b' in pair:
                    block_a = pair.get('block_a', {})
                    block_b = pair.get('block_b', {})
                    src_starts = []
                    src_ends = []
                    if block_a.get('src_trs_start') is not None:
                        src_starts.append(block_a.get('src_trs_start'))
                    if block_a.get('src_trs_end') is not None:
                        src_ends.append(block_a.get('src_trs_end'))
                    if block_b.get('src_trs_start') is not None:
                        src_starts.append(block_b.get('src_trs_start'))
                    if block_b.get('src_trs_end') is not None:
                        src_ends.append(block_b.get('src_trs_end'))
                    
                    if src_starts and src_ends:
                        merged_src_start = min(src_starts)
                        merged_src_end = max(src_ends)
                        url_template = block_a.get('src_trs_url') or block_b.get('src_trs_url')
                        if url_template:
                            doc_id_match = re.search(r'docId=([^&]+)', url_template)
                            if doc_id_match:
                                doc_id = doc_id_match.group(1)
                                merged_src_url = f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}&offsetStart={merged_src_start}&offsetEnd={merged_src_end}"
                                st.caption("Target Essay (merged range)")
                                _render_preview(merged_src_url, preview_mode, "Target Essay")
                
                # Destination previews
                prev_col_dst1, prev_col_dst2 = st.columns(2)
                with prev_col_dst1:
                    if 'block_a' in pair:
                        block_a = pair['block_a']
                        st.caption("Destination (Block A)")
                        if block_a.get('dst_trs_url'):
                            _render_preview(
                                block_a.get('dst_trs_url'),
                                preview_mode,
                                "Block A Destination",
                            )
                with prev_col_dst2:
                    if 'block_b' in pair:
                        block_b = pair['block_b']
                        st.caption("Destination (Block B)")
                        if block_b.get('dst_trs_url'):
                            _render_preview(
                                block_b.get('dst_trs_url'),
                                preview_mode,
                                "Block B Destination",
                            )

    # --- Destination Propagation Timeline (for most reprinted target essay) ---
    st.divider()
    st.markdown("### Destination Propagation Timeline")
    
    # Count reprints per target essay
    essay_reprint_counts: dict[tuple, int] = {}
    for pair in filtered_data:
        essay_key = (pair.get('src_doc_id'), pair.get('src_section_id'))
        essay_reprint_counts[essay_key] = essay_reprint_counts.get(essay_key, 0) + 1
    
    if not essay_reprint_counts:
        st.info("No target essays found for timeline visualization.")
        return
    
    # Find the target essay with most reprints
    most_reprinted_essay_key = max(essay_reprint_counts.items(), key=lambda x: x[1])[0]
    most_reprinted_src_doc_id, most_reprinted_src_section_id = most_reprinted_essay_key
    reprint_count = essay_reprint_counts[most_reprinted_essay_key]
    
    st.info(f"**Most reprinted target essay:** `{most_reprinted_src_doc_id}` (Section `{most_reprinted_src_section_id}`) - {reprint_count} reprints")
    
    # Collect all reprint events for this target essay
    timeline_entries = []
    for pair in filtered_data:
        if (pair.get('src_doc_id'), pair.get('src_section_id')) == most_reprinted_essay_key:
            block_a = pair.get('block_a', {})
            block_b = pair.get('block_b', {})
            for block in [block_a, block_b]:
                if block.get('dst_doc_id'):
                    dst_year = _get_dst_publication_year(block.get('dst_doc_id'), data_type)
                    if dst_year is None:
                        dst_year = _extract_year_from_doc_id(block.get('dst_doc_id'))
                    if dst_year:
                        dst_pub_date = _get_dst_publication_date(block.get('dst_doc_id'), data_type)
                        timeline_entries.append({
                            "year": dst_year,
                            "dst_doc_id": block.get('dst_doc_id'),
                            "dst_publication_date": dst_pub_date,
                            "block": block,
                            "pair": pair,
                        })
    
    if not timeline_entries:
        st.info("No reprint events with year information found for this target essay.")
        return
    
    # Create timeline visualization
    df_timeline = pd.DataFrame([
        {
            "year": entry["year"],
            "dst_doc_id": entry["dst_doc_id"],
            "publication_date": entry.get("dst_publication_date"),
        }
        for entry in timeline_entries
    ])
    
    if not df_timeline.empty:
        # Count reprints per year for each destination
        df_grouped = df_timeline.groupby(["year", "dst_doc_id"]).size().reset_index(name="count")
        
        fig = px.line(
            df_grouped,
            x="year",
            y="count",
            color="dst_doc_id",
            markers=True,
            hover_data=["dst_doc_id"],
            labels={
                "year": "Year",
                "count": "Reprint Count",
                "dst_doc_id": "Destination Doc ID",
            },
            title=f"Propagation timeline for target essay {most_reprinted_src_doc_id} (Section {most_reprinted_src_section_id})",
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timeline data is missing year information.")
    
    # Group by destination and year for detailed view
    destination_year_map: dict[tuple, list] = {}
    for entry in timeline_entries:
        key = (entry["dst_doc_id"], entry["year"])
        if key not in destination_year_map:
            destination_year_map[key] = []
        destination_year_map[key].append(entry)
    
    if destination_year_map:
        sorted_destinations = sorted(
            destination_year_map.items(),
            key=lambda x: (x[0][1], x[0][0])  # Sort by year, then dst_doc_id
        )
        
        st.markdown("#### Reprint events by year and destination")
        for (dst_doc_id, year), entries in sorted_destinations:
            dst_pub_date = entries[0].get("dst_publication_date")
            with st.expander(
                f"{year} - {dst_doc_id} ({len(entries)} reprint{'s' if len(entries) > 1 else ''})",
                expanded=False
            ):
                st.markdown(f"**Destination:** `{dst_doc_id}`")
                st.markdown(f"**Year:** `{year}`")
                if dst_pub_date:
                    st.markdown(f"**Publication date:** `{dst_pub_date}`")
                st.markdown(f"**Reprint count:** `{len(entries)}`")
                
                for idx, entry in enumerate(entries, start=1):
                    block = entry["block"]
                    st.markdown(f"**Reprint {idx}**")
                    st.markdown(f"- dst_trs_start: `{block.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block.get('dst_piece_length', 'N/A')}`")
                    if block.get('dst_trs_url'):
                        st.markdown(f"- [dst_trs_url]({block['dst_trs_url']})")
                    st.divider()

with blocks_tab1:
    ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
    ecco_blocks_data = load_blocks_data(ecco_blocks_path)
    render_block_comparison(ecco_blocks_data, "ecco")

with blocks_tab2:
    newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
    newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
    render_block_comparison(newspaper_blocks_data, "newspaper")

st.header("Network", divider="rainbow")
network_tab1, network_tab2 = st.tabs(["ECCO-ECCO Network", "ECCO-Newspaper Network"])


def _render_pair_summary(pair: dict, data_type: str, preview_mode: str, title: str | None = None) -> None:
    """Render a concise summary of a pair (used in the Network view)."""
    if title:
        st.markdown(f"**{title}**")

    src_pub_date = _get_src_publication_date(pair.get('src_doc_id'), data_type)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("**Pair details**")
        st.markdown(f"- overlap_ratio: `{pair.get('overlap_ratio', 0):.4f}`")
        st.markdown(f"- intersection_len: `{pair.get('intersection_len', 0)}`")
        st.markdown(f"- min_block_length: `{pair.get('min_block_length', 0)}`")
    with col_info2:
        st.markdown("**Target essay**")
        st.markdown(f"- src_doc_id: `{pair.get('src_doc_id', 'N/A')}`")
        st.markdown(f"- src_section_id: `{pair.get('src_section_id', 'N/A')}`")
        if pair.get("src_section_header"):
            st.markdown(f"- src_section_header: `{pair['src_section_header']}`")
        if src_pub_date:
            st.markdown(f"- src_publication_date: `{src_pub_date}`")

    st.markdown("**Block A**")
    block_a = pair.get("block_a")
    if block_a:
        dst_a_pub_date = _get_dst_publication_date(block_a.get('dst_doc_id'), data_type)
        st.markdown(
            f"- dst_doc_id: `{block_a.get('dst_doc_id', 'N/A')}` | "
            f"dst_trs_start: `{block_a.get('dst_trs_start', 'N/A')}` | "
            f"dst_piece_length: `{block_a.get('dst_piece_length', 'N/A')}`"
        )
        if dst_a_pub_date:
            st.markdown(f"  · dst_publication_date: `{dst_a_pub_date}`")
    else:
        st.markdown("- None")

    st.markdown("**Block B**")
    block_b = pair.get("block_b")
    if block_b:
        dst_b_pub_date = _get_dst_publication_date(block_b.get('dst_doc_id'), data_type)
        st.markdown(
            f"- dst_doc_id: `{block_b.get('dst_doc_id', 'N/A')}` | "
            f"dst_trs_start: `{block_b.get('dst_trs_start', 'N/A')}` | "
            f"dst_piece_length: `{block_b.get('dst_piece_length', 'N/A')}`"
        )
        if dst_b_pub_date:
            st.markdown(f"  · dst_publication_date: `{dst_b_pub_date}`")
    else:
        st.markdown("- None")

    if preview_mode != "Links only":
        st.divider()
        st.markdown("**Target Essay preview**")
        if block_a and block_a.get("src_trs_url"):
            _render_preview(
                block_a.get("src_trs_url"),
                preview_mode,
                "Target Essay (Block A)",
            )
        if block_b and block_b.get("src_trs_url"):
            _render_preview(
                block_b.get("src_trs_url"),
                preview_mode,
                "Target Essay (Block B)",
            )
        st.markdown("**Destination preview**")
        if block_a and block_a.get("dst_trs_url"):
            _render_preview(
                block_a.get("dst_trs_url"),
                preview_mode,
                "Block A Destination",
            )
        if block_b and block_b.get("dst_trs_url"):
            _render_preview(
                block_b.get("dst_trs_url"),
                preview_mode,
                "Block B Destination",
            )


def render_network_view(blocks_data: list[dict] | None, data_type: str) -> None:
    """Render the propagation network view."""
    if not blocks_data:
        st.warning("No data available.")
        return

    _ensure_src_headers(blocks_data, data_type)

    overlap_values = [pair.get("overlap_ratio", 0) for pair in blocks_data]
    min_ratio = float(min(overlap_values)) if overlap_values else 0.0
    max_ratio = float(max(overlap_values)) if overlap_values else 1.0

    st.markdown("#### Filters")
    ratio_threshold = st.slider(
        "Minimum overlap_ratio (keep reprints at or above this value)",
        min_value=min_ratio,
        max_value=max_ratio,
        value=min_ratio,
        step=0.01,
        key=f"network_overlap_{data_type}",
    )

    filtered_pairs = [
        pair for pair in blocks_data if pair.get("overlap_ratio", 0) >= ratio_threshold
    ]
    if not filtered_pairs:
        st.info("No reprints satisfy the filter.")
        return

    timeline_entries = []
    for pair in filtered_pairs:
        years = []
        for block_key in ("block_a", "block_b"):
            block = pair.get(block_key)
            if block:
                # Use dst_publication_date for timeline (when it was reprinted)
                year = _get_dst_publication_year(block.get("dst_doc_id"), data_type)
                if year is None:
                    year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                if year:
                    years.append(year)
        if not years:
            continue
        event_year = min(years)
        decade = (event_year // 10) * 10
        timeline_entries.append(
            {
                "pair": pair,
                "year": event_year,
                "decade": decade,
                "src_doc_id": pair.get("src_doc_id"),
                "src_section_id": pair.get("src_section_id"),
                "src_section_header": pair.get("src_section_header"),
            }
        )

    if not timeline_entries:
        st.info("Filtered reprints lack year information, so the chart cannot be generated.")
        return

    df_timeline = pd.DataFrame(
        [{"decade": entry["decade"], "count": 1} for entry in timeline_entries]
    )
    decade_summary = df_timeline.groupby("decade", as_index=False).sum()

    st.markdown("#### Reprint counts per decade (10-year intervals)")
    fig = px.bar(
        decade_summary,
        x="decade",
        y="count",
        labels={"decade": "Start year", "count": "Reprint count"},
        title="Reprints per decade (10-year bins)",
    )
    fig.update_traces(marker_color="#4C78A8")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    available_decades = sorted(decade_summary["decade"].tolist())
    selected_decade = st.selectbox(
        "Select a decade to review",
        options=available_decades,
        format_func=lambda x: f"{x}-{x + 9}",
        key=f"network_decade_{data_type}",
    )

    selected_entries = [entry for entry in timeline_entries if entry["decade"] == selected_decade]
    if not selected_entries:
        st.info("No records exist for the selected decade.")
        return

    essay_map: dict[tuple, dict] = {}
    for entry in selected_entries:
        key = (entry["src_doc_id"], entry["src_section_id"])
        bucket = essay_map.setdefault(
            key,
            {
                "src_doc_id": entry["src_doc_id"],
                "src_section_id": entry["src_section_id"],
                "src_section_header": entry.get("src_section_header"),
                "src_publication_date": _get_src_publication_date(entry["src_doc_id"], data_type),
                "pairs": [],
            },
        )
        bucket["pairs"].append(entry["pair"])

    sorted_essays = sorted(
        essay_map.values(),
        key=lambda item: len(item["pairs"]),
        reverse=True,
    )

    st.markdown(
        f"#### Target essays between {selected_decade}-{selected_decade + 9} (sorted by reprint count)"
    )
    
    if not sorted_essays:
        st.info("No target essays found for the selected decade.")
        return
    
    essay_options = [
        f"{essay['src_doc_id']} (Section {essay['src_section_id']}) - {len(essay['pairs'])} reprints"
        for essay in sorted_essays
    ]
    
    selected_essay_idx = st.selectbox(
        "Select target essay",
        options=list(range(len(essay_options))),
        format_func=lambda idx: essay_options[idx],
        key=f"network_essay_selector_{data_type}",
    )
    
    selected_essay = sorted_essays[selected_essay_idx]
    
    st.info(
        f"**Target essay:** `{selected_essay['src_doc_id']}` (Section `{selected_essay['src_section_id']}`) - {len(selected_essay['pairs'])} reprints"
    )
    if selected_essay.get("src_section_header"):
        st.caption(f"Section header: {selected_essay['src_section_header'].strip()}")
    if selected_essay.get("src_publication_date"):
        st.caption(f"Publication date: {selected_essay['src_publication_date']}")
    
    preview_mode = st.radio(
        "URL preview mode (Network view)",
        options=("Links only", "Embed webpage", "Try displaying image"),
        horizontal=True,
        key=f"network_preview_mode_{data_type}",
    )
    
    st.divider()
    
    for pair_idx, pair in enumerate(selected_essay["pairs"], start=1):
        st.markdown(f"**Reprint {pair_idx}**")
        _render_pair_summary(
            pair,
            data_type,
            preview_mode,
            title=None,
        )
        st.divider()


with network_tab1:
    if 'ecco_blocks_data' in locals() and ecco_blocks_data is not None:
        render_network_view(ecco_blocks_data, "ecco")
    else:
        ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
        ecco_blocks_data = load_blocks_data(ecco_blocks_path)
        render_network_view(ecco_blocks_data, "ecco")

with network_tab2:
    if 'newspaper_blocks_data' in locals() and newspaper_blocks_data is not None:
        render_network_view(newspaper_blocks_data, "newspaper")
    else:
        newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
        newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
        render_network_view(newspaper_blocks_data, "newspaper")

st.header("Target Essay Propagation Analysis", divider="rainbow")
propagation_tab1, propagation_tab2 = st.tabs(["ECCO-ECCO Propagation", "ECCO-Newspaper Propagation"])


def render_propagation_analysis(blocks_data: list[dict] | None, data_type: str) -> None:
    """Render target essay propagation analysis with reprint counts and timeline."""
    if not blocks_data:
        st.warning("No data available.")
        return
    
    _ensure_src_headers(blocks_data, data_type)
    
    overlap_values = [pair.get("overlap_ratio", 0) for pair in blocks_data]
    min_ratio = float(min(overlap_values)) if overlap_values else 0.0
    max_ratio = float(max(overlap_values)) if overlap_values else 1.0
    
    st.markdown("#### Filters")
    ratio_threshold = st.slider(
        "Minimum overlap_ratio (keep reprints at or above this value)",
        min_value=min_ratio,
        max_value=max_ratio,
        value=min_ratio,
        step=0.01,
        key=f"propagation_overlap_{data_type}",
    )
    
    filtered_pairs = [
        pair for pair in blocks_data if pair.get("overlap_ratio", 0) >= ratio_threshold
    ]
    if not filtered_pairs:
        st.info("No reprints satisfy the filter.")
        return
    
    # Count reprints per target essay
    essay_reprint_counts: dict[tuple, dict] = {}
    for pair in filtered_pairs:
        essay_key = (pair.get('src_doc_id'), pair.get('src_section_id'))
        if essay_key not in essay_reprint_counts:
            essay_reprint_counts[essay_key] = {
                "src_doc_id": pair.get('src_doc_id'),
                "src_section_id": pair.get('src_section_id'),
                "src_section_header": pair.get('src_section_header'),
                "count": 0,
            }
        essay_reprint_counts[essay_key]["count"] += 1
    
    if not essay_reprint_counts:
        st.info("No target essays found.")
        return
    
    # Sort by reprint count (descending)
    sorted_essays = sorted(
        essay_reprint_counts.values(),
        key=lambda x: x["count"],
        reverse=True,
    )
    
    st.markdown("#### Target Essays (sorted by reprint count)")
    
    essay_options = [
        f"{essay['src_doc_id']} (Section {essay['src_section_id']}) - {essay['count']} reprints"
        for essay in sorted_essays
    ]
    
    selected_essay_idx = st.selectbox(
        "Select a target essay to view its propagation timeline",
        options=list(range(len(essay_options))),
        format_func=lambda idx: essay_options[idx],
        key=f"propagation_essay_selector_{data_type}",
    )
    
    selected_essay = sorted_essays[selected_essay_idx]
    selected_essay_key = (selected_essay["src_doc_id"], selected_essay["src_section_id"])
    
    st.info(
        f"**Selected target essay:** `{selected_essay['src_doc_id']}` "
        f"(Section `{selected_essay['src_section_id']}`) - {selected_essay['count']} reprints"
    )
    if selected_essay.get("src_section_header"):
        st.caption(f"Section header: {selected_essay['src_section_header'].strip()}")
    
    # Collect all reprint events for the selected target essay
    timeline_entries = []
    for pair in filtered_pairs:
        if (pair.get('src_doc_id'), pair.get('src_section_id')) == selected_essay_key:
            block_a = pair.get('block_a', {})
            block_b = pair.get('block_b', {})
            for block in [block_a, block_b]:
                if block.get('dst_doc_id'):
                    dst_year = _get_dst_publication_year(block.get('dst_doc_id'), data_type)
                    if dst_year is None:
                        dst_year = _extract_year_from_doc_id(block.get('dst_doc_id'))
                    if dst_year:
                        dst_pub_date = _get_dst_publication_date(block.get('dst_doc_id'), data_type)
                        timeline_entries.append({
                            "year": dst_year,
                            "dst_doc_id": block.get('dst_doc_id'),
                            "dst_publication_date": dst_pub_date,
                            "block": block,
                            "pair": pair,
                        })
    
    if not timeline_entries:
        st.info("No reprint events with year information found for this target essay.")
        return
    
    # Create timeline visualization
    df_timeline = pd.DataFrame([
        {
            "year": entry["year"],
            "dst_doc_id": entry["dst_doc_id"],
            "publication_date": entry.get("dst_publication_date"),
        }
        for entry in timeline_entries
    ])
    
    if not df_timeline.empty:
        # Count reprints per year for each destination
        df_grouped = df_timeline.groupby(["year", "dst_doc_id"]).size().reset_index(name="count")
        
        st.markdown("#### Propagation Timeline")
        fig = px.line(
            df_grouped,
            x="year",
            y="count",
            color="dst_doc_id",
            markers=True,
            hover_data=["dst_doc_id"],
            labels={
                "year": "Year",
                "count": "Reprint Count",
                "dst_doc_id": "Destination Doc ID",
            },
            title=f"Propagation timeline for target essay {selected_essay['src_doc_id']} (Section {selected_essay['src_section_id']})",
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timeline data is missing year information.")


with propagation_tab1:
    if 'ecco_blocks_data' in locals() and ecco_blocks_data is not None:
        render_propagation_analysis(ecco_blocks_data, "ecco")
    else:
        ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
        ecco_blocks_data = load_blocks_data(ecco_blocks_path)
        render_propagation_analysis(ecco_blocks_data, "ecco")

with propagation_tab2:
    if 'newspaper_blocks_data' in locals() and newspaper_blocks_data is not None:
        render_propagation_analysis(newspaper_blocks_data, "newspaper")
    else:
        newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
        newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
        render_propagation_analysis(newspaper_blocks_data, "newspaper")