import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components


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

DST_METADATA_FILES = {
    "ecco": DERIVED_ECCO_DIR / "hume_outgoing_ecco-ecco_original_only_merged_with_urls.json",
    "newspaper": DERIVED_NEWSPAPER_DIR / "hume_outgoing_ecco-newspaper_original_only_merged_with_urls.json",
}

NEWSPAPER_ID_MAPPING_PATH = DERIVED_NEWSPAPER_DIR / "newspaper_id_mapping.json"

_METADATA_CACHE: dict[str, dict[str, dict]] = {}
_NEWSPAPER_ARTICLE_MAPPING_CACHE: dict[str, dict[str, str]] | None = None


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


def _get_src_publication_date(src_doc_id: str | None, data_type: str) -> str | None:
    if not src_doc_id:
        return None
    return _get_metadata(data_type)["src_pub_dates"].get(str(src_doc_id))


def _get_src_publication_year(src_doc_id: str | None, data_type: str) -> int | None:
    if not src_doc_id:
        return None
    return _get_metadata(data_type)["src_pub_years"].get(str(src_doc_id))


def _get_dst_publication_date(dst_doc_id: str | None, data_type: str) -> str | None:
    if not dst_doc_id:
        return None
    return _get_metadata(data_type)["dst_pub_dates"].get(str(dst_doc_id))


def _get_dst_publication_year(dst_doc_id: str | None, data_type: str) -> int | None:
    if not dst_doc_id:
        return None
    return _get_metadata(data_type)["dst_pub_years"].get(str(dst_doc_id))


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


def _extract_year_from_doc_id(doc_id: str | None) -> int | None:
    if not doc_id:
        return None
    match = re.search(r"_(\d{4})_", doc_id)
    if match:
        year = int(match.group(1))
        if 1400 <= year <= 1900:
            return year
    match = re.search(r"(\d{4})", doc_id)
    if match:
        year = int(match.group(1))
        if 1400 <= year <= 1900:
            return year
    return None


def _load_newspaper_article_mapping() -> dict[str, dict[str, str]]:
    """Load newspaper article mapping and create a cache of articleID -> {articleAssetID, articleType}."""
    global _NEWSPAPER_ARTICLE_MAPPING_CACHE
    if _NEWSPAPER_ARTICLE_MAPPING_CACHE is not None:
        return _NEWSPAPER_ARTICLE_MAPPING_CACHE

    _NEWSPAPER_ARTICLE_MAPPING_CACHE = {}
    if not NEWSPAPER_ID_MAPPING_PATH.exists():
        st.warning(f"Newspaper ID mapping file not found: {NEWSPAPER_ID_MAPPING_PATH}")
        return _NEWSPAPER_ARTICLE_MAPPING_CACHE

    try:
        with open(NEWSPAPER_ID_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        for entry in mapping_data:
            pages = entry.get("pages", [])
            for page in pages:
                articles = page.get("articles", [])
                for article in articles:
                    article_id = article.get("articleID")
                    article_asset_id = article.get("articleAssetID")
                    article_type = article.get("articleType")
                    if article_id and article_asset_id:
                        _NEWSPAPER_ARTICLE_MAPPING_CACHE[article_id] = {
                            "articleAssetID": article_asset_id,
                            "articleType": article_type or "Unknown",
                        }
    except Exception as exc:
        st.warning(f"Failed to load newspaper article mapping: {exc}")
        _NEWSPAPER_ARTICLE_MAPPING_CACHE = {}

    return _NEWSPAPER_ARTICLE_MAPPING_CACHE


def _get_newspaper_image_urls(dst_doc_id: str, src_section_id: str | None = None) -> list[str]:
    """Get newspaper image URLs using articleAssetID from newspaper_id_mapping.json."""
    if not dst_doc_id:
        return []

    # dst_doc_id is the articleID
    article_id = dst_doc_id

    # Load mapping and find articleAssetID
    mapping = _load_newspaper_article_mapping()
    article_info = mapping.get(article_id)
    if not article_info:
        st.warning(f"Could not find articleAssetID for articleID: {article_id}")
        return []

    article_asset_id = article_info.get("articleAssetID")
    article_type = article_info.get("articleType", "Unknown")
    if not article_asset_id:
        st.warning(f"articleAssetID not found for articleID: {article_id}")
        return []

    collections = {
        "nichols": {"prodId": "NICN", "prefix": ""},
        "burney": {"prodId": "BBCN", "prefix": "Z"},
    }

    # Determine collection from articleID
    if article_id.upper().startswith("W"):
        collection = "burney"
    elif article_id.upper().startswith("N"):
        collection = "nichols"
    else:
        collection = "nichols"

    config = collections.get(collection, collections["nichols"])
    prod_id = config["prodId"]
    prefix = config["prefix"]

    # Use articleAssetID to build the request
    gale_doc_id = f"{prefix}{article_asset_id}" if collection == "burney" and not article_asset_id.startswith(prefix) else article_asset_id

    target = (
        f"https://go.gale.com/ps/retrieve.do?"
        f"docId=GALE%7C{requests.utils.quote(gale_doc_id)}"
        f"&prodId={prod_id}"
        f"&userGroupName=uhelsink"
        f"&aty=ip"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(target, headers=headers, timeout=30, allow_redirects=True)
    response.raise_for_status()
    html = response.text

    match = (
        re.search(r"var\s+dviResponse\s*=\s*(\{[\s\S]*?\});", html)
        or re.search(r"dviResponse\s*=\s*(\{[\s\S]*?\});", html)
    )
    if not match:
        raise RuntimeError("dviResponse object not found in retrieved HTML")

    obj_text = match.group(1)
    sanitized = re.sub(r",\s*}", "}", obj_text)
    sanitized = re.sub(r",\s*]", "]", sanitized)

    try:
        dvi_response = json.loads(sanitized.replace("'", '"'))
    except Exception:
        dvi_response = eval(sanitized)  # noqa: S307

    if not isinstance(dvi_response.get("pageDocuments"), list):
        raise RuntimeError("dviResponse.pageDocuments missing or not an array")

    image_list = dvi_response.get("imageList", [])
    if isinstance(image_list, list):
        current_article_images = [img for img in image_list if img.get("currentArticle")]
        image_urls: list[str] = []
        for image in current_article_images:
            image_id = image.get("recordId")
            if image_id:
                separator = "&" if "?" in image_id else "?"
                url = f"https://luna.gale.com/imgsrv/FastFetch/UBER2/{image_id}{separator}format=jpeg"
                image_urls.append(url)
        if image_urls:
            return image_urls
    return []


def _render_newspaper_preview(dst_doc_id: str | None, src_section_id: str | None, mode: str, label: str) -> None:
    if not dst_doc_id:
        st.info(f"{label} has no dst_doc_id (article ID)")
        return

    collection_info = ""
    if dst_doc_id:
        if str(dst_doc_id).upper().startswith("W"):
            collection_info = " (Burney collection)"
        elif str(dst_doc_id).upper().startswith("N"):
            collection_info = " (Nichols collection)"
    
    # Get article info from mapping
    mapping = _load_newspaper_article_mapping()
    article_info = mapping.get(dst_doc_id, {})
    article_type = article_info.get("articleType", "Unknown")
    
    st.info(f"Article ID: `{dst_doc_id}`{collection_info}")
    st.info(f"Article Type: `{article_type}`")
    if src_section_id:
        st.info(f"src_section_id: `{src_section_id}`")

    if mode == "Links only":
        return

    with st.spinner(f"Fetching images for {label}..."):
        try:
            image_urls = _get_newspaper_image_urls(dst_doc_id, src_section_id)
        except Exception as exc:
            st.error(f"Failed to fetch images: {exc}")
            return

    if image_urls:
        st.success(f"Fetched {len(image_urls)} images")
        for idx, img_url in enumerate(image_urls):
            st.image(img_url, caption=f"{label} - Image {idx + 1}", use_container_width=True)
    else:
        st.warning(f"No images found for {label} (article ID: {dst_doc_id})")


def load_blocks_data(data_path: Path):
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        return None
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Unable to load data: {exc}")
        return None


def _render_pair_summary(pair: dict, data_type: str, preview_mode: str, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")

    src_pub_date = _get_src_publication_date(pair.get("src_doc_id"), data_type)

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

    # Block A and Block B side by side
    col_block_a, col_block_b = st.columns(2)
    
    with col_block_a:
        st.markdown("**Block A**")
        block_a = pair.get("block_a")
        if block_a:
            dst_a_pub_date = _get_dst_publication_date(block_a.get("dst_doc_id"), data_type)
            st.markdown(
                f"- dst_doc_id: `{block_a.get('dst_doc_id', 'N/A')}` | "
                f"dst_trs_start: `{block_a.get('dst_trs_start', 'N/A')}` | "
                f"dst_piece_length: `{block_a.get('dst_piece_length', 'N/A')}`"
            )
            if dst_a_pub_date:
                st.markdown(f"  · dst_publication_date: `{dst_a_pub_date}`")
        else:
            st.markdown("- None")

    with col_block_b:
        st.markdown("**Block B**")
        block_b = pair.get("block_b")
        if block_b:
            dst_b_pub_date = _get_dst_publication_date(block_b.get("dst_doc_id"), data_type)
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
        prev_col_src1, prev_col_src2 = st.columns(2)
        with prev_col_src1:
            if block_a and block_a.get("src_trs_url"):
                st.markdown("- Block A Source Preview")
                components.iframe(block_a["src_trs_url"], height=420)
        with prev_col_src2:
            if block_b and block_b.get("src_trs_url"):
                st.markdown("- Block B Source Preview")
                components.iframe(block_b["src_trs_url"], height=420)
        
        # Newspaper images (for newspaper data type)
        if data_type == "newspaper":
            st.markdown("**Newspaper Images**")
            prev_col_newspaper1, prev_col_newspaper2 = st.columns(2)
            with prev_col_newspaper1:
                if block_a:
                    dst_doc_id_a = block_a.get("dst_doc_id")
                    src_section_id_a = block_a.get("src_section_id")
                    if dst_doc_id_a:
                        _render_newspaper_preview(
                            dst_doc_id_a,
                            src_section_id_a,
                            preview_mode,
                            "Block A Source (Newspaper)",
                        )
            with prev_col_newspaper2:
                if block_b:
                    dst_doc_id_b = block_b.get("dst_doc_id")
                    src_section_id_b = block_b.get("src_section_id")
                    if dst_doc_id_b:
                        _render_newspaper_preview(
                            dst_doc_id_b,
                            src_section_id_b,
                            preview_mode,
                            "Block B Source (Newspaper)",
                        )
        
        st.markdown("**Destination preview**")
        prev_col_dst1, prev_col_dst2 = st.columns(2)
        with prev_col_dst1:
            if block_a and block_a.get("dst_trs_url"):
                _render_pair_link(block_a["dst_trs_url"], preview_mode, "Block A Destination")
        with prev_col_dst2:
            if block_b and block_b.get("dst_trs_url"):
                _render_pair_link(block_b["dst_trs_url"], preview_mode, "Block B Destination")


def _render_pair_link(url: str, preview_mode: str, label: str) -> None:
    if preview_mode == "Embed webpage":
        components.iframe(url, height=420)
    elif preview_mode == "Try displaying image":
        st.image(url, caption=label, use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")


def render_network_view(blocks_data: list[dict] | None, data_type: str) -> None:
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
        f"**Target essay:** `{selected_essay['src_doc_id']}` (Section `{selected_essay['src_section_id']}`) - "
        f"{len(selected_essay['pairs'])} reprints"
    )
    if selected_essay.get("src_section_header"):
        st.caption(f"Section header: {selected_essay['src_section_header'].strip()}")
    if selected_essay.get("src_publication_date"):
        st.caption(f"Publication date: {selected_essay['src_publication_date']}")

    # Collect reprint years for this target essay
    reprint_years = []
    for pair in selected_essay["pairs"]:
        for block_key in ("block_a", "block_b"):
            block = pair.get(block_key)
            if block:
                year = _get_dst_publication_year(block.get("dst_doc_id"), data_type)
                if year is None:
                    year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                if year:
                    reprint_years.append(year)

    # Create timeline visualization
    if reprint_years:
        st.markdown("#### Reprint Timeline")
        year_counts = {}
        for year in reprint_years:
            year_counts[year] = year_counts.get(year, 0) + 1

        df_timeline = pd.DataFrame(
            [
                {"year": year, "count": count}
                for year, count in sorted(year_counts.items())
            ]
        )

        fig = px.line(
            df_timeline,
            x="year",
            y="count",
            markers=True,
            labels={"year": "Year", "count": "Reprint Count"},
            title=f"Reprint timeline for target essay {selected_essay['src_doc_id']} (Section {selected_essay['src_section_id']})",
        )
        fig.update_traces(marker=dict(size=8), line=dict(width=2))
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reprints", len(reprint_years))
        with col2:
            st.metric("Years Span", f"{min(reprint_years)} - {max(reprint_years)}")
        with col3:
            st.metric("Unique Years", len(year_counts))
    else:
        st.info("No year information available for reprints of this target essay.")

    st.divider()

    preview_mode = st.radio(
        "URL preview mode (Network view)",
        options=("Links only", "Embed webpage", "Try displaying image"),
        horizontal=True,
        key=f"network_preview_mode_{data_type}",
    )

    # Pagination controls
    st.markdown("#### Reprint List")
    total_pairs = len(selected_essay["pairs"])
    items_per_page = st.slider(
        "Items per page",
        min_value=10,
        max_value=min(100, total_pairs),
        value=min(20, total_pairs),
        step=10,
        key=f"network_pagination_{data_type}",
    )
    
    total_pages = (total_pairs + items_per_page - 1) // items_per_page
    if total_pages > 1:
        page_num = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"network_page_{data_type}",
        )
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        pairs_to_show = selected_essay["pairs"][start_idx:end_idx]
        st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_pairs)} of {total_pairs} reprints")
        pair_start_num = start_idx + 1
    else:
        pairs_to_show = selected_essay["pairs"]
        pair_start_num = 1

    st.divider()

    # Use expanders for lazy loading - only render preview when expanded
    for pair_idx, pair in enumerate(pairs_to_show, start=pair_start_num):
        # Create a summary line for the expander title
        block_a = pair.get("block_a", {})
        block_b = pair.get("block_b", {})
        dst_a = block_a.get("dst_doc_id", "N/A") if block_a else "N/A"
        dst_b = block_b.get("dst_doc_id", "N/A") if block_b else "N/A"
        overlap = pair.get("overlap_ratio", 0)
        
        with st.expander(
            f"Reprint {pair_idx}: {dst_a} <-> {dst_b} (overlap: {overlap:.4f})",
            expanded=False,
        ):
            _render_pair_summary(pair, data_type, preview_mode)


def render_propagation_analysis(blocks_data: list[dict] | None, data_type: str) -> None:
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

    essay_reprint_counts: dict[tuple, dict] = {}
    for pair in filtered_pairs:
        essay_key = (pair.get("src_doc_id"), pair.get("src_section_id"))
        if essay_key not in essay_reprint_counts:
            essay_reprint_counts[essay_key] = {
                "src_doc_id": pair.get("src_doc_id"),
                "src_section_id": pair.get("src_section_id"),
                "src_section_header": pair.get("src_section_header"),
                "count": 0,
            }
        essay_reprint_counts[essay_key]["count"] += 1

    if not essay_reprint_counts:
        st.info("No target essays found.")
        return

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

    timeline_entries = []
    for pair in filtered_pairs:
        if (pair.get("src_doc_id"), pair.get("src_section_id")) == selected_essay_key:
            for block in (pair.get("block_a", {}), pair.get("block_b", {})):
                if block.get("dst_doc_id"):
                    dst_year = _get_dst_publication_year(block.get("dst_doc_id"), data_type)
                    if dst_year is None:
                        dst_year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                    if dst_year:
                        dst_pub_date = _get_dst_publication_date(block.get("dst_doc_id"), data_type)
                        timeline_entries.append(
                            {
                                "year": dst_year,
                                "dst_doc_id": block.get("dst_doc_id"),
                                "dst_publication_date": dst_pub_date,
                                "block": block,
                                "pair": pair,
                            }
                        )

    if not timeline_entries:
        st.info("No reprint events with year information found for this target essay.")
        return

    df_timeline = pd.DataFrame(
        [
            {"year": entry["year"], "dst_doc_id": entry["dst_doc_id"], "publication_date": entry.get("dst_publication_date")}
            for entry in timeline_entries
        ]
    )

    if not df_timeline.empty:
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


def render_comparison_view(
    ecco_blocks_data: list[dict] | None,
    newspaper_blocks_data: list[dict] | None,
) -> None:
    """Compare year span of a target essay between ecco-ecco and ecco-newspaper datasets."""
    if not ecco_blocks_data or not newspaper_blocks_data:
        st.warning("Both datasets must be loaded for comparison.")
        return

    # Ensure headers are loaded
    _ensure_src_headers(ecco_blocks_data, "ecco")
    _ensure_src_headers(newspaper_blocks_data, "newspaper")

    # Collect all unique target essays from both datasets
    essay_map: dict[tuple, dict] = {}
    
    for pair in ecco_blocks_data:
        key = (pair.get("src_doc_id"), pair.get("src_section_id"))
        if key not in essay_map:
            essay_map[key] = {
                "src_doc_id": pair.get("src_doc_id"),
                "src_section_id": pair.get("src_section_id"),
                "src_section_header": pair.get("src_section_header"),
                "ecco_count": 0,
                "newspaper_count": 0,
            }
        essay_map[key]["ecco_count"] += 1
    
    for pair in newspaper_blocks_data:
        key = (pair.get("src_doc_id"), pair.get("src_section_id"))
        if key not in essay_map:
            essay_map[key] = {
                "src_doc_id": pair.get("src_doc_id"),
                "src_section_id": pair.get("src_section_id"),
                "src_section_header": pair.get("src_section_header"),
                "ecco_count": 0,
                "newspaper_count": 0,
            }
        essay_map[key]["newspaper_count"] += 1

    # Filter essays that appear in both datasets
    common_essays = [
        essay for essay in essay_map.values()
        if essay["ecco_count"] > 0 and essay["newspaper_count"] > 0
    ]

    if not common_essays:
        st.info("No target essays found in both datasets.")
        return

    # Sort by total reprint count
    sorted_essays = sorted(
        common_essays,
        key=lambda x: x["ecco_count"] + x["newspaper_count"],
        reverse=True,
    )

    st.markdown("#### Select Target Essay")
    essay_options = [
        f"{essay['src_doc_id']} (Section {essay['src_section_id']}) - "
        f"ECCO: {essay['ecco_count']}, Newspaper: {essay['newspaper_count']}"
        for essay in sorted_essays
    ]

    selected_essay_idx = st.selectbox(
        "Choose a target essay to compare",
        options=list(range(len(essay_options))),
        format_func=lambda idx: essay_options[idx],
        key="comparison_essay_selector",
    )

    selected_essay = sorted_essays[selected_essay_idx]
    selected_key = (selected_essay["src_doc_id"], selected_essay["src_section_id"])

    st.info(
        f"**Target essay:** `{selected_essay['src_doc_id']}` "
        f"(Section `{selected_essay['src_section_id']}`)"
    )
    if selected_essay.get("src_section_header"):
        st.caption(f"Section header: {selected_essay['src_section_header'].strip()}")

    # Collect years from both datasets
    ecco_years = []
    newspaper_years = []

    for pair in ecco_blocks_data:
        if (pair.get("src_doc_id"), pair.get("src_section_id")) == selected_key:
            for block_key in ("block_a", "block_b"):
                block = pair.get(block_key)
                if block:
                    year = _get_dst_publication_year(block.get("dst_doc_id"), "ecco")
                    if year is None:
                        year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                    if year:
                        ecco_years.append(year)

    for pair in newspaper_blocks_data:
        if (pair.get("src_doc_id"), pair.get("src_section_id")) == selected_key:
            for block_key in ("block_a", "block_b"):
                block = pair.get(block_key)
                if block:
                    year = _get_dst_publication_year(block.get("dst_doc_id"), "newspaper")
                    if year is None:
                        year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                    if year:
                        newspaper_years.append(year)

    # Create comparison visualization
    st.markdown("#### Year Span Comparison")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ECCO-ECCO**")
        if ecco_years:
            year_counts_ecco = {}
            for year in ecco_years:
                year_counts_ecco[year] = year_counts_ecco.get(year, 0) + 1
            
            df_ecco = pd.DataFrame(
                [
                    {"year": year, "count": count, "dataset": "ECCO-ECCO"}
                    for year, count in sorted(year_counts_ecco.items())
                ]
            )
            
            st.metric("Total Reprints", len(ecco_years))
            st.metric("Year Span", f"{min(ecco_years)} - {max(ecco_years)}")
            st.metric("Unique Years", len(year_counts_ecco))
        else:
            st.info("No reprints found in ECCO-ECCO dataset.")
            df_ecco = pd.DataFrame(columns=["year", "count", "dataset"])

    with col2:
        st.markdown("**ECCO-Newspaper**")
        if newspaper_years:
            year_counts_newspaper = {}
            for year in newspaper_years:
                year_counts_newspaper[year] = year_counts_newspaper.get(year, 0) + 1
            
            df_newspaper = pd.DataFrame(
                [
                    {"year": year, "count": count, "dataset": "ECCO-Newspaper"}
                    for year, count in sorted(year_counts_newspaper.items())
                ]
            )
            
            st.metric("Total Reprints", len(newspaper_years))
            st.metric("Year Span", f"{min(newspaper_years)} - {max(newspaper_years)}")
            st.metric("Unique Years", len(year_counts_newspaper))
        else:
            st.info("No reprints found in ECCO-Newspaper dataset.")
            df_newspaper = pd.DataFrame(columns=["year", "count", "dataset"])

    # Combined timeline visualization
    if not df_ecco.empty or not df_newspaper.empty:
        df_combined = pd.concat([df_ecco, df_newspaper], ignore_index=True)
        
        fig = px.line(
            df_combined,
            x="year",
            y="count",
            color="dataset",
            markers=True,
            labels={"year": "Year", "count": "Reprint Count", "dataset": "Dataset"},
            title=f"Reprint timeline comparison for {selected_essay['src_doc_id']} (Section {selected_essay['src_section_id']})",
        )
        fig.update_traces(marker=dict(size=8), line=dict(width=2))
        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No year information available for comparison.")


def main():
    st.set_page_config(page_title="Reuses of Hume – Network", layout="wide")
    st.title("Reuses of Hume – Network & Propagation")

    st.header("Network", divider="rainbow")
    network_tab1, network_tab2 = st.tabs(["ECCO-ECCO Network", "ECCO-Newspaper Network"])

    with network_tab1:
        ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
        ecco_blocks_data = load_blocks_data(ecco_blocks_path)
        render_network_view(ecco_blocks_data, "ecco")

    with network_tab2:
        newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
        newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
        render_network_view(newspaper_blocks_data, "newspaper")

    st.header("Target Essay Propagation Analysis", divider="rainbow")
    propagation_tab1, propagation_tab2 = st.tabs(["ECCO-ECCO Propagation", "ECCO-Newspaper Propagation"])

    with propagation_tab1:
        ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
        ecco_blocks_data = load_blocks_data(ecco_blocks_path)
        render_propagation_analysis(ecco_blocks_data, "ecco")

    with propagation_tab2:
        newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
        newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
        render_propagation_analysis(newspaper_blocks_data, "newspaper")

    st.header("Cross-Dataset Comparison", divider="rainbow")
    ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
    newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
    ecco_blocks_data = load_blocks_data(ecco_blocks_path)
    newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
    render_comparison_view(ecco_blocks_data, newspaper_blocks_data)


if __name__ == "__main__":
    main()

