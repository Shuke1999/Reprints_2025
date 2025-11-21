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


def _render_preview(url: str | None, mode: str, label: str, height: int = 420) -> None:
    if not url:
        st.info(f"{label} has no URL")
        return
    try:
        if mode == "Try displaying image":
            st.image(url, caption=label, use_container_width=True)
        elif mode == "Embed webpage":
            components.iframe(url, height=height)
        else:
            st.markdown(f"[Open link]({url})")
    except Exception as exc:
        st.warning(f"{label} preview failed: {exc} (link still available above)")


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


def _infer_collection_from_doc_id(doc_id: str | None) -> str:
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


def render_block_comparison(blocks_data, data_type: str):
    if blocks_data is None:
        st.warning("Data file is missing or could not be loaded.")
        return

    st.info(f"Loaded {len(blocks_data)} block pair records.")

    overlap_ratios = [pair.get("overlap_ratio", 0) for pair in blocks_data]
    min_block_lengths = [pair.get("min_block_length", 0) for pair in blocks_data]

    min_overlap = min(overlap_ratios) if overlap_ratios else 0
    max_overlap = max(overlap_ratios) if overlap_ratios else 1
    # Ensure max_overlap is greater than min_overlap
    if max_overlap <= min_overlap:
        max_overlap = min_overlap + 0.01
    
    min_length = min(min_block_lengths) if min_block_lengths else 0
    max_length = max(min_block_lengths) if min_block_lengths else 1000
    # Ensure max_length is greater than min_length
    if max_length <= min_length:
        max_length = min_length + 1

    col1, col2 = st.columns(2)
    with col1:
        overlap_min = st.slider(
            "Minimum overlap_ratio",
            min_value=float(min_overlap),
            max_value=float(max_overlap),
            value=float(min_overlap),
            step=0.01,
            key=f"overlap_min_{data_type}",
        )
    with col2:
        overlap_max = st.slider(
            "Maximum overlap_ratio",
            min_value=float(min_overlap),
            max_value=float(max_overlap),
            value=float(max_overlap),
            step=0.01,
            key=f"overlap_max_{data_type}",
        )

    col3, col4 = st.columns(2)
    with col3:
        length_min = st.slider(
            "Minimum min_block_length",
            min_value=int(min_length),
            max_value=int(max_length),
            value=int(min_length),
            step=1,
            key=f"length_min_{data_type}",
        )
    with col4:
        length_max = st.slider(
            "Maximum min_block_length",
            min_value=int(min_length),
            max_value=int(max_length),
            value=int(max_length),
            step=1,
            key=f"length_max_{data_type}",
        )

    filtered_data = [
        pair
        for pair in blocks_data
        if (
            overlap_min <= pair.get("overlap_ratio", 0) <= overlap_max
            and length_min <= pair.get("min_block_length", 0) <= length_max
        )
    ]

    st.success(f"{len(filtered_data)} records remain after filtering.")

    # Ensure src_section_header is available for each pair (target essay)
    _ensure_src_headers(filtered_data, data_type)

    essays_dict = {}
    for pair in filtered_data:
        src_doc_id = pair.get("src_doc_id")
        src_section_id = pair.get("src_section_id")
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
        key=f"essay_selector_{data_type}",
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
        block_a = pair.get("block_a", {})
        block_b = pair.get("block_b", {})
        if block_a.get("src_trs_start") is not None:
            all_src_starts.append(block_a.get("src_trs_start"))
        if block_a.get("src_trs_end") is not None:
            all_src_ends.append(block_a.get("src_trs_end"))
        if block_b.get("src_trs_start") is not None:
            all_src_starts.append(block_b.get("src_trs_start"))
        if block_b.get("src_trs_end") is not None:
            all_src_ends.append(block_b.get("src_trs_end"))

    if all_src_starts and all_src_ends:
        essay_merged_start = min(all_src_starts)
        essay_merged_end = max(all_src_ends)
        first_pair = selected_pairs[0]
        block_a = first_pair.get("block_a", {})
        block_b = first_pair.get("block_b", {})
        url_template = block_a.get("src_trs_url") or block_b.get("src_trs_url")
        if url_template:
            doc_id_match = re.search(r"docId=([^&]+)", url_template)
            if doc_id_match:
                doc_id = doc_id_match.group(1)
                essay_merged_url = (
                    f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}"
                    f"&offsetStart={essay_merged_start}&offsetEnd={essay_merged_end}"
                )
                st.markdown(f"**Target essay full URL:** [{essay_merged_url}]({essay_merged_url})")
                section_url_source = block_a.get("src_section_url") or block_b.get("src_section_url")
                if section_url_source:
                    urls = [u.strip() for u in section_url_source.split("\n") if u.strip()]
                    if urls:
                        st.markdown(f"**First page link:** [{urls[0]}]({urls[0]})")

    st.divider()

    preview_mode = st.radio(
        "URL preview mode",
        options=("Links only", "Embed webpage", "Try displaying image"),
        horizontal=True,
        key=f"preview_mode_{data_type}",
    )

    destination_entries: list[dict] = []

    for idx, pair in enumerate(selected_pairs):
        with st.expander(
            f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')} "
            f"(overlap: {pair.get('overlap_ratio', 0):.4f}, min_length: {pair.get('min_block_length', 0)})",
            expanded=False,
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
                if "block_a" in pair:
                    st.markdown("#### Block A")
                    block_a = pair["block_a"]
                    dst_a_pub_date = _get_dst_publication_date(block_a.get("dst_doc_id"), data_type)
                    destination_entries.append(
                        {
                            "dst_doc_id": block_a.get("dst_doc_id"),
                            "dst_publication_date": dst_a_pub_date,
                            "year": _get_dst_publication_year(block_a.get("dst_doc_id"), data_type)
                            or _extract_year_from_doc_id(block_a.get("dst_doc_id")),
                            "block_label": "Block A",
                            "pair_index": idx + 1,
                            "pair_summary": (
                                f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')}"
                            ),
                            "block": block_a,
                        }
                    )

                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_a.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_a.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_a.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_a.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_a.get('src_piece_length', 'N/A')}`")
                    if data_type == "newspaper" and block_a.get("src_section_id"):
                        st.markdown(f"- src_section_id: `{block_a.get('src_section_id', 'N/A')}`")
                    src_a_pub_date = _get_src_publication_date(block_a.get("src_doc_id"), data_type)
                    if src_a_pub_date:
                        st.markdown(f"- src_publication_date: `{src_a_pub_date}`")
                    if block_a.get("src_trs_url"):
                        st.markdown(f"- [src_trs_url]({block_a['src_trs_url']})")

                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_a.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_a.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_a.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_a.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_a.get('fragment_count', 'N/A')}`")
                    if dst_a_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_a_pub_date}`")
                    if block_a.get("dst_trs_url"):
                        st.markdown(f"- [dst_trs_url]({block_a['dst_trs_url']})")

            # Block B
            with col_block_b:
                if "block_b" in pair:
                    st.markdown("#### Block B")
                    block_b = pair["block_b"]
                    dst_b_pub_date = _get_dst_publication_date(block_b.get("dst_doc_id"), data_type)
                    destination_entries.append(
                        {
                            "dst_doc_id": block_b.get("dst_doc_id"),
                            "dst_publication_date": dst_b_pub_date,
                            "year": _get_dst_publication_year(block_b.get("dst_doc_id"), data_type)
                            or _extract_year_from_doc_id(block_b.get("dst_doc_id")),
                            "block_label": "Block B",
                            "pair_index": idx + 1,
                            "pair_summary": (
                                f"Pair {idx + 1}: {pair.get('dst_doc_id_a', 'N/A')} <-> {pair.get('dst_doc_id_b', 'N/A')}"
                            ),
                            "block": block_b,
                        }
                    )

                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_b.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_b.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_b.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_b.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_b.get('src_piece_length', 'N/A')}`")
                    if data_type == "newspaper" and block_b.get("src_section_id"):
                        st.markdown(f"- src_section_id: `{block_b.get('src_section_id', 'N/A')}`")
                    src_b_pub_date = _get_src_publication_date(block_b.get("src_doc_id"), data_type)
                    if src_b_pub_date:
                        st.markdown(f"- src_publication_date: `{src_b_pub_date}`")
                    if block_b.get("src_trs_url"):
                        st.markdown(f"- [src_trs_url]({block_b['src_trs_url']})")

                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_b.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_b.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_b.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_b.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_b.get('fragment_count', 'N/A')}`")
                    if dst_b_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_b_pub_date}`")
                    if block_b.get("dst_trs_url"):
                        st.markdown(f"- [dst_trs_url]({block_b['dst_trs_url']})")

            # Preview area
            if preview_mode != "Links only":
                st.divider()
                st.markdown("### Preview")
                
                # Target essay preview using merged range (src is the target essay)
                if "block_a" in pair or "block_b" in pair:
                    block_a = pair.get("block_a", {})
                    block_b = pair.get("block_b", {})
                    src_starts = []
                    src_ends = []
                    if block_a.get("src_trs_start") is not None:
                        src_starts.append(block_a.get("src_trs_start"))
                    if block_a.get("src_trs_end") is not None:
                        src_ends.append(block_a.get("src_trs_end"))
                    if block_b.get("src_trs_start") is not None:
                        src_starts.append(block_b.get("src_trs_start"))
                    if block_b.get("src_trs_end") is not None:
                        src_ends.append(block_b.get("src_trs_end"))
                    
                    if src_starts and src_ends:
                        merged_src_start = min(src_starts)
                        merged_src_end = max(src_ends)
                        url_template = block_a.get("src_trs_url") or block_b.get("src_trs_url")
                        if url_template:
                            doc_id_match = re.search(r"docId=([^&]+)", url_template)
                            if doc_id_match:
                                doc_id = doc_id_match.group(1)
                                merged_src_url = (
                                    f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}"
                                    f"&offsetStart={merged_src_start}&offsetEnd={merged_src_end}"
                                )
                                st.caption("Target Essay (merged range)")
                                _render_preview(merged_src_url, preview_mode, "Target Essay")
                
                # Source previews (for newspaper, show images)
                if data_type == "newspaper":
                    st.markdown("#### Source (Newspaper) Images")
                    prev_col_src1, prev_col_src2 = st.columns(2)
                    with prev_col_src1:
                        if "block_a" in pair:
                            block_a = pair["block_a"]
                            dst_doc_id = block_a.get("dst_doc_id")  # newspaper ID
                            src_section_id = block_a.get("src_section_id")
                            if dst_doc_id:
                                _render_newspaper_preview(
                                    dst_doc_id,
                                    src_section_id,
                                    preview_mode,
                                    "Block A Source (Newspaper)",
                                )
                    with prev_col_src2:
                        if "block_b" in pair:
                            block_b = pair["block_b"]
                            dst_doc_id = block_b.get("dst_doc_id")  # newspaper ID
                            src_section_id = block_b.get("src_section_id")
                            if dst_doc_id:
                                _render_newspaper_preview(
                                    dst_doc_id,
                                    src_section_id,
                                    preview_mode,
                                    "Block B Source (Newspaper)",
                                )

                # Destination previews
                st.markdown("#### Destination Previews")
                prev_col_dst1, prev_col_dst2 = st.columns(2)
                with prev_col_dst1:
                    if "block_a" in pair:
                        block_a = pair["block_a"]
                        st.caption("Destination (Block A)")
                        if block_a.get("dst_trs_url"):
                            _render_preview(
                                block_a.get("dst_trs_url"),
                                preview_mode,
                                "Block A Destination",
                            )
                with prev_col_dst2:
                    if "block_b" in pair:
                        block_b = pair["block_b"]
                        st.caption("Destination (Block B)")
                        if block_b.get("dst_trs_url"):
                            _render_preview(
                                block_b.get("dst_trs_url"),
                                preview_mode,
                                "Block B Destination",
                            )

    # --- Destination Propagation Timeline (for most reprinted target essay) ---
    st.divider()
    st.markdown("### Destination Propagation Timeline")
    
    # Count reprints per target essay
    essay_reprint_counts: dict[tuple, int] = {}
    for pair in filtered_data:
        essay_key = (pair.get("src_doc_id"), pair.get("src_section_id"))
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
        if (pair.get("src_doc_id"), pair.get("src_section_id")) == most_reprinted_essay_key:
            block_a = pair.get("block_a", {})
            block_b = pair.get("block_b", {})
            for block in [block_a, block_b]:
                if block.get("dst_doc_id"):
                    dst_year = _get_dst_publication_year(block.get("dst_doc_id"), data_type)
                    if dst_year is None:
                        dst_year = _extract_year_from_doc_id(block.get("dst_doc_id"))
                    if dst_year:
                        dst_pub_date = _get_dst_publication_date(block.get("dst_doc_id"), data_type)
                        timeline_entries.append({
                            "year": dst_year,
                            "dst_doc_id": block.get("dst_doc_id"),
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


def render_essay_search_page():
    """Search for a specific target essay and display detailed visualizations."""
    st.header("Target Essay Search", divider="rainbow")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_doc_id = st.text_input(
            "Target Essay Doc ID",
            value="",
            placeholder="e.g., 0036100901",
            key="essay_search_doc_id",
        )
    with col2:
        search_section_id = st.text_input(
            "Target Essay Section ID",
            value="",
            placeholder="e.g., 8",
            key="essay_search_section_id",
        )
    with col3:
        search_data_type = st.selectbox(
            "Dataset",
            options=("ecco", "newspaper"),
            key="essay_search_data_type",
        )
    
    if not search_doc_id or not search_section_id:
        st.info("Please enter both Doc ID and Section ID to search.")
        return
    
    try:
        search_section_id_int = int(search_section_id)
    except ValueError:
        st.error("Section ID must be a number.")
        return
    
    # Load data
    if search_data_type == "ecco":
        data_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
    else:
        data_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
    
    blocks_data = load_blocks_data(data_path)
    if not blocks_data:
        st.warning("Failed to load data.")
        return
    
    # Find matching pairs
    matching_pairs = []
    for pair in blocks_data:
        if (str(pair.get("src_doc_id")) == str(search_doc_id) and 
            pair.get("src_section_id") == search_section_id_int):
            matching_pairs.append(pair)
    
    if not matching_pairs:
        st.warning(f"No reprints found for target essay: Doc ID `{search_doc_id}`, Section ID `{search_section_id_int}`")
        return
    
    st.success(f"Found {len(matching_pairs)} reprint pairs for this target essay.")
    
    # Get essay metadata
    _ensure_src_headers(matching_pairs, search_data_type)
    src_section_header = _get_src_section_header(search_doc_id, search_section_id_int, search_data_type)
    src_publication_date = _get_src_publication_date(search_doc_id, search_data_type)
    src_publication_year = _get_src_publication_year(search_doc_id, search_data_type)
    
    st.info(f"**Target essay:** `{search_doc_id}` (Section `{search_section_id_int}`) - {len(matching_pairs)} reprint pairs")
    if src_section_header:
        st.caption(f"Section header: {src_section_header.strip()}")
    if src_publication_date:
        st.caption(f"Publication date: {src_publication_date}")
    elif src_publication_year:
        st.caption(f"Publication year: {src_publication_year}")
    
    # Preview mode selection
    preview_mode = st.radio(
        "URL preview mode",
        options=("Links only", "Embed webpage", "Try displaying image"),
        horizontal=True,
        key=f"essay_search_preview_mode_{search_data_type}",
    )
    
    # Pagination
    st.markdown("#### Reprint Pairs")
    total_pairs = len(matching_pairs)
    if total_pairs == 0:
        return
    
    # Ensure max_value is at least equal to min_value
    max_items = max(5, min(50, total_pairs))
    items_per_page = st.slider(
        "Items per page",
        min_value=5,
        max_value=max_items,
        value=min(10, total_pairs),
        step=5,
        key=f"essay_search_pagination_{search_data_type}",
    )
    
    total_pages = (total_pairs + items_per_page - 1) // items_per_page
    if total_pages > 1:
        page_num = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key=f"essay_search_page_{search_data_type}",
        )
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        pairs_to_show = matching_pairs[start_idx:end_idx]
        st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_pairs)} of {total_pairs} reprint pairs")
        pair_start_num = start_idx + 1
    else:
        pairs_to_show = matching_pairs
        pair_start_num = 1
    
    st.divider()
    
    # Display pairs with expanders
    for pair_idx, pair in enumerate(pairs_to_show, start=pair_start_num):
        block_a = pair.get("block_a", {})
        block_b = pair.get("block_b", {})
        dst_a = block_a.get("dst_doc_id", "N/A") if block_a else "N/A"
        dst_b = block_b.get("dst_doc_id", "N/A") if block_b else "N/A"
        overlap = pair.get("overlap_ratio", 0)
        
        with st.expander(
            f"Pair {pair_idx}: {dst_a} <-> {dst_b} (overlap: {overlap:.4f}, min_length: {pair.get('min_block_length', 0)})",
            expanded=False,
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
                if "block_a" in pair:
                    st.markdown("#### Block A")
                    block_a = pair["block_a"]
                    dst_a_pub_date = _get_dst_publication_date(block_a.get("dst_doc_id"), search_data_type)
                    
                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_a.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_a.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_a.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_a.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_a.get('src_piece_length', 'N/A')}`")
                    if search_data_type == "newspaper" and block_a.get("src_section_id"):
                        st.markdown(f"- src_section_id: `{block_a.get('src_section_id', 'N/A')}`")
                    src_a_pub_date = _get_src_publication_date(block_a.get("src_doc_id"), search_data_type)
                    if src_a_pub_date:
                        st.markdown(f"- src_publication_date: `{src_a_pub_date}`")
                    if block_a.get("src_trs_url"):
                        st.markdown(f"- [src_trs_url]({block_a['src_trs_url']})")
                    
                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_a.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_a.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_a.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_a.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_a.get('fragment_count', 'N/A')}`")
                    if dst_a_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_a_pub_date}`")
                    if block_a.get("dst_trs_url"):
                        st.markdown(f"- [dst_trs_url]({block_a['dst_trs_url']})")
            
            # Block B
            with col_block_b:
                if "block_b" in pair:
                    st.markdown("#### Block B")
                    block_b = pair["block_b"]
                    dst_b_pub_date = _get_dst_publication_date(block_b.get("dst_doc_id"), search_data_type)
                    
                    st.markdown("**Target Essay (Source):**")
                    st.markdown(f"- src_doc_id: `{block_b.get('src_doc_id', 'N/A')}`")
                    st.markdown(f"- src_trs_id: `{block_b.get('src_trs_id', 'N/A')}`")
                    st.markdown(f"- src_trs_start: `{block_b.get('src_trs_start', 'N/A')}`")
                    st.markdown(f"- src_trs_end: `{block_b.get('src_trs_end', 'N/A')}`")
                    st.markdown(f"- src_piece_length: `{block_b.get('src_piece_length', 'N/A')}`")
                    if search_data_type == "newspaper" and block_b.get("src_section_id"):
                        st.markdown(f"- src_section_id: `{block_b.get('src_section_id', 'N/A')}`")
                    src_b_pub_date = _get_src_publication_date(block_b.get("src_doc_id"), search_data_type)
                    if src_b_pub_date:
                        st.markdown(f"- src_publication_date: `{src_b_pub_date}`")
                    if block_b.get("src_trs_url"):
                        st.markdown(f"- [src_trs_url]({block_b['src_trs_url']})")
                    
                    st.markdown("**-> Destination:**")
                    st.markdown(f"- dst_doc_id: `{block_b.get('dst_doc_id', 'N/A')}`")
                    st.markdown(f"- dst_trs_start: `{block_b.get('dst_trs_start', 'N/A')}`")
                    st.markdown(f"- dst_trs_end: `{block_b.get('dst_trs_end', 'N/A')}`")
                    st.markdown(f"- dst_piece_length: `{block_b.get('dst_piece_length', 'N/A')}`")
                    st.markdown(f"- fragment_count: `{block_b.get('fragment_count', 'N/A')}`")
                    if dst_b_pub_date:
                        st.markdown(f"- dst_publication_date: `{dst_b_pub_date}`")
                    if block_b.get("dst_trs_url"):
                        st.markdown(f"- [dst_trs_url]({block_b['dst_trs_url']})")
            
            # Preview area
            if preview_mode != "Links only":
                st.divider()
                st.markdown("### Preview")
                
                # Target essay preview
                if "block_a" in pair or "block_b" in pair:
                    block_a = pair.get("block_a", {})
                    block_b = pair.get("block_b", {})
                    src_starts = []
                    src_ends = []
                    if block_a.get("src_trs_start") is not None:
                        src_starts.append(block_a.get("src_trs_start"))
                    if block_a.get("src_trs_end") is not None:
                        src_ends.append(block_a.get("src_trs_end"))
                    if block_b.get("src_trs_start") is not None:
                        src_starts.append(block_b.get("src_trs_start"))
                    if block_b.get("src_trs_end") is not None:
                        src_ends.append(block_b.get("src_trs_end"))
                    
                    if src_starts and src_ends:
                        merged_src_start = min(src_starts)
                        merged_src_end = max(src_ends)
                        url_template = block_a.get("src_trs_url") or block_b.get("src_trs_url")
                        if url_template:
                            doc_id_match = re.search(r"docId=([^&]+)", url_template)
                            if doc_id_match:
                                doc_id = doc_id_match.group(1)
                                merged_src_url = (
                                    f"https://onko-sivu.2.rahtiapp.fi/ecco?docId={doc_id}"
                                    f"&offsetStart={merged_src_start}&offsetEnd={merged_src_end}"
                                )
                                st.caption("Target Essay (merged range)")
                                _render_preview(merged_src_url, preview_mode, "Target Essay")
                
                # Newspaper images (for newspaper data type)
                if search_data_type == "newspaper":
                    st.markdown("#### Source (Newspaper) Images")
                    prev_col_newspaper1, prev_col_newspaper2 = st.columns(2)
                    with prev_col_newspaper1:
                        if "block_a" in pair:
                            block_a = pair["block_a"]
                            dst_doc_id = block_a.get("dst_doc_id")
                            src_section_id = block_a.get("src_section_id")
                            if dst_doc_id:
                                _render_newspaper_preview(
                                    dst_doc_id,
                                    src_section_id,
                                    preview_mode,
                                    "Block A Source (Newspaper)",
                                )
                    with prev_col_newspaper2:
                        if "block_b" in pair:
                            block_b = pair["block_b"]
                            dst_doc_id = block_b.get("dst_doc_id")
                            src_section_id = block_b.get("src_section_id")
                            if dst_doc_id:
                                _render_newspaper_preview(
                                    dst_doc_id,
                                    src_section_id,
                                    preview_mode,
                                    "Block B Source (Newspaper)",
                                )
                
                # Destination previews
                st.markdown("#### Destination Previews")
                prev_col_dst1, prev_col_dst2 = st.columns(2)
                with prev_col_dst1:
                    if "block_a" in pair:
                        block_a = pair["block_a"]
                        st.caption("Destination (Block A)")
                        if block_a.get("dst_trs_url"):
                            _render_preview(
                                block_a.get("dst_trs_url"),
                                preview_mode,
                                "Block A Destination",
                            )
                with prev_col_dst2:
                    if "block_b" in pair:
                        block_b = pair["block_b"]
                        st.caption("Destination (Block B)")
                        if block_b.get("dst_trs_url"):
                            _render_preview(
                                block_b.get("dst_trs_url"),
                                preview_mode,
                                "Block B Destination",
                            )


def render_block_pairs_page():
    st.header("Block Pairs Comparison", divider="rainbow")
    blocks_tab1, blocks_tab2 = st.tabs(["ECCO-ECCO Blocks", "ECCO-Newspaper Blocks"])

    with blocks_tab1:
        ecco_blocks_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
        ecco_blocks_data = load_blocks_data(ecco_blocks_path)
        render_block_comparison(ecco_blocks_data, "ecco")

    with blocks_tab2:
        newspaper_blocks_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
        newspaper_blocks_data = load_blocks_data(newspaper_blocks_path)
        render_block_comparison(newspaper_blocks_data, "newspaper")


def render_issue_tracking_page():
    """Track and export problematic entries."""
    st.header("Issue Tracking & Export", divider="rainbow")
    
    # Initialize session state for marked issues
    if "marked_issues" not in st.session_state:
        st.session_state.marked_issues = []
    
    # Search interface
    st.markdown("#### Search Entry")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_doc_id = st.text_input(
            "Doc ID (src_doc_id or dst_doc_id)",
            value="",
            placeholder="e.g., 0036100901 or WO2_B0420...",
            key="issue_search_doc_id",
        )
    with col2:
        search_data_type = st.selectbox(
            "Dataset",
            options=("ecco", "newspaper"),
            key="issue_search_data_type",
        )
    with col3:
        search_button = st.button("Search", key="issue_search_button")
    
    # Load data
    if search_data_type == "ecco":
        data_path = DERIVED_ECCO_DIR / "all_reprint_pairs_enriched.json"
    else:
        data_path = DERIVED_NEWSPAPER_DIR / "all_reprint_pairs_enriched.json"
    
    blocks_data = load_blocks_data(data_path)
    if not blocks_data:
        st.warning("Failed to load data.")
        return
    
    # Search for matching pairs
    matching_pairs = []
    if search_button and search_doc_id:
        for idx, pair in enumerate(blocks_data):
            # Check src_doc_id
            if str(pair.get("src_doc_id", "")) == str(search_doc_id):
                matching_pairs.append((idx, pair))
            # Check dst_doc_id in blocks
            block_a = pair.get("block_a", {})
            block_b = pair.get("block_b", {})
            if block_a and str(block_a.get("dst_doc_id", "")) == str(search_doc_id):
                matching_pairs.append((idx, pair))
            if block_b and str(block_b.get("dst_doc_id", "")) == str(search_doc_id):
                matching_pairs.append((idx, pair))
    
    # Display search results
    if matching_pairs:
        st.success(f"Found {len(matching_pairs)} matching entries.")
        
        for pair_idx, (data_idx, pair) in enumerate(matching_pairs):
            # Create unique identifier for this pair
            pair_id = f"{search_data_type}_{data_idx}"
            
            # Check if already marked
            is_marked = any(issue.get("id") == pair_id for issue in st.session_state.marked_issues)
            
            col1, col2 = st.columns([10, 1])
            with col1:
                block_a = pair.get("block_a", {})
                block_b = pair.get("block_b", {})
                dst_a = block_a.get("dst_doc_id", "N/A") if block_a else "N/A"
                dst_b = block_b.get("dst_doc_id", "N/A") if block_b else "N/A"
                overlap = pair.get("overlap_ratio", 0)
                
                st.markdown(
                    f"**Entry {pair_idx + 1}:** "
                    f"src_doc_id: `{pair.get('src_doc_id', 'N/A')}`, "
                    f"src_section_id: `{pair.get('src_section_id', 'N/A')}`, "
                    f"dst: {dst_a} <-> {dst_b}, "
                    f"overlap: {overlap:.4f}"
                )
            with col2:
                if st.checkbox(
                    "Mark as issue",
                    value=is_marked,
                    key=f"issue_checkbox_{pair_id}",
                ):
                    # Add to marked issues if not already there
                    if not is_marked:
                        issue_entry = {
                            "id": pair_id,
                            "data_type": search_data_type,
                            "data_index": data_idx,
                            "src_doc_id": pair.get("src_doc_id"),
                            "src_section_id": pair.get("src_section_id"),
                            "pair": pair,
                            "marked_at": str(pd.Timestamp.now()),
                        }
                        st.session_state.marked_issues.append(issue_entry)
                        st.rerun()
                else:
                    # Remove from marked issues if it was marked
                    if is_marked:
                        st.session_state.marked_issues = [
                            issue for issue in st.session_state.marked_issues
                            if issue.get("id") != pair_id
                        ]
                        st.rerun()
    elif search_button:
        st.info("No matching entries found.")
    
    st.divider()
    
    # Display marked issues
    st.markdown("#### Marked Issues")
    if st.session_state.marked_issues:
        st.info(f"Total marked issues: {len(st.session_state.marked_issues)}")
        
        # Show list of marked issues
        for idx, issue in enumerate(st.session_state.marked_issues):
            col1, col2 = st.columns([10, 1])
            with col1:
                pair = issue.get("pair", {})
                block_a = pair.get("block_a", {})
                block_b = pair.get("block_b", {})
                dst_a = block_a.get("dst_doc_id", "N/A") if block_a else "N/A"
                dst_b = block_b.get("dst_doc_id", "N/A") if block_b else "N/A"
                
                st.markdown(
                    f"{idx + 1}. **{issue.get('data_type', 'unknown')}** - "
                    f"src_doc_id: `{issue.get('src_doc_id', 'N/A')}`, "
                    f"src_section_id: `{issue.get('src_section_id', 'N/A')}`, "
                    f"dst: {dst_a} <-> {dst_b}"
                )
            with col2:
                if st.button("Remove", key=f"remove_issue_{issue.get('id')}"):
                    st.session_state.marked_issues = [
                        i for i in st.session_state.marked_issues
                        if i.get("id") != issue.get("id")
                    ]
                    st.rerun()
        
        st.divider()
        
        # Export functionality
        st.markdown("#### Export Marked Issues")
        col1, col2 = st.columns(2)
        
        with col1:
            export_filename = st.text_input(
                "Export filename",
                value="marked_issues.json",
                key="export_filename",
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("Export to JSON", key="export_button"):
                # Prepare export data
                export_data = {
                    "exported_at": str(pd.Timestamp.now()),
                    "total_issues": len(st.session_state.marked_issues),
                    "issues": st.session_state.marked_issues,
                }
                
                # Convert to JSON string
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                # Create download button
                st.download_button(
                    label="Download JSON file",
                    data=json_str,
                    file_name=export_filename,
                    mime="application/json",
                    key="download_json",
                )
                
                st.success(f"Ready to download {len(st.session_state.marked_issues)} marked issues.")
        
        # Clear all button
        if st.button("Clear All Marked Issues", key="clear_all_issues"):
            st.session_state.marked_issues = []
            st.rerun()
    else:
        st.info("No issues marked yet. Use the search above to find and mark problematic entries.")


def main():
    st.set_page_config(page_title="Reuses of Hume – Block Pairs", layout="wide")
    st.title("Reuses of Hume – Block Pairs Exploration")
    
    page_tabs = st.tabs(["Block Pairs Comparison", "Target Essay Search", "Issue Tracking"])
    
    with page_tabs[0]:
        render_block_pairs_page()
    
    with page_tabs[1]:
        render_essay_search_page()
    
    with page_tabs[2]:
        render_issue_tracking_page()


if __name__ == "__main__":
    main()

