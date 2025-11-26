import gzip
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


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

DATASETS = {
    "ECCO → ECCO": DERIVED_ECCO_DIR / "hume_outgoing_ecco-ecco_original_only_merged_with_urls.json",
    "ECCO → Newspaper": DERIVED_NEWSPAPER_DIR / "hume_outgoing_ecco-newspaper_original_only_merged_with_urls.json",
}

NEWSPAPER_ID_MAPPING_PATH = DERIVED_NEWSPAPER_DIR / "newspaper_id_mapping.json"

_NEWSPAPER_ARTICLE_MAPPING_CACHE: dict[str, dict[str, str]] | None = None


# ---------------------------------------------------------------------------
# Newspaper preview helpers (reuse the logic from existing apps)
# ---------------------------------------------------------------------------


def _load_newspaper_article_mapping() -> dict[str, dict[str, str]]:
    """Load newspaper article mapping (articleID -> metadata)."""
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
                page_asset_id = page.get("pageAssetID")
                articles = page.get("articles", [])
                for article in articles:
                    article_id = article.get("articleID")
                    article_asset_id = article.get("articleAssetID")
                    if article_id and article_asset_id:
                        _NEWSPAPER_ARTICLE_MAPPING_CACHE[article_id] = {
                            "articleAssetID": article_asset_id,
                            "articleType": article.get("articleType") or "Unknown",
                            "pageAssetID": page_asset_id,
                            "docId": article.get("docId"),
                        }
    except Exception as exc:
        st.warning(f"Failed to load newspaper article mapping: {exc}")
        _NEWSPAPER_ARTICLE_MAPPING_CACHE = {}

    return _NEWSPAPER_ARTICLE_MAPPING_CACHE


def _normalize_doc_id(raw_doc_id: str | None, asset_id: str | None, collection: str) -> str | None:
    if raw_doc_id:
        return raw_doc_id.split("|", 1)[-1]
    if not asset_id:
        return None
    prefix = "Z" if collection == "burney" else "N" if collection == "nichols" else ""
    if asset_id.upper().startswith(("Z", "N")):
        return asset_id
    return f"{prefix}{asset_id}" if prefix else asset_id


def _find_image_in_toc(nodes: list | None, target_doc_id: str | None) -> str | None:
    if not nodes or not target_doc_id:
        return None
    for node in nodes:
        node_doc_id = node.get("docId")
        if node_doc_id:
            normalized = node_doc_id.split("|", 1)[-1]
            if normalized == target_doc_id and node.get("image"):
                return node["image"]
        nested = _find_image_in_toc(node.get("subArticleDocuments"), target_doc_id)
        if nested:
            return nested
    return None


def _build_image_url(record_id: str | None) -> str | None:
    if not record_id:
        return None
    separator = "&" if "?" in record_id else "?"
    return f"https://luna.gale.com/imgsrv/FastFetch/UBER2/{record_id}{separator}format=jpeg"


def _get_newspaper_image_urls(dst_doc_id: str, src_section_id: str | None = None) -> list[str]:
    """Get newspaper image URLs by requesting Gale page and parsing dviResponse."""
    if not dst_doc_id:
        return []

    mapping = _load_newspaper_article_mapping()
    article_info = mapping.get(dst_doc_id)
    if not article_info:
        st.warning(f"Could not find article metadata for articleID: {dst_doc_id}")
        return []

    article_asset_id = article_info.get("articleAssetID")
    page_asset_id = article_info.get("pageAssetID")
    doc_id_from_mapping = article_info.get("docId")

    if dst_doc_id.upper().startswith("W"):
        collection = "burney"
    elif dst_doc_id.upper().startswith("N"):
        collection = "nichols"
    else:
        collection = "nichols"

    config = {"burney": {"prodId": "BBCN"}, "nichols": {"prodId": "NICN"}}.get(collection, {"prodId": "NICN"})
    prod_id = config["prodId"]

    gale_doc_id = _normalize_doc_id(doc_id_from_mapping, article_asset_id, collection)
    if not gale_doc_id and page_asset_id:
        gale_doc_id = _normalize_doc_id(None, page_asset_id, collection)
    if not gale_doc_id:
        st.warning(f"Unable to derive Gale docId for articleID: {dst_doc_id}")
        return []

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

    try:
        response = requests.get(target, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if "gzip" in content_encoding:
            try:
                html = gzip.decompress(response.content).decode("utf-8")
            except Exception:
                response.encoding = response.encoding or "utf-8"
                html = response.text
        else:
            response.encoding = response.encoding or "utf-8"
            html = response.text
    except requests.RequestException as exc:
        st.error(f"Failed to fetch HTML from Gale: {exc}")
        return []

    patterns = [
        r"var\s+dviResponse\s*=\s*(\{[\s\S]*?\});",
        r"dviResponse\s*=\s*(\{[\s\S]*?\});",
        r"window\.dviResponse\s*=\s*(\{[\s\S]*?\});",
        r'"dviResponse"\s*:\s*(\{[\s\S]*?\})',
    ]

    match = None
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            break

    if not match:
        st.warning(f"dviResponse object not found for articleID: {dst_doc_id}")
        return []

    obj_text = match.group(1)
    sanitized = re.sub(r",\s*}", "}", obj_text)
    sanitized = re.sub(r",\s*]", "]", sanitized)

    try:
        dvi_response = json.loads(sanitized.replace("'", '"'))
    except Exception:
        try:
            dvi_response = eval(sanitized)  # noqa: S307
        except Exception:
            st.warning(f"Failed to parse dviResponse for articleID: {dst_doc_id}")
            return []

    if not isinstance(dvi_response.get("pageDocuments"), list):
        st.warning(f"dviResponse.pageDocuments missing for articleID: {dst_doc_id}")
        return []

    toc = dvi_response.get("articleTableOfContents", [])
    image_id = _find_image_in_toc(toc, gale_doc_id)
    if image_id:
        url = _build_image_url(image_id)
        if url:
            return [url]

    image_list = dvi_response.get("imageList", [])
    if isinstance(image_list, list):
        urls: list[str] = []
        for image in image_list:
            if not image.get("currentArticle"):
                continue
            url = _build_image_url(image.get("recordId"))
            if url:
                urls.append(url)
        if urls:
            return urls
    return []


def _render_preview(url: str | None, mode: str, label: str, height: int = 420) -> None:
    if not url:
        st.info(f"{label} has no URL")
        return
    if mode == "Links only":
        st.markdown(f"[Open link]({url})")
        return
    is_onko = "onko-sivu" in url
    try:
        if mode == "Try displaying image" and not is_onko:
            st.image(url, caption=label, use_container_width=True)
        else:
            # Either user explicitly selected iframe mode or ECCO (onko) pages
            # which only render correctly via embedding.
            components.iframe(url, height=height)
    except Exception as exc:
        st.warning(f"{label} preview failed: {exc} (link still available above)")


def _render_newspaper_preview(dst_doc_id: str | None, src_section_id: str | None, mode: str, label: str) -> None:
    if not dst_doc_id:
        st.info(f"{label} has no dst_doc_id")
        return
    st.info(f"Article ID: `{dst_doc_id}` | src_section_id: `{src_section_id}`")
    if mode == "Links only":
        return
    with st.spinner(f"Fetching images for {label}..."):
        try:
            image_urls = _get_newspaper_image_urls(dst_doc_id, src_section_id)
        except Exception as exc:
            st.error(f"Failed to fetch images for {label}: {exc}")
            return
    if image_urls:
        st.success(f"Fetched {len(image_urls)} image(s)")
        for idx, img_url in enumerate(image_urls, start=1):
            try:
                st.image(img_url, caption=f"{label} - Image {idx}", use_container_width=True)
            except Exception as exc:
                st.warning(f"Failed to display image {idx}: {exc}")
                st.markdown(f"[Open image URL]({img_url})")
    else:
        st.warning(f"No images found for {label}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=len(DATASETS))
def _load_dataset(name: str) -> list[dict[str, Any]]:
    """Load a single JSON dataset into memory."""
    path = DATASETS[name]
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    for record in records:
        record["dataset"] = name
    return records


@st.cache_data(show_spinner=True)
def load_all_records() -> list[dict[str, Any]]:
    """Load and merge both datasets (cached by Streamlit)."""
    all_records: list[dict[str, Any]] = []
    for name in DATASETS:
        all_records.extend(_load_dataset(name))
    return all_records


@st.cache_data(show_spinner=False)
def build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert the records into a tidy dataframe for filtering & viz."""
    rows = []
    for record in records:
        ratio = record.get("reuse_ratio")
        ratio = float(ratio) if ratio not in (None, "") else None
        rows.append(
            {
                "dataset": record.get("dataset"),
                "reuse_ratio": ratio,
                "src_doc_id": record.get("src_doc_id"),
                "src_section_id": record.get("src_section_id"),
                "src_section_header": record.get("src_section_header"),
                "src_publication_date": record.get("src_publication_date"),
                "src_section_url": record.get("src_section_url"),
                "src_trs_url": record.get("src_trs_url"),
                "dst_doc_id": record.get("dst_doc_id"),
                "dst_publication_date": record.get("dst_publication_date"),
                "dst_title": record.get("dst_title"),
                "dst_section_url": record.get("dst_section_url"),
                "dst_trs_url": record.get("dst_trs_url"),
                "fragment_count": record.get("fragment_count"),
                "src_text": record.get("src_text"),
                "dst_text": record.get("dst_text"),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["src_doc_id", "src_section_id", "reuse_ratio"], ascending=[True, True, False]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_target_summary(df: pd.DataFrame, dataset_names: list[str]) -> pd.DataFrame:
    """Aggregate counts per target essay."""
    if df.empty:
        return pd.DataFrame()

    base = (
        df.groupby(["src_doc_id", "src_section_id", "src_section_header"], dropna=False)
        .agg(
            total_blocks=("dataset", "size"),
            min_ratio=("reuse_ratio", "min"),
            max_ratio=("reuse_ratio", "max"),
            avg_ratio=("reuse_ratio", "mean"),
        )
        .reset_index()
    )

    dataset_counts = (
        df.groupby(["src_doc_id", "src_section_id", "src_section_header", "dataset"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot_table(
            index=["src_doc_id", "src_section_id", "src_section_header"],
            columns="dataset",
            values="count",
            fill_value=0,
        )
        .reset_index()
    )

    summary = base.merge(dataset_counts, on=["src_doc_id", "src_section_id", "src_section_header"], how="left")
    for dataset in dataset_names:
        if dataset not in summary.columns:
            summary[dataset] = 0

    summary["min_ratio"] = summary["min_ratio"].round(4)
    summary["max_ratio"] = summary["max_ratio"].round(4)
    summary["avg_ratio"] = summary["avg_ratio"].round(4)
    summary["essay_label"] = summary.apply(
        lambda row: f"{row['src_doc_id']} · sec {row['src_section_id']} · {row.get('src_section_header') or 'Untitled'}",
        axis=1,
    )
    return summary.sort_values("total_blocks", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _normalize_multiline_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [line.strip() for line in str(value).splitlines() if line.strip()]


def _render_block_pair(record: dict[str, Any], idx: int, preview_mode: str) -> None:
    """Render a single block pair with src (Hume) on the left plus media previews."""
    ratio = record.get("reuse_ratio")
    ratio_label = f"{ratio:.4f}" if ratio is not None else "N/A"
    st.markdown(f"#### Pair {idx + 1}: {record.get('dataset')} (reuse_ratio={ratio_label})")
    cols = st.columns(2, gap="large")

    with cols[0]:
        st.caption("Hume source (src)")
        header = record.get("src_section_header") or "No section header"
        st.markdown(f"**Section**: {header}")
        st.markdown(
            f"- `src_doc_id`: `{record.get('src_doc_id')}`\n"
            f"- `src_section_id`: `{record.get('src_section_id')}`\n"
            f"- Publication date: {record.get('src_publication_date') or 'Unknown'}"
        )
        st.code(record.get("src_text") or "No source text available", language="markdown")

    with cols[1]:
        st.caption("Destination (dst)")
        st.markdown(f"**Title**: {record.get('dst_title') or 'Unknown'}")
        st.markdown(
            f"- `dst_doc_id`: `{record.get('dst_doc_id')}`\n"
            f"- Publication date: {record.get('dst_publication_date') or 'Unknown'}\n"
            f"- Fragments in block: {record.get('fragment_count') or 0}"
        )
        st.code(record.get("dst_text") or "No destination text available", language="markdown")

    _render_block_media(record, preview_mode)


def _render_block_media(record: dict[str, Any], preview_mode: str) -> None:
    dataset = record.get("dataset")
    with st.expander("Images & links", expanded=False):
        st.caption("Side-by-side visualization of source and destination")
        col_src, col_dst = st.columns(2)

        # Left: Hume source (ECCO)
        with col_src:
            st.caption("Hume source (src)")
            _render_preview(record.get("src_trs_url"), preview_mode, "Hume source")

        # Right: destination side
        with col_dst:
            if dataset == "ECCO → ECCO":
                st.caption("ECCO reprint (dst)")
                _render_preview(record.get("dst_trs_url"), preview_mode, "ECCO reprint")
            else:
                st.caption("Newspaper/ECCO destination (dst)")
                # Newspaper images (if Newspaper)
                _render_newspaper_preview(
                    record.get("dst_doc_id"),
                    str(record.get("src_section_id")),
                    preview_mode,
                    "Newspaper destination",
                )
                # 若有 ECCO dst_transcript 也一起嵌入
                if record.get("dst_trs_url"):
                    _render_preview(record.get("dst_trs_url"), preview_mode, "Destination transcript")


def _render_visualization(df: pd.DataFrame) -> None:
    """Render reuse_ratio distribution chart using Plotly."""
    if df.empty:
        st.info("No data available for visualization with the current filters.")
        return
    fig = px.histogram(
        df,
        x="reuse_ratio",
        color="dataset",
        nbins=40,
        barmode="overlay",
        opacity=0.6,
        title="Reuse ratio distribution",
        labels={"reuse_ratio": "reuse_ratio (src_piece_length / section span)"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Hume reuse ratios", layout="wide")
st.title("Hume Reuse Explorer (reuse_ratio focused)")
st.caption(
    "Data sources: `data/data_2011/derived-ecco/hume_outgoing_ecco-ecco_original_only_merged_with_urls.json` "
    "and `data/data_2011/derived-newspaper/hume_outgoing_ecco-newspaper_original_only_merged_with_urls.json`."
)

records = load_all_records()
df = build_dataframe(records)
available_datasets = sorted(df["dataset"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    dataset_selection = st.multiselect(
        "Datasets",
        options=available_datasets,
        default=available_datasets,
    )
    ratio_min = float(df["reuse_ratio"].min(skipna=True)) if not df.empty else 0.0
    ratio_max = float(df["reuse_ratio"].max(skipna=True)) if not df.empty else 1.0
    ratio_range = st.slider(
        "reuse_ratio range",
        min_value=0.0,
        max_value=max(1.0, ratio_max),
        value=(ratio_min, ratio_max),
        step=0.001,
        help="Filter blocks whose reuse_ratio (src_piece_length / section span) falls within this range.",
    )

filtered_df = df[df["dataset"].isin(dataset_selection)]
filtered_df = filtered_df[
    (filtered_df["reuse_ratio"].notna())
    & (filtered_df["reuse_ratio"] >= ratio_range[0])
    & (filtered_df["reuse_ratio"] <= ratio_range[1])
]

st.subheader("Overview")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Blocks matching filters", len(filtered_df))
with col_b:
    st.metric("Datasets selected", len(dataset_selection))
with col_c:
    st.metric("reuse_ratio range", f"{ratio_range[0]:.3f} – {ratio_range[1]:.3f}")

_render_visualization(filtered_df)

st.subheader("Target essay overview")
summary_df = build_target_summary(filtered_df, available_datasets)
if summary_df.empty:
    st.warning("No target essays match the current filters.")
    st.stop()

summary_display = summary_df[
    [
        "essay_label",
        "total_blocks",
        "min_ratio",
        "max_ratio",
        "avg_ratio",
        *available_datasets,
    ]
]
st.dataframe(summary_display, use_container_width=True, hide_index=True)

selected_label = st.selectbox(
    "Select a target essay to inspect",
    options=summary_df["essay_label"].tolist(),
)

selected_row = summary_df[summary_df["essay_label"] == selected_label].iloc[0]
target_mask = (
    (filtered_df["src_doc_id"] == selected_row["src_doc_id"])
    & (filtered_df["src_section_id"] == selected_row["src_section_id"])
)
target_df = filtered_df[target_mask].sort_values("reuse_ratio", ascending=False)

st.subheader("Block pairs (for selected target essay)")
st.caption(
    f"{selected_label} · {len(target_df)} blocks | "
    f"reuse_ratio range {selected_row['min_ratio']:.4f} – {selected_row['max_ratio']:.4f}"
)

preview_mode = st.radio(
    "Preview mode",
    options=["Links only", "Try displaying image", "Embed webpage"],
    index=1,
    horizontal=True,
)

for idx, row in target_df.reset_index(drop=True).iterrows():
    _render_block_pair(row.to_dict(), idx, preview_mode)
    st.divider()

