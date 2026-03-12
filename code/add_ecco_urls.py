"""
Add src_url and dst_url to all JSON data under data/ that contain offset fields.
URL format: https://onko-sivu.2.rahtiapp.fi/ecco?docId=...&offsetStart=...&offsetEnd=...&isAlreadyOctavified=0

Supported offset field names:
  - src: src_trs_start/end, src_offset_start/end, src_start_offset/src_end_offset,
         predicted_src_span [start,end], src_range "start-end"
  - dst: dst_trs_start/end, dst_offset_start/end, dst_start_offset/dst_end_offset,
         predicted_dst_span [start,end], dst_range "start-end"

Usage:
  python code/add_ecco_urls.py [data_dir]        # default data_dir = repo/data, writes in place
  python code/add_ecco_urls.py data --dry-run    # only report what would be updated
"""
import argparse
import json
import sys
from pathlib import Path

ECCO_BASE = "https://onko-sivu.2.rahtiapp.fi/ecco"


def build_url(doc_id: str, start: int, end: int) -> str:
    return f"{ECCO_BASE}?docId={doc_id}&offsetStart={start}&offsetEnd={end}&isAlreadyOctavified=0"


def get_src_params(obj: dict) -> tuple[str, int, int] | None:
    """Return (doc_id, start, end) for source if all present."""
    doc_id = obj.get("src_doc_id")
    if not doc_id:
        return None
    start, end = None, None
    if "predicted_src_span" in obj:
        span = obj["predicted_src_span"]
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = int(span[0]), int(span[1])
    if start is None and "src_trs_start" in obj and "src_trs_end" in obj:
        start, end = int(obj["src_trs_start"]), int(obj["src_trs_end"])
    if start is None and "src_offset_start" in obj and "src_offset_end" in obj:
        start, end = int(obj["src_offset_start"]), int(obj["src_offset_end"])
    if start is None and "src_start_offset" in obj and "src_end_offset" in obj:
        start, end = int(obj["src_start_offset"]), int(obj["src_end_offset"])
    if start is None and "src_range" in obj:
        parts = str(obj["src_range"]).split("-")
        if len(parts) >= 2:
            try:
                start, end = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                pass
    if doc_id and start is not None and end is not None:
        return (str(doc_id), start, end)
    return None


def get_dst_params(obj: dict) -> tuple[str, int, int] | None:
    """Return (doc_id, start, end) for destination if all present."""
    doc_id = obj.get("dst_doc_id")
    if not doc_id:
        return None
    start, end = None, None
    if "predicted_dst_span" in obj:
        span = obj["predicted_dst_span"]
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = int(span[0]), int(span[1])
    if start is None and "dst_trs_start" in obj and "dst_trs_end" in obj:
        start, end = int(obj["dst_trs_start"]), int(obj["dst_trs_end"])
    if start is None and "dst_offset_start" in obj and "dst_offset_end" in obj:
        start, end = int(obj["dst_offset_start"]), int(obj["dst_offset_end"])
    if start is None and "dst_start_offset" in obj and "dst_end_offset" in obj:
        start, end = int(obj["dst_start_offset"]), int(obj["dst_end_offset"])
    if start is None and "dst_range" in obj:
        parts = str(obj["dst_range"]).split("-")
        if len(parts) >= 2:
            try:
                start, end = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                pass
    if doc_id and start is not None and end is not None:
        return (str(doc_id), start, end)
    return None


def add_urls_to_obj(obj, stats: dict) -> None:
    """Mutate obj in place: add src_url/dst_url where offset fields exist. Count in stats."""
    if isinstance(obj, list):
        for item in obj:
            add_urls_to_obj(item, stats)
        return
    if isinstance(obj, dict):
        src_params = get_src_params(obj)
        dst_params = get_dst_params(obj)
        if src_params:
            obj["src_url"] = build_url(*src_params)
            stats["src_added"] = stats.get("src_added", 0) + 1
        if dst_params:
            obj["dst_url"] = build_url(*dst_params)
            stats["dst_added"] = stats.get("dst_added", 0) + 1
        if src_params and dst_params:
            stats["pairs_updated"] = stats.get("pairs_updated", 0) + 1
        for v in obj.values():
            add_urls_to_obj(v, stats)
        return
    # str, int, bool, etc. – skip
    return


def process_file(path: Path, dry_run: bool) -> dict:
    """Load JSON, add URLs, optionally write back. Return stats for this file."""
    stats = {"file": str(path), "pairs_updated": 0, "src_added": 0, "dst_added": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        stats["error"] = str(e)
        return stats
    add_urls_to_obj(data, stats)
    if not dry_run and (stats.get("pairs_updated") or stats.get("src_added") or stats.get("dst_added")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Add src_url and dst_url to JSON data under data/")
    parser.add_argument("data_dir", nargs="?", default=None, help="Data directory (default: repo/data)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data")
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)
    json_files = [p for p in data_dir.rglob("*.json") if p.name != "reprint_detection_bucket5000.json"]
    print(f"Found {len(json_files)} JSON files under {data_dir} (excluding reprint_detection_bucket5000.json)")
    for path in sorted(json_files):
        s = process_file(path, args.dry_run)
        if s.get("error"):
            print(f"  ERROR {path}: {s['error']}")
        elif s.get("pairs_updated") or s.get("src_added") or s.get("dst_added"):
            print(f"  {path.relative_to(data_dir)}: pairs_updated={s.get('pairs_updated', 0)} src_added={s.get('src_added', 0)} dst_added={s.get('dst_added', 0)}")
    print("Done.")


if __name__ == "__main__":
    main()
