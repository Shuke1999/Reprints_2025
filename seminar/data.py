"""
Shared paths, constants, and static data for the seminar presentation.
Uses REPRINTS_REPO_ROOT / REPRINTS_DATA_DIR if set; otherwise resolves from this file.
"""
import os
from pathlib import Path


def _resolve_repo_root() -> Path:
    # seminar/data.py -> parents[1] = repo root (directory containing seminar/)
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

# Paths from plan §10 (relative to DATA_DIR or REPO_ROOT; data lives under repo/data)
PATH_GT_ORIGIN = DATA_DIR / "gt_offset_origin.json"
PATH_HUME_GT_LIST = DATA_DIR / "hume_gt_list.txt"
PATH_TOPK_BY_FEATURE = DATA_DIR / "topk_by_feature.json"
PATH_REPRINT_DETECTION = DATA_DIR / "reprint_detection_bucket5000.json"
DIR_GT_SPLITS = DATA_DIR / "data0309" / "gt_splits"
DIR_CONTEXT_200 = DATA_DIR / "data0309" / "context_200"
DIR_CONTEXT_500 = DATA_DIR / "data0309" / "context_500"
DIR_CONTEXT_1000 = DATA_DIR / "data0309" / "context_1000"
PATH_SIGNALS_200 = DATA_DIR / "data0309" / "signals_discovery_200.jsonl"
PATH_SIGNALS_500 = DATA_DIR / "data0309" / "signals_discovery_500.jsonl"
PATH_SIGNALS_1000 = DATA_DIR / "data0309" / "signals_discovery_1000.jsonl"

# --- Ground truth & overview stats ---
GT_POSITIVE_PAIRS = 176
GT_NEGATIVE_PAIRS = 961
GT_ATTRIBUTED_REPRINT = 130
GT_UNATTRIBUTED_REPRINT = 46
GT_QUOTED = 746
GT_CRIBBED = 212
RAW_INTERVALS = 122_656
RAW_UNIQUE_PAIRS = 26_770
HUME_ECCO_IDS = 17

# --- Feature comparison table (Positive vs Negative): mean, median, std, and difference ---
FEATURE_COMPARISON_ROWS = [
    {"feature": "reuse_length_dst", "pos_mean": 2347.0170, "pos_median": 1695.0, "pos_std": 1982.8751,
     "neg_mean": 103.0281, "neg_median": 219.0, "neg_std": 8083.2807, "diff_mean": 2243.9889, "diff_median": 1476.0},
    {"feature": "reuse_length_src", "pos_mean": 2348.8295, "pos_median": 1700.0, "pos_std": 1989.4000,
     "neg_mean": 357.0333, "neg_median": 214.0, "neg_std": 398.6721, "diff_mean": 1991.7962, "diff_median": 1486.0},
    {"feature": "num_fragments", "pos_mean": 1.0625, "pos_median": 1.0, "pos_std": 0.3727,
     "neg_mean": 0.9854, "neg_median": 1.0, "neg_std": 0.2184, "diff_mean": 0.0771, "diff_median": 0.0},
    {"feature": "span_ratio_src", "pos_mean": 1.0122, "pos_median": 1.0042, "pos_std": 0.2180,
     "neg_mean": 1.0323, "neg_median": 1.0060, "neg_std": 0.3334, "diff_mean": -0.0202, "diff_median": -0.0017},
    {"feature": "span_ratio_dst", "pos_mean": 1.0201, "pos_median": 1.0104, "pos_std": 0.2186,
     "neg_mean": 1.0549, "neg_median": 1.0361, "neg_std": 0.3355, "diff_mean": -0.0348, "diff_median": -0.0258},
    {"feature": "overlap_ratio_src", "pos_mean": 0.9645, "pos_median": 1.0, "pos_std": 0.1817,
     "neg_mean": 0.9612, "neg_median": 0.9976, "neg_std": 0.1730, "diff_mean": 0.0033, "diff_median": 0.0024},
    {"feature": "overlap_ratio_dst", "pos_mean": 0.9647, "pos_median": 1.0, "pos_std": 0.1818,
     "neg_mean": 0.9641, "neg_median": 1.0, "neg_std": 0.1734, "diff_mean": 0.0006, "diff_median": 0.0},
    {"feature": "pair_reuse_density", "pos_mean": 38.3239, "pos_median": 24.5, "pos_std": 37.7334,
     "neg_mean": 23.5671, "neg_median": 20.0, "neg_std": 18.7738, "diff_mean": 14.7567, "diff_median": 4.5},
    {"feature": "alignment_stability.shift_stability", "pos_mean": 0.8850, "pos_median": 1.0, "pos_std": 0.3136,
     "neg_mean": 0.9529, "neg_median": 1.0, "neg_std": 0.2104, "diff_mean": -0.0679, "diff_median": 0.0},
    {"feature": "alignment_stability.shift_std", "pos_mean": 11.6855, "pos_median": 0.0, "pos_std": 123.0941,
     "neg_mean": 0.9249, "neg_median": 0.0, "neg_std": 11.2363, "diff_mean": 10.7606, "diff_median": 0.0},
]

# --- Threshold scan tables: feature_key -> list of {threshold, tpr, fpr, separability_proxy}, direction note ---
THRESHOLD_SCANS = {
    "basic_features.reuse_length_dst": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [
            (1679.0, 0.5057, 0.0239, 0.9548), (2164.0, 0.4091, 0.0114, 0.9728),
            (2619.0, 0.3068, 0.0042, 0.9866), (3519.0, 0.2102, 0.0, 1.0),
            (5461.0, 0.1080, 0.0, 1.0), (6062.0, 0.0568, 0.0, 1.0),
            (8431.0, 0.0170, 0.0, 1.0), (10624.0, 0.0057, 0.0, 1.0),
        ],
    },
    "basic_features.reuse_length_src": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [
            (1688.0, 0.5057, 0.0239, 0.9548), (2133.0, 0.4091, 0.0114, 0.9728),
            (2586.0, 0.3068, 0.0042, 0.9866), (3535.0, 0.2102, 0.0, 1.0),
            (5489.0, 0.1080, 0.0, 1.0), (6150.0, 0.0568, 0.0, 1.0),
            (8363.0, 0.0170, 0.0, 1.0), (10632.0, 0.0057, 0.0, 1.0),
        ],
    },
    "basic_features.num_fragments": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [(1.0, 0.9659, 0.9688, 0.4993), (2.0, 0.0909, 0.0166, 0.8452), (3.0, 0.0057, 0.0, 1.0)],
    },
    "basic_features.span_ratio_src": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [
            (1.0099, 0.5057, 0.6774, 0.4274), (1.0254, 0.4091, 0.5692, 0.4182),
            (1.0345, 0.3068, 0.5172, 0.3724), (1.0764, 0.2102, 0.3684, 0.3633),
            (1.1326, 0.1080, 0.2341, 0.3156), (1.2857, 0.0568, 0.0447, 0.5594),
            (1.4720, 0.0170, 0.0062, 0.7319), (1.6019, 0.0057, 0.0052, 0.5220),
        ],
    },
    "basic_features.span_ratio_dst": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [
            (1.0099, 0.5057, 0.6774, 0.4274), (1.0345, 0.3068, 0.5172, 0.3724),
            (1.1326, 0.1080, 0.2341, 0.3156), (1.4720, 0.0170, 0.0062, 0.7319),
        ],
    },
    "basic_features.overlap_ratio_src": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [(1.0, 0.5057, 0.4589, 0.5243)],
    },
    "basic_features.overlap_ratio_dst": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [(1.0, 0.6080, 0.5994, 0.5036)],
    },
    "pair_reuse_density": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [
            (24.0, 0.5284, 0.3455, 0.6047), (29.0, 0.4375, 0.2279, 0.6575),
            (41.0, 0.3068, 0.0520, 0.8550), (47.0, 0.2159, 0.0427, 0.8350),
            (58.0, 0.1193, 0.0260, 0.8210), (136.0, 0.0568, 0.0062, 0.9010),
            (187.0, 0.0170, 0.0052, 0.7661),
        ],
    },
    "alignment_stability.shift_stability": {
        "direction": "≥ threshold (is_reverse=False)",
        "rows": [(1.0, 0.8807, 0.9521, 0.4805)],
    },
    "alignment_stability.shift_std": {
        "direction": "≤ threshold (is_reverse=True)",
        "rows": [
            (0.0, 0.9148, 0.9834, 0.4819), (18.3848, 0.9545, 0.9865, 0.4918),
            (76.3675, 0.9886, 0.9979, 0.4977), (76.3675, 0.9886, 0.9979, 0.4977),
        ],
    },
}

# --- Signal strength for ranking (bar chart / takeaway) ---
SIGNAL_STRENGTH = [
    {"name": "Length", "strength": "strong"},
    {"name": "Density", "strength": "weak"},
    {"name": "Fragmentation", "strength": "ineffective"},
    {"name": "Span ratio", "strength": "ineffective"},
    {"name": "Coverage", "strength": "ineffective"},
    {"name": "Alignment", "strength": "ineffective"},
]

# --- Split counts (Discovery / Main Eval / Hard Eval) ---
DISCOVERY_COUNTS = {"Attributed Reprint": 38, "Unattributed Reprint": 10, "Quoted": 30, "Cribbed": 6}
MAIN_EVAL_COUNTS = {"Attributed Reprint": 76, "Unattributed Reprint": 25, "Quoted": 152, "Cribbed": 48}
HARD_EVAL_COUNTS = {"Unattributed Reprint": 11, "Attributed Reprint": 17, "Quoted": 94, "Cribbed": 43}
GROUP_B_REPRINTS = 24  # pairs
GROUP_B_ENTRIES = 25
HARD_NEGATIVES_PAIRS = 75
HARD_NEGATIVES_ENTRIES = 140

# --- Discovery set by src_doc_id ---
DISCOVERY_BY_SRC = {
    "0255100701": 20, "0255100702": 11, "0409602200": 6, "1653000501": 5, "0302000100": 26,
    "0913700101": 5, "0375001000": 3, "0379203102": 1, "0421601400": 2, "0441200400": 2, "1566600301": 3,
}
MAIN_EVAL_BY_SRC = {
    "0255100701": 101, "0302000100": 69, "0409602200": 13, "0255100702": 27, "0421601400": 10,
    "1566600301": 10, "0455700101": 3, "0375001000": 10, "1070400201": 10, "0441200400": 10,
    "1498700304": 7, "0691100800": 4, "1653000501": 10, "1424100200": 2, "0913700101": 10,
    "0379203102": 1, "1723500601": 4,
}
HARD_EVAL_BY_SRC = {
    "0255100701": 38, "1653000501": 3, "0255100702": 15, "1070400201": 2, "0302000100": 53,
    "0421601400": 34, "0913700101": 7, "0409602200": 8, "0441200400": 1, "0691100800": 3, "0375001000": 1,
}
