from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .dataset_schema import CanonicalRecord


GENERIC_NAME_WORDS = {
    "accessory",
    "adapter",
    "bundle",
    "camera",
    "card",
    "computer",
    "digital",
    "dvd",
    "electronics",
    "kit",
    "laptop",
    "monitor",
    "player",
    "portable",
    "printer",
    "screen",
    "software",
    "system",
    "television",
    "tv",
    "video",
    "wireless",
}

GENERIC_CODE_STOPWORDS = {
    "1080p",
    "720p",
    "480p",
    "54mbps",
    "300mbps",
    "320gb",
    "250gb",
    "160gb",
    "120gb",
    "80gb",
    "60gb",
    "40gb",
    "20gb",
    "16gb",
    "8gb",
    "4gb",
    "2gb",
    "1gb",
    "dvd",
    "usb",
    "hdmi",
    "lcd",
    "hdtv",
    "wifi",
    "bluetooth",
    "mp3",
    "mp4",
}

VARIANT_COLOR_WORDS = {
    "black",
    "white",
    "silver",
    "red",
    "blue",
    "green",
    "pink",
    "gold",
    "gray",
    "grey",
    "brown",
    "purple",
    "yellow",
}

VARIANT_STORAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(?:gb|mb|tb)\b", re.IGNORECASE)
VARIANT_SIZE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(?:inch|in|\"|cm)\b", re.IGNORECASE)
VARIANT_PACK_PATTERN = re.compile(r"\b\d+\s?(?:pack|pk)\b", re.IGNORECASE)
VARIANT_GEN_PATTERN = re.compile(r"\b(?:mk\d+|gen\d+|series\s*[ivx\d]+|v\d+(?:\.\d+)?)\b", re.IGNORECASE)


def load_dataset(
    table_a_path: str,
    table_b_path: Optional[str] = None,
    pairs_path: Optional[str] = None,
):
    def _read_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    if table_b_path is None and pairs_path is None:
        return _read_csv(table_a_path)

    if table_b_path is None or pairs_path is None:
        raise ValueError("load_dataset() must receive either 1 path or 3 paths.")

    table_a = _read_csv(table_a_path)
    table_b = _read_csv(table_b_path)
    pairs = _read_csv(pairs_path)
    return table_a, table_b, pairs


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    text = _safe_str(text).lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9\s\-./]", " ", text)
    text = _normalize_ws(text)
    return text


def _alpha_num_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize_text(text))


def _tokenize(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text)


def _token_set(text: str) -> Set[str]:
    return set(_tokenize(text))


def _seq_ratio(a: str, b: str) -> float:
    import difflib

    a = _normalize_text(a)
    b = _normalize_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _overlap_coeff(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = min(len(sa), len(sb))
    return inter / denom if denom else 0.0


def _numeric_tokens(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    vals = re.findall(r"\b\d+(?:\.\d+)?[a-z]*\b", text.lower())
    return set(v.strip() for v in vals if v.strip())


def normalize_phone(value: str) -> str:
    return re.sub(r"\D", "", _safe_str(value))


def normalize_code(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _safe_str(value).lower())


def parse_numeric(value: str, kind: Optional[str] = None) -> Optional[float]:
    text = _safe_str(value).lower()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("usd", "")
    text = text.replace("%", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_time_value(value: str, kind: Optional[str] = None):
    text = _safe_str(value)
    if not text:
        return None

    if kind == "year":
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return int(m.group(0))

    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return int(m.group(0))

    return text


def is_generic_name(text: str) -> int:
    toks = _tokenize(text)
    if not toks:
        return 0
    long_toks = [t for t in toks if len(t) >= 3]
    if not long_toks:
        return 1
    generic_hits = sum(1 for t in long_toks if t in GENERIC_NAME_WORDS)
    ratio = generic_hits / len(long_toks)
    return 1 if ratio >= 0.6 else 0


def normalize_model_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^a-z0-9]", "", token)
    return token


def is_model_like_token(token: str) -> bool:
    token = normalize_model_token(token)
    if not token:
        return False
    if token in GENERIC_CODE_STOPWORDS:
        return False
    if len(token) < 4:
        return False

    has_alpha = bool(re.search(r"[a-z]", token))
    has_digit = bool(re.search(r"\d", token))
    if not (has_alpha and has_digit):
        return False

    if re.fullmatch(r"\d+(gb|mb|tb|p|hz|inch|in)", token):
        return False

    return True


def is_strong_model_like_token(token: str) -> bool:
    token = normalize_model_token(token)
    if not token:
        return False
    if token in GENERIC_CODE_STOPWORDS:
        return False
    if len(token) < 5:
        return False

    has_alpha = bool(re.search(r"[a-z]", token))
    has_digit = bool(re.search(r"\d", token))
    if not (has_alpha and has_digit):
        return False

    if re.search(r"\d+(?:user|pack|suite|edition|version|pro|plus|deluxe)", token):
        return False
    if any(w in token for w in ["user", "pack", "suite", "edition", "version"]):
        return False

    if not (
        re.search(r"[a-z]{2,}\d{1,}", token)
        or re.search(r"\d{1,}[a-z]{2,}", token)
        or re.search(r"[a-z]+\d+[a-z0-9]+", token)
    ):
        return False

    if re.fullmatch(r"\d+(gb|mb|tb|p|hz|inch|in)", token):
        return False

    return True


def extract_model_tokens(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()

    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/\-\.]{2,}", text.lower())
    result: Set[str] = set()

    for tok in raw_tokens:
        norm = normalize_model_token(tok)
        if is_model_like_token(norm):
            result.add(norm)

    return result


def extract_strong_model_tokens(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()

    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/\-\.]{2,}", text.lower())
    result: Set[str] = set()
    for tok in raw_tokens:
        norm = normalize_model_token(tok)
        if is_strong_model_like_token(norm):
            result.add(norm)
    return result


def _extract_variant_tokens(text: str) -> Set[str]:
    text = _normalize_text(text)
    if not text:
        return set()

    result: Set[str] = set()
    for color in VARIANT_COLOR_WORDS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            result.add(color)

    for pattern in [VARIANT_STORAGE_PATTERN, VARIANT_SIZE_PATTERN, VARIANT_PACK_PATTERN, VARIANT_GEN_PATTERN]:
        for match in pattern.findall(text):
            result.add(_normalize_ws(str(match).lower()))

    return result


def _variant_conflict(left_text: str, right_text: str) -> float:
    left_vars = _extract_variant_tokens(left_text)
    right_vars = _extract_variant_tokens(right_text)
    if not left_vars or not right_vars:
        return 0.0

    left_storage = {v for v in left_vars if VARIANT_STORAGE_PATTERN.fullmatch(v)}
    right_storage = {v for v in right_vars if VARIANT_STORAGE_PATTERN.fullmatch(v)}
    if left_storage and right_storage and left_storage.isdisjoint(right_storage):
        return 1.0

    left_size = {v for v in left_vars if VARIANT_SIZE_PATTERN.fullmatch(v)}
    right_size = {v for v in right_vars if VARIANT_SIZE_PATTERN.fullmatch(v)}
    if left_size and right_size and left_size.isdisjoint(right_size):
        return 1.0

    left_pack = {v for v in left_vars if VARIANT_PACK_PATTERN.fullmatch(v)}
    right_pack = {v for v in right_vars if VARIANT_PACK_PATTERN.fullmatch(v)}
    if left_pack and right_pack and left_pack.isdisjoint(right_pack):
        return 1.0

    left_gen = {v for v in left_vars if VARIANT_GEN_PATTERN.fullmatch(v)}
    right_gen = {v for v in right_vars if VARIANT_GEN_PATTERN.fullmatch(v)}
    if left_gen and right_gen and left_gen.isdisjoint(right_gen):
        return 1.0

    left_colors = left_vars & VARIANT_COLOR_WORDS
    right_colors = right_vars & VARIANT_COLOR_WORDS
    if left_colors and right_colors and left_colors.isdisjoint(right_colors):
        return 1.0

    return 0.0


def _price_conflict(left_numeric: str, right_numeric: str) -> float:
    n1 = parse_numeric(left_numeric, "numeric")
    n2 = parse_numeric(right_numeric, "numeric")
    if n1 is None or n2 is None:
        return 0.0
    if min(n1, n2) <= 0:
        return 0.0
    rel_diff = abs(n1 - n2) / max(abs(n1), abs(n2), 1.0)
    return 1.0 if rel_diff >= 0.50 else 0.0


def _category_conflict(left_category: str, right_category: str) -> float:
    left_tokens = _token_set(left_category)
    right_tokens = _token_set(right_category)
    if not left_tokens or not right_tokens:
        return 0.0
    return 1.0 if _overlap_coeff(left_tokens, right_tokens) == 0.0 else 0.0


def _is_missing_text(value) -> bool:
    text = _safe_str(value)
    return text == "" or text.lower() in {"nan", "none", "null"}


def _get_field_value(row: pd.Series, field_name) -> str:
    """
    Support schema mapping values of type:
    - str: single source column
    - list[str]: fallback columns, return the first non-empty value
    """
    if not field_name:
        return ""

    # fallback chain
    if isinstance(field_name, list):
        for col in field_name:
            if col in row.index:
                value = row[col]
                if not _is_missing_text(value):
                    return _safe_str(value)
        return ""

    # single column
    if isinstance(field_name, str):
        if field_name not in row.index:
            return ""
        value = row[field_name]
        if _is_missing_text(value):
            return ""
        return _safe_str(value)

    return ""



def row_to_canonical(row: pd.Series, schema, side: str) -> CanonicalRecord:
    mapping = schema.left_map if side == "left" else schema.right_map

    def g(key: str) -> str:
        return _get_field_value(row, mapping.get(key))

    # extra 在 CanonicalRecord 中是 str，不是 dict
    extra_parts: List[str] = []

    major_keys = {
        "id",
        "name",
        "description",
        "party",
        "category",
        "location",
        "time",
        "numeric",
        "code",
        "extra",
    }

    for k, src in mapping.items():
        if k in major_keys:
            continue
        value = _get_field_value(row, src)
        if value:
            extra_parts.append(f"{k}: {value}")

    # 如果 schema 本身定义了 extra，也一起纳入
    extra_value = g("extra")
    if extra_value:
        extra_parts.insert(0, extra_value)

    rec = CanonicalRecord(
        id=g("id"),
        name=g("name"),
        description=g("description"),
        party=g("party"),
        category=g("category"),
        location=g("location"),
        time=g("time"),
        numeric=g("numeric"),
        code=g("code"),
        extra="; ".join(extra_parts),
    )
    return rec



def build_entity_desc_from_canonical(rec: CanonicalRecord, desc_order: Sequence[str]) -> str:
    parts: List[str] = []
    for field in desc_order:
        value = getattr(rec, field, "")
        value = _safe_str(value)
        if value:
            parts.append(f"{field}: {value}")
    return "; ".join(parts)


def _active_major_fields(schema) -> List[str]:
    fields = []
    for key in ["name", "description", "party", "category", "location", "time", "numeric", "code"]:
        left_col = schema.left_map.get(key)
        right_col = schema.right_map.get(key)
        if left_col or right_col:
            fields.append(key)
    return fields


def _missing_count(rec: CanonicalRecord, fields: Sequence[str]) -> int:
    cnt = 0
    for f in fields:
        if not _safe_str(getattr(rec, f, "")):
            cnt += 1
    return cnt


def _relative_numeric_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return -1.0
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _build_inferred_model_features(left: CanonicalRecord, right: CanonicalRecord) -> Dict[str, float]:
    left_text = f"{left.name or ''} {left.description or ''}"
    right_text = f"{right.name or ''} {right.description or ''}"

    left_codes = extract_model_tokens(left_text)
    right_codes = extract_model_tokens(right_text)
    shared_codes = left_codes & right_codes

    return {
        "inferred_code_overlap_count": float(len(shared_codes)),
        "inferred_code_exact": 1.0 if len(shared_codes) > 0 else 0.0,
        "inferred_code_jaccard": (
            len(shared_codes) / len(left_codes | right_codes)
            if (left_codes or right_codes)
            else 0.0
        ),
        "inferred_left_code_count": float(len(left_codes)),
        "inferred_right_code_count": float(len(right_codes)),
    }


def _build_brand_and_model_features(left: CanonicalRecord, right: CanonicalRecord) -> Dict[str, float]:
    left_brand = _safe_str(left.party)
    right_brand = _safe_str(right.party)
    left_brand_tokens = _token_set(left_brand)
    right_brand_tokens = _token_set(right_brand)

    left_text = f"{left.name or ''} {left.description or ''}"
    right_text = f"{right.name or ''} {right.description or ''}"

    left_strong_models = extract_strong_model_tokens(left_text)
    right_strong_models = extract_strong_model_tokens(right_text)
    shared_strong = left_strong_models & right_strong_models

    strong_exact = 1.0 if left_strong_models and right_strong_models and left_strong_models == right_strong_models else 0.0

    return {
        "brand_exact": 1.0 if _alpha_num_compact(left_brand) and _alpha_num_compact(left_brand) == _alpha_num_compact(right_brand) else 0.0,
        "brand_overlap": _overlap_coeff(left_brand_tokens, right_brand_tokens),
        "strong_model_exact": strong_exact,
        "strong_model_overlap_count": float(len(shared_strong)),
        "variant_conflict": _variant_conflict(left_text, right_text),
        "price_conflict": _price_conflict(left.numeric, right.numeric),
        "category_conflict": _category_conflict(left.category, right.category),
        "left_strong_model_count": float(len(left_strong_models)),
        "right_strong_model_count": float(len(right_strong_models)),
    }


def compare_records(left: CanonicalRecord, right: CanonicalRecord, schema) -> Dict[str, float]:
    left_name = _safe_str(left.name)
    right_name = _safe_str(right.name)
    left_desc = _safe_str(left.description)
    right_desc = _safe_str(right.description)
    left_party = _safe_str(left.party)
    right_party = _safe_str(right.party)
    left_category = _safe_str(left.category)
    right_category = _safe_str(right.category)
    left_location = _safe_str(left.location)
    right_location = _safe_str(right.location)
    left_time = _safe_str(left.time)
    right_time = _safe_str(right.time)
    left_numeric = _safe_str(left.numeric)
    right_numeric = _safe_str(right.numeric)
    left_code = _safe_str(left.code)
    right_code = _safe_str(right.code)

    feats: Dict[str, float] = {}

    feats["name_exact"] = 1.0 if _alpha_num_compact(left_name) and _alpha_num_compact(left_name) == _alpha_num_compact(right_name) else 0.0
    feats["name_seq"] = _seq_ratio(left_name, right_name)
    feats["name_jaccard"] = _jaccard(_token_set(left_name), _token_set(right_name))
    feats["name_overlap"] = _overlap_coeff(_token_set(left_name), _token_set(right_name))
    feats["both_generic_name"] = 1.0 if is_generic_name(left_name) and is_generic_name(right_name) else 0.0

    feats["desc_seq"] = _seq_ratio(left_desc, right_desc)
    feats["desc_jaccard"] = _jaccard(_token_set(left_desc), _token_set(right_desc))
    feats["desc_overlap"] = _overlap_coeff(_token_set(left_desc), _token_set(right_desc))

    left_party_tokens = _token_set(left_party)
    right_party_tokens = _token_set(right_party)
    feats["party_exact"] = 1.0 if _alpha_num_compact(left_party) and _alpha_num_compact(left_party) == _alpha_num_compact(right_party) else 0.0
    feats["party_overlap"] = _overlap_coeff(left_party_tokens, right_party_tokens)

    feats["category_exact"] = 1.0 if _alpha_num_compact(left_category) and _alpha_num_compact(left_category) == _alpha_num_compact(right_category) else 0.0
    feats["category_seq"] = _seq_ratio(left_category, right_category)

    feats["location_seq"] = _seq_ratio(left_location, right_location)

    t1 = parse_time_value(left_time, "year")
    t2 = parse_time_value(right_time, "year")
    feats["time_equal"] = 1.0 if (t1 is not None and t2 is not None and t1 == t2) else 0.0

    n1 = parse_numeric(left_numeric, "numeric")
    n2 = parse_numeric(right_numeric, "numeric")
    feats["numeric_equal"] = 1.0 if (n1 is not None and n2 is not None and abs(n1 - n2) < 1e-9) else 0.0
    feats["numeric_rel_diff"] = _relative_numeric_diff(n1, n2)

    left_numeric_tokens = _numeric_tokens(f"{left_name} {left_desc} {left_numeric}")
    right_numeric_tokens = _numeric_tokens(f"{right_name} {right_desc} {right_numeric}")
    feats["shared_numeric_tokens"] = float(len(left_numeric_tokens & right_numeric_tokens))

    norm_left_code = normalize_code(left_code)
    norm_right_code = normalize_code(right_code)
    feats["code_exact"] = 1.0 if norm_left_code and norm_left_code == norm_right_code else 0.0

    left_code_tokens = {norm_left_code} if norm_left_code else set()
    right_code_tokens = {norm_right_code} if norm_right_code else set()
    feats["code_token_overlap"] = _overlap_coeff(left_code_tokens, right_code_tokens)

    feats.update(_build_inferred_model_features(left, right))
    feats.update(_build_brand_and_model_features(left, right))

    active_fields = _active_major_fields(schema)
    feats["left_major_missing_count"] = float(_missing_count(left, active_fields))
    feats["right_major_missing_count"] = float(_missing_count(right, active_fields))
    feats["major_missing_count"] = float(_missing_count(left, active_fields) + _missing_count(right, active_fields))

    feats["party_tokens_jaccard"] = _jaccard(left_party_tokens, right_party_tokens)

    return feats


def summarize_feature_stats_unlabeled_from_pairs(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    pairs: pd.DataFrame,
    schema,
) -> Dict[str, Dict[str, float]]:
    if pairs.empty:
        return {}

    a_id_col = table_a.columns[0]
    b_id_col = table_b.columns[0]

    rows: List[Dict[str, float]] = []
    for row in pairs.itertuples(index=False):
        l_id = str(getattr(row, "ltable_id"))
        r_id = str(getattr(row, "rtable_id"))

        a_match = table_a[table_a[a_id_col] == l_id]
        b_match = table_b[b_id_col == b_id_col] if False else table_b[table_b[b_id_col] == r_id]
        if a_match.empty or b_match.empty:
            continue

        left = row_to_canonical(a_match.iloc[0], schema, "left")
        right = row_to_canonical(b_match.iloc[0], schema, "right")
        comp = compare_records(left, right, schema)
        rows.append(comp)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    stats: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        stats[col] = {
            "min": round(float(s.min()), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p90": round(float(s.quantile(0.90)), 6),
            "p95": round(float(s.quantile(0.95)), 6),
            "max": round(float(s.max()), 6),
        }
    return stats
