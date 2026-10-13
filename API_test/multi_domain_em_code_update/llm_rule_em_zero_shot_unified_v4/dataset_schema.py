from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional

Domain = Literal["paper", "product", "restaurant", "beer", "music", "generic"]


@dataclass
class CanonicalRecord:
    id: str
    name: str = ""
    description: str = ""
    party: str = ""
    category: str = ""
    location: str = ""
    time: str = ""
    numeric: str = ""
    code: str = ""
    extra: str = ""


@dataclass
class DatasetSchema:
    dataset_name: str
    domain: Domain
    left_map: Dict[str, str | list[str]]
    right_map: Dict[str, str | list[str]]
    desc_order: list[str]
    feature_profile: str
    generic_name_lexicon: set[str] = field(default_factory=set)
    numeric_kind: Optional[str] = None
    time_kind: Optional[str] = None
    code_kind: Optional[str] = None


ALIAS_GROUPS: dict[str, set[str]] = {
    "id": {"id", "_id"},
    "name": {"name", "title", "paper_title", "product_name", "beer_name", "song_name", "restaurant_name"},
    "description": {"description", "desc", "album_name", "addr", "address", "subtitle", "specification", "specs"},
    "party": {"authors", "author", "brand", "manufacturer", "artist_name", "brew_factory_name", "brewery", "creator", "creators"},
    "category": {"venue", "journal", "booktitle", "conference", "publisher", "source", "category", "genre", "style", "type", "cuisine"},
    "location": {"city", "state", "region"},
    "time": {"year", "date", "pub_year", "publication_year", "release_year", "released"},
    "numeric": {"price", "abv", "class", "score", "rating"},
    "code": {"modelno", "model", "phone", "sku", "upc"},
    "extra": {"copyright", "copy_right", "copy right", "time", "duration", "notes", "comment"},
}


CANONICAL_FIELDS = ["id", "name", "description", "party", "category", "location", "time", "numeric", "code", "extra"]

DATASET_SCHEMAS: dict[str, DatasetSchema] = {
    "dblp_acm": DatasetSchema(
        dataset_name="dblp_acm",
        domain="paper",
        left_map={"id": "id", "name": "title", "party": "authors", "category": "venue", "time": "year"},
        right_map={"id": "id", "name": "title", "party": "authors", "category": "venue", "time": "year"},
        desc_order=["name", "party", "category", "time"],
        feature_profile="paper",
        generic_name_lexicon={"editorial", "preface", "foreword", "introduction", "announcement", "tutorial", "panel"},
        time_kind="year",
    ),
    "dblp_scholar": DatasetSchema(
        dataset_name="dblp_scholar",
        domain="paper",
        left_map={"id": "id", "name": "title", "party": "authors", "category": "venue", "time": "year"},
        right_map={"id": "id", "name": "title", "party": "authors", "category": "venue", "time": "year"},
        desc_order=["name", "party", "category", "time"],
        feature_profile="paper",
        generic_name_lexicon={"editorial", "preface", "foreword", "introduction", "announcement", "tutorial", "panel"},
        time_kind="year",
    ),
    "abt_buy": DatasetSchema(
        dataset_name="abt_buy",
        domain="product",
        left_map={"id": "id", "name": "name", "description": "description", "numeric": "price"},
        right_map={"id": "id", "name": "name", "description": "description", "numeric": "price"},
        desc_order=["name", "description", "numeric"],
        feature_profile="product",
        numeric_kind="price",
    ),
    "walmart_amazon": DatasetSchema(
        dataset_name="walmart_amazon",
        domain="product",
        left_map={"id": "id", "name": "title", "party": "brand", "category": "category", "code": "modelno", "numeric": "price"},
        right_map={"id": "id", "name": "title", "party": "brand", "category": "category", "code": "modelno", "numeric": "price"},
        desc_order=["name", "party", "category", "code", "numeric"],
        feature_profile="product",
        numeric_kind="price",
        code_kind="model",
    ),
    "beer": DatasetSchema(
        dataset_name="beer",
        domain="beer",
        left_map={"id": "id", "name": "Beer_Name", "party": "Brew_Factory_Name", "category": "Style", "numeric": "ABV"},
        right_map={"id": "id", "name": "Beer_Name", "party": "Brew_Factory_Name", "category": "Style", "numeric": "ABV"},
        desc_order=["name", "party", "category", "numeric"],
        feature_profile="beer",
        numeric_kind="abv",
    ),
    "fodors_zagat": DatasetSchema(
        dataset_name="fodors_zagat",
        domain="restaurant",
        left_map={"id": "id", "name": "name", "description": "addr", "location": "city", "code": "phone", "category": "type", "numeric": "class"},
        right_map={"id": "id", "name": "name", "description": "addr", "location": "city", "code": "phone", "category": "type", "numeric": "class"},
        desc_order=["name", "description", "location", "code", "category"],
        feature_profile="restaurant",
        numeric_kind="class",
        code_kind="phone",
    ),
    "itunes_amazon": DatasetSchema(
        dataset_name="itunes_amazon",
        domain="music",
        left_map={"id": "id", "name": "Song_Name", "party": "Artist_Name", "description": "Album_Name", "category": "Genre", "time": "Released", "numeric": "Price", "extra": ["CopyRight", "Time"]},
        right_map={"id": "id", "name": "Song_Name", "party": "Artist_Name", "description": "Album_Name", "category": "Genre", "time": "Released", "numeric": "Price", "extra": ["CopyRight", "Time"]},
        desc_order=["name", "party", "description", "category", "time", "numeric"],
        feature_profile="music",
        numeric_kind="price",
        time_kind="date",
    ),
}


def _lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def _find_alias_column(columns: Iterable[str], aliases: set[str]) -> Optional[str]:
    lowered = _lower_map(columns)
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    for key, raw in lowered.items():
        if key.replace("_", " ") in aliases:
            return raw
    return None


def _materialize_mapping(raw_map: Dict[str, str | list[str]], columns: Iterable[str]) -> Dict[str, str | list[str]]:
    lowered = _lower_map(columns)
    resolved: Dict[str, str | list[str]] = {}
    for canonical_field, source in raw_map.items():
        if isinstance(source, list):
            actual = [lowered[s.lower()] for s in source if s.lower() in lowered]
            if actual:
                resolved[canonical_field] = actual
            continue
        actual = lowered.get(source.lower())
        if actual:
            resolved[canonical_field] = actual
    return resolved


def _resolve_registry_schema(schema: DatasetSchema, left_cols: Iterable[str], right_cols: Iterable[str]) -> DatasetSchema:
    return DatasetSchema(
        dataset_name=schema.dataset_name,
        domain=schema.domain,
        left_map=_materialize_mapping(schema.left_map, left_cols),
        right_map=_materialize_mapping(schema.right_map, right_cols),
        desc_order=list(schema.desc_order),
        feature_profile=schema.feature_profile,
        generic_name_lexicon=set(schema.generic_name_lexicon),
        numeric_kind=schema.numeric_kind,
        time_kind=schema.time_kind,
        code_kind=schema.code_kind,
    )


def auto_detect_schema(left_cols: Iterable[str], right_cols: Iterable[str], dataset_name: str = "auto") -> DatasetSchema:
    def detect_side(cols: Iterable[str]) -> Dict[str, str | list[str]]:
        mapping: Dict[str, str | list[str]] = {}
        for canonical_field, aliases in ALIAS_GROUPS.items():
            found = _find_alias_column(cols, aliases)
            if found:
                mapping[canonical_field] = found
        return mapping

    left_map = detect_side(left_cols)
    right_map = detect_side(right_cols)
    domain = "generic"
    profile = "generic"
    if any(x in left_map or x in right_map for x in ["category", "time"]) and ("party" in left_map or "party" in right_map):
        if (left_map.get("category") == "venue") or (right_map.get("category") == "venue"):
            domain = profile = "paper"
    if any("phone" in str(v).lower() for v in list(left_map.values()) + list(right_map.values())):
        domain = profile = "restaurant"
    elif any("abv" in str(v).lower() for v in list(left_map.values()) + list(right_map.values())):
        domain = profile = "beer"
    elif any("artist" in str(v).lower() or "song" in str(v).lower() for v in list(left_map.values()) + list(right_map.values())):
        domain = profile = "music"
    elif any(x in left_map or x in right_map for x in ["numeric", "code", "description"]):
        domain = profile = "product"
    elif domain == "generic" and ("time" in left_map or "time" in right_map) and ("party" in left_map or "party" in right_map):
        domain = profile = "paper"

    desc_order = [f for f in ["name", "party", "description", "category", "location", "time", "numeric", "code", "extra"] if f in left_map or f in right_map]
    if not desc_order:
        desc_order = [f for f in ["name", "description", "party", "category", "time"] if f in CANONICAL_FIELDS]
    return DatasetSchema(
        dataset_name=dataset_name,
        domain=domain,
        left_map=left_map,
        right_map=right_map,
        desc_order=desc_order,
        feature_profile=profile,
        generic_name_lexicon=set(),
        numeric_kind="price" if profile in {"product", "music"} else ("abv" if profile == "beer" else None),
        time_kind="year" if profile == "paper" else None,
        code_kind="phone" if profile == "restaurant" else None,
    )


def resolve_schema(dataset_name: str, left_cols: Iterable[str], right_cols: Iterable[str]) -> DatasetSchema:
    if dataset_name and dataset_name in DATASET_SCHEMAS:
        return _resolve_registry_schema(DATASET_SCHEMAS[dataset_name], left_cols, right_cols)
    return auto_detect_schema(left_cols, right_cols, dataset_name=dataset_name or "auto")
