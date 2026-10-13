from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureProfile:
    name_signals: list[str]
    corroborators: list[str]
    generic_flag: str
    numeric_rel_diff_feature: str | None = None
    time_diff_feature: str | None = None
    safe_yes_numeric_rel_diff_max: float | None = None
    safe_yes_time_diff_max: float | None = None


FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "paper": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard"],
        corroborators=["party_overlap", "first_party_same", "time_equal", "category_exact"],
        generic_flag="both_generic_name",
        time_diff_feature="time_diff",
        safe_yes_time_diff_max=0,
    ),
    "product": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard", "code_exact"],
        corroborators=["party_exact", "code_exact", "numeric_equal", "category_exact", "shared_numeric_tokens"],
        generic_flag="both_generic_name",
        numeric_rel_diff_feature="numeric_rel_diff",
        safe_yes_numeric_rel_diff_max=0.10,
    ),
    "restaurant": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard", "code_exact"],
        corroborators=["code_exact", "location_seq", "desc_seq", "category_exact"],
        generic_flag="both_generic_name",
        numeric_rel_diff_feature="numeric_rel_diff",
    ),
    "beer": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard"],
        corroborators=["party_exact", "category_exact", "numeric_rel_diff"],
        generic_flag="both_generic_name",
        numeric_rel_diff_feature="numeric_rel_diff",
        safe_yes_numeric_rel_diff_max=0.08,
    ),
    "music": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard"],
        corroborators=["party_exact", "desc_seq", "time_equal", "category_exact"],
        generic_flag="both_generic_name",
        numeric_rel_diff_feature="numeric_rel_diff",
        safe_yes_numeric_rel_diff_max=0.20,
    ),
    "generic": FeatureProfile(
        name_signals=["name_exact", "name_seq", "name_jaccard"],
        corroborators=["desc_seq", "party_overlap", "category_exact", "time_equal", "numeric_equal", "code_exact"],
        generic_flag="both_generic_name",
        numeric_rel_diff_feature="numeric_rel_diff",
        time_diff_feature="time_diff",
    ),
}
