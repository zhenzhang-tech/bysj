from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _iter_conditions(rule: Dict):
    return rule.get("all", []) if isinstance(rule, dict) else []


def _has_feature(rule: Dict, feature: str, op: Optional[str] = None, value=None) -> bool:
    for cond in _iter_conditions(rule):
        if cond.get("feature") != feature:
            continue
        if op is not None and cond.get("op") != op:
            continue
        if value is not None and cond.get("value") != value:
            continue
        return True
    return False


def _has_threshold(rule: Dict, feature: str, threshold: float, direction: str = ">=") -> bool:
    for cond in _iter_conditions(rule):
        if cond.get("feature") != feature:
            continue
        op = cond.get("op")
        val = cond.get("value")
        if not isinstance(val, (int, float)):
            continue
        if direction == ">=" and op in {">=", ">"} and val >= threshold:
            return True
        if direction == "<=" and op in {"<=", "<"} and val <= threshold:
            return True
    return False


def _rule_references_only_allowed(rule: Dict, allowed_features: set) -> bool:
    for cond in _iter_conditions(rule):
        if cond.get("feature") not in allowed_features:
            return False
    return True


def _estimate_zero_coverage(rule: Dict, stats: Optional[Dict[str, Dict]]) -> bool:
    if not stats:
        return False

    for cond in _iter_conditions(rule):
        feat = cond.get("feature")
        op = cond.get("op")
        val = cond.get("value")

        if feat not in stats:
            continue
        feat_stats = stats[feat]
        if not isinstance(feat_stats, dict):
            continue

        observed_min = feat_stats.get("min")
        observed_max = feat_stats.get("max")

        if not isinstance(val, (int, float)):
            continue
        if observed_min is None or observed_max is None:
            continue

        if op in {">=", ">"} and val > observed_max:
            return True
        if op in {"<=", "<"} and val < observed_min:
            return True
        if op == "==" and (val < observed_min or val > observed_max):
            return True

    return False


def _positive_anchor_ok(rule: Dict) -> bool:
    return (
        _has_feature(rule, "name_exact", "==", 1)
        or _has_threshold(rule, "name_seq", 0.85, direction=">=")
        or _has_feature(rule, "inferred_code_exact", "==", 1)
        or _has_threshold(rule, "inferred_code_overlap_count", 1, direction=">=")
        or (_has_feature(rule, "brand_exact", "==", 1) and _has_feature(rule, "strong_model_exact", "==", 1))
        or _has_threshold(rule, "strong_model_overlap_count", 1, direction=">=")
    )


def _negative_anchor_ok(rule: Dict) -> bool:
    if (
        _has_feature(rule, "inferred_code_overlap_count", "==", 0)
        and _has_threshold(rule, "name_seq", 0.55, direction="<=")
    ):
        return True

    if _has_threshold(rule, "name_seq", 0.45, direction="<="):
        return True

    if _has_feature(rule, "variant_conflict", "==", 1):
        return True

    if _has_feature(rule, "category_conflict", "==", 1):
        return True

    if _has_feature(rule, "brand_exact", "==", 0) and _has_threshold(rule, "strong_model_overlap_count", 0, direction="<="):
        return True

    return False


def _is_product_profile(schema) -> bool:
    return getattr(schema, "feature_profile", None) == "product"


def _product_positive_anchor_ok(rule: Dict) -> bool:
    return (
        (_has_feature(rule, "brand_exact", "==", 1) and _has_feature(rule, "strong_model_exact", "==", 1))
        or _has_threshold(rule, "strong_model_overlap_count", 1, direction=">=")
        or (_has_feature(rule, "name_exact", "==", 1) and _has_feature(rule, "variant_conflict", "==", 0))
        or (_has_feature(rule, "brand_exact", "==", 1) and _has_threshold(rule, "name_seq", 0.60, direction=">="))
    )


def _product_positive_has_conflict_guard(rule: Dict) -> bool:
    return (
        _has_feature(rule, "variant_conflict", "==", 0)
        or _has_feature(rule, "price_conflict", "==", 0)
        or _has_feature(rule, "category_conflict", "==", 0)
    )


def _product_positive_generic_only(rule: Dict) -> bool:
    generic_positive_features = {"name_seq", "desc_seq", "shared_numeric_tokens"}
    used_features = {cond.get("feature") for cond in _iter_conditions(rule)}
    has_anchor = any(
        feat in used_features
        for feat in {
            "brand_exact",
            "brand_overlap",
            "strong_model_exact",
            "strong_model_overlap_count",
            "inferred_code_exact",
            "inferred_code_overlap_count",
        }
    )
    if has_anchor:
        return False
    return bool(used_features) and used_features.issubset(generic_positive_features)


def sanitize_rules(
    rules: List[Dict],
    schema,
    stats: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[Dict], Dict]:
    from .rule_generator import allowed_features_by_schema

    allowed = allowed_features_by_schema(schema)
    kept: List[Dict] = []
    dropped: List[Dict] = []

    for rule in rules:
        if not isinstance(rule, dict):
            dropped.append({"rule": rule, "reason": "invalid_rule_object"})
            continue

        if "decision" not in rule or "all" not in rule:
            dropped.append({"rule": rule, "reason": "missing_required_fields"})
            continue

        if not rule.get("all"):
            dropped.append({"rule": rule, "reason": "empty_conditions"})
            continue

        if not _rule_references_only_allowed(rule, allowed):
            dropped.append({"rule": rule, "reason": "contains_disallowed_feature"})
            continue

        if _estimate_zero_coverage(rule, stats):
            dropped.append({"rule": rule, "reason": "zero_coverage_on_current_batch"})
            continue

        decision = int(rule.get("decision", 0))
        is_product = _is_product_profile(schema)
        is_paper = _is_paper_profile(schema)
        is_beer = _is_beer_profile(schema)
        is_walmart = _is_walmart_amazon(schema)

        if decision == 1:
            if is_walmart:
                if not _walmart_positive_anchor_ok(rule):
                    dropped.append({"rule": rule, "reason": "walmart_yes_missing_model_anchor"})
                    continue
            elif is_product:
                if not _product_positive_anchor_ok(rule):
                    dropped.append({"rule": rule, "reason": "product_yes_missing_strong_anchor"})
                    continue
                if not _product_positive_has_conflict_guard(rule):
                    dropped.append({"rule": rule, "reason": "product_yes_missing_conflict_guard"})
                    continue
                if _product_positive_generic_only(rule):
                    dropped.append({"rule": rule, "reason": "product_yes_generic_signal_only"})
                    continue
            elif is_paper:
                if not _paper_positive_anchor_ok(rule):
                    dropped.append({"rule": rule, "reason": "paper_yes_missing_strong_anchor"})
                    continue
            elif is_beer:
                if not _beer_positive_anchor_ok(rule):
                    dropped.append({"rule": rule, "reason": "beer_yes_missing_strong_anchor"})
                    continue
            else:
                if not _positive_anchor_ok(rule):
                    dropped.append({"rule": rule, "reason": "yes_missing_strong_anchor"})
                    continue
                if _has_feature(rule, "both_generic_name", "==", 1) and not _has_feature(rule, "inferred_code_exact", "==", 1):
                    dropped.append({"rule": rule, "reason": "generic_yes_forbidden"})
                    continue
        if decision == 0:
            if not _negative_anchor_ok(rule):
                dropped.append({"rule": rule, "reason": "no_rule_too_loose"})
                continue
        kept.append(rule)

    kept = sorted(
        kept,
        key=lambda r: (
            0 if int(r.get("decision", 0)) == 0 else 1,
            -int(r.get("priority", 0)),
            r.get("name", ""),
        ),
    )

    report = {
        "num_input_rules": len(rules),
        "num_kept_rules": len(kept),
        "num_dropped_rules": len(dropped),
        "dropped": dropped,
    }
    return kept, report


def _is_paper_profile(schema) -> bool:
    return getattr(schema, "feature_profile", None) == "paper"
def _is_beer_profile(schema) -> bool:
    return getattr(schema, "feature_profile", None) == "beer"
def _is_walmart_amazon(schema) -> bool:
    return getattr(schema, "dataset_name", "") == "walmart_amazon"


def _walmart_positive_anchor_ok(rule: Dict) -> bool:
    if _has_feature(rule, "code_exact", "==", 1):
        return True
    if _has_feature(rule, "strong_model_exact", "==", 1):
        return True
    if (
        _has_threshold(rule, "strong_model_overlap_count", 1, direction=">=")
        and _has_feature(rule, "code_exact", "==", 1)
    ):
        return True
    return False
def _paper_positive_anchor_ok(rule: Dict) -> bool:
    # 标题强 + 作者支持
    if _has_threshold(rule, "name_seq", 0.90, direction=">=") and (
        _has_threshold(rule, "party_overlap", 0.50, direction=">=")
        or _has_threshold(rule, "party_tokens_jaccard", 0.33, direction=">=")
        or _has_feature(rule, "party_exact", "==", 1)
    ):
        return True
    # 完全相同标题
    if _has_feature(rule, "name_exact", "==", 1) and (
        _has_threshold(rule, "party_overlap", 0.50, direction=">=")
        or _has_threshold(rule, "party_tokens_jaccard", 0.25, direction=">=")
    ):
        return True
    # 年份一致可以作为辅证，但不能是必须条件
    if _has_threshold(rule, "name_seq", 0.88, direction=">=") and _has_feature(rule, "time_equal", "==", 1):
        return True
    return False
def _beer_positive_anchor_ok(rule: Dict) -> bool:
    # Pattern 1:
    # high token overlap on beer name
    if _has_threshold(rule, "name_jaccard", 0.75, direction=">="):
        return True
    # Pattern 2:
    # decent seq + strong token overlap
    if (
        _has_threshold(rule, "name_seq", 0.78, direction=">=")
        and _has_threshold(rule, "name_jaccard", 0.67, direction=">=")
    ):
        return True
    # Pattern 3:
    # name similarity + brewery support
    if (
        _has_threshold(rule, "name_seq", 0.78, direction=">=")
        and _has_threshold(rule, "name_jaccard", 0.60, direction=">=")
        and _has_threshold(rule, "party_overlap", 0.50, direction=">=")
    ):
        return True
    # Pattern 4:
    # name token overlap + brewery token support
    if (
        _has_threshold(rule, "name_jaccard", 0.60, direction=">=")
        and (
            _has_threshold(rule, "party_overlap", 0.50, direction=">=")
            or _has_threshold(rule, "party_tokens_jaccard", 0.33, direction=">=")
        )
    ):
        return True
    return False



