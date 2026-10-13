from __future__ import annotations

import json
from typing import Dict, List, Optional, Set


PRODUCT_BASE_FEATURES = {
    "name_exact",
    "name_seq",
    "name_jaccard",
    "name_overlap",
    "desc_seq",
    "desc_jaccard",
    "desc_overlap",
    "numeric_equal",
    "numeric_rel_diff",
    "shared_numeric_tokens",
    "both_generic_name",
    "major_missing_count",
    "inferred_code_exact",
    "inferred_code_overlap_count",
    "inferred_code_jaccard",
    "brand_exact",
    "brand_overlap",
    "strong_model_exact",
    "strong_model_overlap_count",
    "variant_conflict",
    "price_conflict",
    "category_conflict",
}

PAPER_BASE_FEATURES = {
    "name_exact",
    "name_seq",
    "name_jaccard",
    "name_overlap",
    "party_exact",
    "party_overlap",
    "party_tokens_jaccard",
    "category_exact",
    "category_seq",
    "time_equal",
    "both_generic_name",
    "major_missing_count",
}


RESTAURANT_BASE_FEATURES = {
    "name_exact",
    "name_seq",
    "desc_seq",
    "location_seq",
    "code_exact",
    "category_exact",
    "major_missing_count",
}

MUSIC_BASE_FEATURES = {
    "name_exact",
    "name_seq",
    "party_exact",
    "party_overlap",
    "category_exact",
    "time_equal",
    "numeric_rel_diff",
    "major_missing_count",
}
BEER_BASE_FEATURES = {
    "name_exact",
    "name_seq",
    "name_jaccard",
    "name_overlap",
    "party_exact",
    "party_overlap",
    "party_tokens_jaccard",
    "category_exact",
    "category_seq",
    "numeric_equal",
    "numeric_rel_diff",
    "shared_numeric_tokens",
    "major_missing_count",
}

def _feature_profile_from_schema(schema) -> str:
    return getattr(schema, "feature_profile", None) or getattr(schema, "domain", "generic")


def allowed_features_by_schema(schema) -> Set[str]:
    profile = _feature_profile_from_schema(schema)
    left_map = getattr(schema, "left_map", {})
    right_map = getattr(schema, "right_map", {})

    def has_field(field: str) -> bool:
        return bool(left_map.get(field) or right_map.get(field))

    if profile == "product":
        feats = set(PRODUCT_BASE_FEATURES)
        if has_field("code"):
            feats.update({"code_exact", "code_token_overlap"})
        if not has_field("description"):
            feats -= {"desc_seq", "desc_jaccard", "desc_overlap"}
        if not has_field("numeric"):
            feats -= {"numeric_equal", "numeric_rel_diff", "shared_numeric_tokens", "price_conflict"}
        if not has_field("party"):
            feats -= {"brand_exact", "brand_overlap"}
        if not has_field("category"):
            feats -= {"category_conflict"}
        return feats

    if profile == "paper":
        feats = set(PAPER_BASE_FEATURES)
        if has_field("description"):
            feats.update({"desc_seq", "desc_jaccard"})
        return feats

    if profile == "restaurant":
        return set(RESTAURANT_BASE_FEATURES)

    if profile == "music":
        return set(MUSIC_BASE_FEATURES)

    return {
        "name_exact",
        "name_seq",
        "name_jaccard",
        "desc_seq",
        "numeric_rel_diff",
        "major_missing_count",
    }


def _clean_rule(rule: Dict, allowed_features: Set[str]) -> Optional[Dict]:
    if not isinstance(rule, dict):
        return None
    if "all" not in rule or not isinstance(rule["all"], list):
        return None

    cleaned_conditions = []
    for cond in rule["all"]:
        if not isinstance(cond, dict):
            continue
        feat = cond.get("feature")
        if feat not in allowed_features:
            continue
        if "op" not in cond or "value" not in cond:
            continue
        cleaned_conditions.append(
            {
                "feature": feat,
                "op": cond["op"],
                "value": cond["value"],
            }
        )

    if not cleaned_conditions:
        return None

    return {
        "name": rule.get("name", "unnamed_rule"),
        "priority": int(rule.get("priority", 50)),
        "decision": int(rule.get("decision", 0)),
        "all": cleaned_conditions,
    }


def _parse_llm_rule_json(text: str, allowed_features: Set[str]) -> List[Dict]:
    if not text:
        return []

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    try:
        raw = json.loads(text)
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    cleaned = []
    for item in raw:
        rule = _clean_rule(item, allowed_features)
        if rule is not None:
            cleaned.append(rule)
    return cleaned


def _feature_range_text(feature_stats: Optional[Dict[str, Dict]]) -> str:
    if not feature_stats:
        return "No feature range statistics available."
    lines = []
    for feat, stats in sorted(feature_stats.items()):
        if not isinstance(stats, dict):
            continue
        parts = []
        for k in ["min", "p50", "p90", "p95", "max"]:
            if k in stats:
                parts.append(f"{k}={stats[k]}")
        if parts:
            lines.append(f"- {feat}: " + ", ".join(parts))
    return "\n".join(lines) if lines else "No feature range statistics available."


def build_generator_prompt(schema, feature_stats: Optional[Dict[str, Dict]] = None, lang: str = "en") -> str:
    profile = _feature_profile_from_schema(schema)
    allowed = sorted(allowed_features_by_schema(schema))
    feature_ranges = _feature_range_text(feature_stats)

    if getattr(schema, "dataset_name", "") == "walmart_amazon":
        prompt = f"""
    You are an entity matching rule design expert.
    Target Dataset: Walmart-Amazon (Columns: title, brand, category, modelno, price).

    Allowed features:
    {allowed}

    Observed feature ranges:
    {feature_ranges}

    Requirements:
    1. Output ONLY a valid JSON array of rule objects. Do NOT output explanations or markdown.
    2. Generate conservative but useful product matching rules.
    3. For this dataset, model number / code evidence is the primary positive anchor.
    4. Brand alone is NOT sufficient for a positive rule.
    5. Moderate title similarity with the same brand is NOT sufficient for a positive rule.
    6. Prefer positive rules that use:
       - code_exact
       - strong_model_exact
       - strong_model_overlap_count together with code/model support
    7. Use price and category mainly as protective negative evidence.
    8. If brand is the same but code/model evidence is absent, prefer No or leave to gray rather than Yes.
    9. Include both Yes and No rules.
    10. Avoid optimistic Yes rules such as:
       - brand_exact + moderate name similarity
       - same category + same brand + weak title similarity

    Return JSON array only.
    """
        return prompt.strip()
    # 下方保留原本针对其他数据集的逻辑，不做任何修改
    special_guidance = ""
    if profile == "paper":
        prompt = f"""
    You are an entity matching rule design expert.
    Target Dataset: {schema.dataset_name} (academic papers).

    Allowed features:
    {allowed}

    Observed feature ranges:
    {feature_ranges}

    Requirements:
    1. Output ONLY a valid JSON array of rule objects.
    2. Generate conservative but useful paper matching rules.
    3. Use title similarity as the primary signal.
    4. Use author overlap as the main supporting evidence.
    5. Year agreement is useful support, but missing or unequal year must NOT automatically block a positive rule.
    6. Venue/category is noisy on dblp_scholar and should be treated as weak auxiliary evidence only.
    7. Scholar-side author lists are often truncated or abbreviated.
    8. Prefer conservative positive rules such as:
       - near-exact title + strong author support, even when year is missing
       - strong title + moderate author overlap + moderate token overlap on authors
       - exact title + partial author support when one side has missing fields
    9. Negative rules should focus on clearly weak title similarity with weak author evidence.
    10. Include both Yes and No rules.
    11. Avoid optimistic Yes rules for generic titles.

    Return JSON array only.
    """
        return prompt.strip()

    if profile == "product":
        special_guidance = """
For product matching:
- Prefer brand and strong model anchors for positive rules.
- strong_model_exact or strong_model_overlap_count should be treated as stronger than generic text similarity.
- variant_conflict, price_conflict, and category_conflict should be used to protect positive rules.
- Do NOT rely on shared_numeric_tokens alone for positive rules.
- Avoid rules that only use name_seq or desc_seq without a brand/model anchor.
"""

    prompt = f"""
You are generating high-precision entity matching rules.

Current profile: {profile}

Allowed features:
{allowed}

Observed feature ranges:
{feature_ranges}

{special_guidance}

Requirements:
1. Return a JSON array only.
2. Each rule must have fields: name, priority, decision, all.
3. decision: 1 means match, 0 means non-match.
4. Each condition in "all" must have: feature, op, value.
5. Do not use any feature outside the allowed list.
6. Do not propose impossible thresholds beyond the observed ranges.
7. Prefer rules that can cover easy cases safely.
8. For product matching, brand_exact, strong_model_exact, and strong_model_overlap_count are preferred positive anchors.

Example format:
[
  {{
    "name": "Example Rule",
    "priority": 90,
    "decision": 1,
    "all": [
      {{"feature": "strong_model_overlap_count", "op": ">=", "value": 1}},
      {{"feature": "variant_conflict", "op": "==", "value": 0}}
    ]
  }}
]
"""
    return prompt.strip()


def fallback_seed_rules(schema) -> List[Dict]:
    profile = _feature_profile_from_schema(schema)
    allowed = allowed_features_by_schema(schema)
    rules: List[Dict] = []
    if getattr(schema, "dataset_name", "") == "walmart_amazon":
        candidate_rules = [
            {
                "name": "Walmart Amazon Exact Model Code Match",
                "priority": 100,
                "decision": 1,
                "all": [
                    {"feature": "code_exact", "op": "==", "value": 1},
                ],
            },
            {
                "name": "Walmart Amazon Brand And Exact Model Code",
                "priority": 98,
                "decision": 1,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "code_exact", "op": "==", "value": 1},
                ],
            },
            {
                "name": "Walmart Amazon Strong Model Overlap With Exact Code",
                "priority": 96,
                "decision": 1,
                "all": [
                    {"feature": "strong_model_overlap_count", "op": ">=", "value": 1},
                    {"feature": "code_exact", "op": "==", "value": 1},
                ],
            },
            {
                "name": "Walmart Amazon Same Brand Weak Model Evidence",
                "priority": 130,
                "decision": 0,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "code_exact", "op": "==", "value": 0},
                    {"feature": "name_jaccard", "op": "<", "value": 0.50},
                    {"feature": "shared_numeric_tokens", "op": "==", "value": 0},
                ],
            },
            {
                "name": "Walmart Amazon Price Conflict Without Code Match",
                "priority": 128,
                "decision": 0,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "code_exact", "op": "==", "value": 0},
                    {"feature": "price_conflict", "op": "==", "value": 1},
                    {"feature": "name_overlap", "op": "<", "value": 0.80},
                ],
            },
            {
                "name": "Walmart Amazon Category Conflict Weak Match",
                "priority": 124,
                "decision": 0,
                "all": [
                    {"feature": "category_conflict", "op": "==", "value": 1},
                    {"feature": "strong_model_overlap_count", "op": "==", "value": 0},
                    {"feature": "code_exact", "op": "==", "value": 0},
                    {"feature": "name_seq", "op": "<", "value": 0.65},
                ],
            },
        ]
    elif profile == "product":
        candidate_rules = [
            {
                "name": "Brand and Strong Model Match",
                "priority": 100,
                "decision": 1,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "strong_model_exact", "op": "==", "value": 1},
                    {"feature": "variant_conflict", "op": "==", "value": 0},
                ],
            },
            {
                "name": "Strong Model Overlap with Name Support",
                "priority": 96,
                "decision": 1,
                "all": [
                    {"feature": "strong_model_overlap_count", "op": ">=", "value": 1},
                    {"feature": "name_seq", "op": ">=", "value": 0.45},
                    {"feature": "variant_conflict", "op": "==", "value": 0},
                ],
            },
            {
                "name": "Brand Model Match without Price Conflict",
                "priority": 95,
                "decision": 1,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "strong_model_overlap_count", "op": ">=", "value": 1},
                    {"feature": "price_conflict", "op": "==", "value": 0},
                ],
            },
            {
                "name": "Brand with Moderate Name Support",
                "priority": 92,
                "decision": 1,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 1},
                    {"feature": "name_seq", "op": ">=", "value": 0.60},
                    {"feature": "variant_conflict", "op": "==", "value": 0}
                ],
            },
            {
                "name": "Brand Conflict",
                "priority": 130,
                "decision": 0,
                "all": [
                    {"feature": "brand_exact", "op": "==", "value": 0},
                    {"feature": "strong_model_overlap_count", "op": "==", "value": 0},
                    {"feature": "name_seq", "op": "<", "value": 0.55},
                ],
            },
            {
                "name": "Variant Conflict",
                "priority": 128,
                "decision": 0,
                "all": [
                    {"feature": "strong_model_overlap_count", "op": ">=", "value": 1},
                    {"feature": "variant_conflict", "op": "==", "value": 1},
                ],
            },
            {
                "name": "Category Conflict with Weak Match",
                "priority": 122,
                "decision": 0,
                "all": [
                    {"feature": "category_conflict", "op": "==", "value": 1},
                    {"feature": "strong_model_overlap_count", "op": "==", "value": 0},
                    {"feature": "name_seq", "op": "<", "value": 0.65},
                ],
            },
            {
                "name": "Price Conflict and Weak Model Evidence",
                "priority": 119,
                "decision": 0,
                "all": [
                    {"feature": "price_conflict", "op": "==", "value": 1},
                    {"feature": "strong_model_overlap_count", "op": "==", "value": 0},
                    {"feature": "name_seq", "op": "<", "value": 0.60},
                ],
            },
        ]
    elif profile == "beer":
        candidate_rules = [
            {
                "name": "Beer Weak Name And No ABV Support",
                "priority": 120,
                "decision": 0,
                "all": [
                    {"feature": "name_seq", "op": "<=", "value": 0.62},
                    {"feature": "numeric_equal", "op": "==", "value": 0},
                ],
            },
            {
                "name": "Beer Weak Name And Weak Brewery",
                "priority": 118,
                "decision": 0,
                "all": [
                    {"feature": "name_seq", "op": "<=", "value": 0.65},
                    {"feature": "party_overlap", "op": "<=", "value": 0.50},
                ],
            },
            {
                "name": "Beer Strong Name Token Match",
                "priority": 98,
                "decision": 1,
                "all": [
                    {"feature": "name_jaccard", "op": ">=", "value": 0.75},
                ],
            },
            {
                "name": "Beer High Name Similarity With Token Support",
                "priority": 96,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.78},
                    {"feature": "name_jaccard", "op": ">=", "value": 0.67},
                ],
            },
            {
                "name": "Beer Strong Name With Brewery Support",
                "priority": 95,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.78},
                    {"feature": "name_jaccard", "op": ">=", "value": 0.60},
                    {"feature": "party_overlap", "op": ">=", "value": 0.50},
                ],
            },
        ]
    elif profile == "paper":
        candidate_rules = [
            {
                "name": "Paper Near Exact Title with Strong Author Support",
                "priority": 100,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.96},
                    {"feature": "party_overlap", "op": ">=", "value": 0.60},
                ],
            },
            {
                "name": "Paper Strong Title and Moderate Author Support",
                "priority": 98,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.90},
                    {"feature": "party_overlap", "op": ">=", "value": 0.67},
                    {"feature": "party_tokens_jaccard", "op": ">=", "value": 0.33},
                ],
            },
            {
                "name": "Paper Exact Title Under Missing Fields",
                "priority": 96,
                "decision": 1,
                "all": [
                    {"feature": "name_exact", "op": "==", "value": 1},
                    {"feature": "party_overlap", "op": ">=", "value": 0.50},
                    {"feature": "major_missing_count", "op": ">=", "value": 1},
                ],
            },
            {
                "name": "Paper Strong Title Author Year Agreement",
                "priority": 94,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.88},
                    {"feature": "party_overlap", "op": ">=", "value": 0.50},
                    {"feature": "time_equal", "op": "==", "value": 1},
                ],
            },
            {
                "name": "Paper Weak Title and Weak Author",
                "priority": 130,
                "decision": 0,
                "all": [
                    {"feature": "name_seq", "op": "<", "value": 0.45},
                    {"feature": "party_overlap", "op": "<", "value": 0.20},
                ],
            },
            {
                "name": "Paper Weak Title Despite Same Year",
                "priority": 126,
                "decision": 0,
                "all": [
                    {"feature": "name_seq", "op": "<", "value": 0.35},
                    {"feature": "time_equal", "op": "==", "value": 1},
                ],
            },
        ]

    else:
        candidate_rules = [
            {
                "name": "Generic Strong Name",
                "priority": 90,
                "decision": 1,
                "all": [
                    {"feature": "name_seq", "op": ">=", "value": 0.90},
                ],
            },
            {
                "name": "Generic Weak Name",
                "priority": 120,
                "decision": 0,
                "all": [
                    {"feature": "name_seq", "op": "<", "value": 0.40},
                ],
            },
        ]

    for rule in candidate_rules:
        cleaned = _clean_rule(rule, allowed)
        if cleaned is not None:
            rules.append(cleaned)

    return rules


def propose_rules(
    generator_client,
    stats: Optional[Dict[str, Dict]],
    schema,
    lang: str = "en",
) -> List[Dict]:
    allowed = allowed_features_by_schema(schema)
    prompt = build_generator_prompt(schema, feature_stats=stats, lang=lang)

    llm_rules: List[Dict] = []
    if generator_client is not None:
        try:
            raw = generator_client.chat([{"role": "user", "content": prompt}])
            llm_rules = _parse_llm_rule_json(raw, allowed)
        except Exception:
            llm_rules = []

    fallback = fallback_seed_rules(schema)

    merged: List[Dict] = []
    seen = set()

    for rule in llm_rules + fallback:
        key = (
            rule["name"],
            rule["decision"],
            tuple((c["feature"], c["op"], str(c["value"])) for c in rule["all"]),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(rule)

    return merged
