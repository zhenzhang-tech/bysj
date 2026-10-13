from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PromptProfile:
    domain: str
    task: str
    focus_points: list[str]
    examples: list[tuple[str, str, str]]
    baseline_instructions: Optional[list[str]] = None
    gray_instructions: Optional[list[str]] = None


PROMPT_LIBRARY: dict[str, PromptProfile] = {
    "paper": PromptProfile(
        domain="academic papers",
        task="Determine whether the two records refer to the same real-world paper.",
        focus_points=["title", "authors", "venue", "year"],
        examples=[
            (
                "name: ajax : an extensible data cleaning tool; party: daniela florescu, helena galhardas; category: sigmod conference; time: 2000",
                "name: ajax : an extensible data cleaning tool; party: helena galhardas, daniela florescu; category: international conference on management of data; time: 2000",
                "Yes",
            ),
            (
                "name: watermarking relational data : framework , algorithms and analysis; party: peter j. haas, rakesh agrawal; time: 2003",
                "name: a system for watermarking relational databases; party: rakesh agrawal, peter j. haas; time: 2003",
                "No",
            ),
        ],
        baseline_instructions=[
            "Use title as the main signal.",
            "Use author overlap and year agreement as important supporting evidence.",
            "Reordered author names or minor formatting differences may still refer to the same paper.",
            "Generic titles require stronger corroboration.",
        ],
        gray_instructions=[
                "Treat title similarity as the primary evidence.",
                "Author list truncation, abbreviation, and reordering are common on scholar-like sources.",
                "Missing year or missing venue should not by itself imply No.",
                "If title is near-exact and authors strongly overlap, prefer Yes even when venue or year is incomplete.",
        ],
    ),
    "product": PromptProfile(
        domain="retail products",
        task="Determine whether the two records refer to the same real-world product.",
        focus_points=[
            "product title or name",
            "brand or manufacturer",
            "model identifier",
            "variant or edition",
            "price",
            "key specifications",
        ],
        examples=[
            (
                "name: panda internet security 2007 3-user; party: panda software; numeric: 69.95",
                "name: mcafee internet security suite 2007 3-user ( pc ); numeric: 89.99",
                "No",
            ),
            (
                "name: sims 2 glamour life stuff pack; party: aspyr media; numeric: 24.99",
                "name: aspyr media inc sims 2 glamour life stuff pack; numeric: 23.44",
                "Yes",
            ),
            (
                "name: sony cyber-shot dsc-w55 digital camera silver; party: sony; numeric: 179.99",
                "name: sony cybershot dsc w55 digital camera; party: sony; numeric: 183.99",
                "Yes",
            ),
            (
                "name: sandisk cruzer micro 2gb usb flash drive",
                "name: sandisk cruzer micro 4gb usb flash drive",
                "No",
            ),
        ],
        baseline_instructions=[
            "Use product title and model-related tokens as primary evidence.",
            "Brand or manufacturer is important when available, but missing brand on one side is not by itself evidence for No.",
            "Variant differences such as capacity, size, color, edition, pack count, or generation may indicate different products.",
            "Price is auxiliary evidence and should not be used alone.",
        ],
    ),
    "restaurant": PromptProfile(
        domain="restaurant records",
        task="Determine whether the two records refer to the same real-world entity.",
        focus_points=["restaurant name", "address", "city", "phone number", "cuisine type"],
        examples=[
            (
                "name: campanile; description: 624 s. la brea ave.; location: los angeles; code: 2139381447",
                "name: campanile; description: 624 south la brea avenue; location: los angeles; code: 2139381447",
                "Yes",
            ),
            (
                "name: ritz cafe; location: new york; code: 2125551212",
                "name: ritz cafe; location: chicago; code: 3125551212",
                "No",
            ),
        ],
    ),
    "beer": PromptProfile(
        domain="beer records",
        task="Determine whether the two records refer to the same real-world entity.",
        focus_points=["beer name", "brewery", "style", "ABV"],
        examples=[
            (
                "name: stone ipa; party: stone brewing; category: india pale ale; numeric: 6.9",
                "name: stone ipa; party: stone brewing co; category: ipa; numeric: 6.9",
                "Yes",
            ),
            (
                "name: pale ale; party: brewery a; category: pale ale; numeric: 5.0",
                "name: pale ale; party: brewery b; category: pale ale; numeric: 5.0",
                "No",
            ),
        ],
    ),
    "music": PromptProfile(
        domain="music records",
        task="Determine whether the two records refer to the same real-world entity.",
        focus_points=["song name", "artist", "album", "genre", "release date", "duration"],
        examples=[
            (
                "name: halo; party: beyonce; description: i am... sasha fierce; time: 2008-11-18",
                "name: halo; party: beyonce; description: i am sasha fierce; time: 2008-11-18",
                "Yes",
            ),
            (
                "name: stay; party: rihanna",
                "name: stay; party: zedd",
                "No",
            ),
        ],
    ),
    "generic": PromptProfile(
        domain="entity records",
        task="Determine whether the two records refer to the same real-world entity.",
        focus_points=["primary name", "supporting descriptive attributes", "codes", "dates", "numbers"],
        examples=[
            (
                "name: alpha 1000; party: acme; code: x1000",
                "name: alpha-1000; party: acme corp; code: x1000",
                "Yes",
            ),
            (
                "name: alpha 1000; party: acme",
                "name: beta 2000; party: acme",
                "No",
            ),
        ],
    ),
}


def _format_examples(examples: list[tuple[str, str, str]]) -> str:
    blocks = []
    for e1, e2, label in examples:
        blocks.append(f"Entity 1: {e1}\nEntity 2: {e2}\nMatch Decision: {label}")
    return "\n\n".join(blocks)


def _format_instruction_block(profile: PromptProfile) -> str:
    extra = profile.baseline_instructions or []
    if not extra:
        return ""
    return "Decision procedure:\n" + "\n".join(f"{idx}. {item}" for idx, item in enumerate(extra, start=1))


def build_baseline_prompt(
    profile: PromptProfile,
    entity1_desc: str,
    entity2_desc: str,
    lang: str = "en",
) -> str:
    del lang
    instruction_block = _format_instruction_block(profile)

    return f"""
You are an entity matching expert for {profile.domain}.
{profile.task}

Focus on: {', '.join(profile.focus_points)}.

{instruction_block}

Examples:
{_format_examples(profile.examples)}

Now judge the following pair:
Entity 1: {entity1_desc}
Entity 2: {entity2_desc}

Output format:
Match Decision: Yes
or
Match Decision: No
Only output one line.
""".strip()


def build_gray_prompt(
    profile: PromptProfile,
    entity1_desc: str,
    entity2_desc: str,
    pair_features: Dict,
    lang: str = "en",
) -> str:
    del lang

    # 基础 Prompt（无特征，仅文本描述）
    base_prompt = build_baseline_prompt(profile, entity1_desc, entity2_desc)

    # Gray 阶段额外指导语
    guidance = ""
    if profile.gray_instructions:
        guidance = "Gray-stage guidance:\n" + "\n".join(f"- {item}" for item in profile.gray_instructions)

    # 针对不同数据集调整特征展示顺序（Amazon-Google 重点突出 manufacturer 和 title 相关特征）
    if profile.domain == "retail products (weak/missing brand)":
        # 将品牌、强模型、名称相似度等特征优先展示
        preferred_order = [
            "brand_exact",
            "brand_overlap",
            "strong_model_exact",
            "strong_model_overlap_count",
            "name_seq",
            "name_exact",
            "name_jaccard",
            "name_overlap",
            "variant_conflict",
            "category_conflict",
            "price_conflict",      # 价格冲突放在靠后位置
            "numeric_rel_diff",
            "both_generic_name",
            "major_missing_count",
        ]
    else:
        preferred_order = [
            "brand_exact",
            "brand_overlap",
            "strong_model_exact",
            "strong_model_overlap_count",
            "variant_conflict",
            "price_conflict",
            "category_conflict",
        ]

    # 构造有序特征列表，确保所有传入特征都被包含
    ordered_keys = [k for k in preferred_order if k in pair_features] + \
                   [k for k in pair_features if k not in preferred_order]

    feature_lines = "\n".join(f"- {k}: {pair_features[k]}" for k in ordered_keys)

    return f"""
{base_prompt}

{guidance}

Additional structured evidence (auxiliary only, not independent proof):
{feature_lines}
""".strip()


