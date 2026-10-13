from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from .dataset_schema import DatasetSchema
from .llm_client import BaseLLMClient
from .prompt_templates import PROMPT_LIBRARY, build_gray_prompt


def parse_model_answer(answer: str) -> Optional[int]:
    yes_pattern = re.compile(r"Match Decision:\s*(Yes|是)\b", re.I)
    no_pattern = re.compile(r"Match Decision:\s*(No|否)\b", re.I)
    if yes_pattern.search(answer):
        return 1
    if no_pattern.search(answer):
        return 0
    ans = answer.strip().lower()
    if ans in {"yes", "match decision: yes"}:
        return 1
    if ans in {"no", "match decision: no"}:
        return 0
    return None



def judge_gray_pair(client: BaseLLMClient, schema: DatasetSchema, entity1_desc: str, entity2_desc: str, pair_features: Dict, lang: str = "en") -> Tuple[str, Optional[int]]:
    profile = PROMPT_LIBRARY.get(schema.feature_profile, PROMPT_LIBRARY["generic"])
    answer = client.chat([
        {"role": "user", "content": build_gray_prompt(profile, entity1_desc, entity2_desc, pair_features, lang=lang)}
    ]).strip()
    return answer, parse_model_answer(answer)
