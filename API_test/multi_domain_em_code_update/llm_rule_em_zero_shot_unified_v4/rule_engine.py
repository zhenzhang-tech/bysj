from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

def _compare(lhs: Any, op: str, rhs: Any) -> bool:
    if op == "==": return lhs == rhs
    if op == "!=": return lhs != rhs
    if op == ">=": return lhs >= rhs
    if op == "<=": return lhs <= rhs
    if op == ">": return lhs > rhs
    if op == "<": return lhs < rhs
    raise ValueError(f"Unsupported op: {op}")

def rule_matches(row: pd.Series, rule: Dict[str, Any]) -> bool:
    for cond in rule["all"]:
        feature = cond["feature"]
        op = cond["op"]
        value = cond["value"]
        lhs = row.get(feature, None)
        if lhs is None or not _compare(lhs, op, value):
            return False
    return True

def apply_rules_to_row(row: pd.Series, rules: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    no_rules = sorted([r for r in rules if r["decision"] == 0], key=lambda x: (-x["priority"], x["name"]))
    yes_rules = sorted([r for r in rules if r["decision"] == 1], key=lambda x: (-x["priority"], x["name"]))
    for rule in no_rules:
        if rule_matches(row, rule):
            return rule["decision"], rule["name"]
    for rule in yes_rules:
        if rule_matches(row, rule):
            return rule["decision"], rule["name"]
    return None, None
