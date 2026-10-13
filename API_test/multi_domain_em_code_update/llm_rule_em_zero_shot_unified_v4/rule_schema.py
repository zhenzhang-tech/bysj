from __future__ import annotations
from typing import Any, Dict, List

VALID_OPS = {"==", "!=", ">=", "<=", ">", "<"}

def validate_condition(cond: Dict[str, Any]) -> None:
    required = {"feature", "op", "value"}
    missing = required - set(cond.keys())
    if missing:
        raise ValueError(f"Condition missing fields: {missing}")
    if cond["op"] not in VALID_OPS:
        raise ValueError(f"Invalid op: {cond['op']}")

def validate_rule(rule: Dict[str, Any]) -> None:
    required = {"name", "priority", "decision", "all", "rationale"}
    missing = required - set(rule.keys())
    if missing:
        raise ValueError(f"Rule missing fields: {missing}")
    if rule["decision"] not in [0, 1]:
        raise ValueError("decision must be 0 or 1")
    if not isinstance(rule["all"], list) or len(rule["all"]) == 0:
        raise ValueError("all must be a non-empty list")
    for cond in rule["all"]:
        validate_condition(cond)

def validate_rules(rules: List[Dict[str, Any]]) -> None:
    if not isinstance(rules, list):
        raise ValueError("rules must be a list")
    for rule in rules:
        validate_rule(rule)
