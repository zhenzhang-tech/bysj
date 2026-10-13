from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .dataset_schema import resolve_schema
from .features import build_entity_desc_from_canonical, compare_records, load_dataset, row_to_canonical
from .gray_matcher import judge_gray_pair
from .llm_client import LLMConfig, build_client
from .prompt_templates import PROMPT_LIBRARY
from .rule_engine import apply_rules_to_row
from .rule_generator import fallback_seed_rules, propose_rules
from .rule_sanity import sanitize_rules
from .rule_schema import validate_rules



def summarize_feature_stats_unlabeled_from_pairs(table_a, table_b, pairs, schema) -> Dict:
    a_id_col, b_id_col = table_a.columns[0], table_b.columns[0]
    rows = []
    for row in pairs.itertuples(index=False):
        l_id = str(getattr(row, "ltable_id"))
        r_id = str(getattr(row, "rtable_id"))
        a_row = table_a[table_a[a_id_col] == l_id].iloc[0]
        b_row = table_b[table_b[b_id_col] == r_id].iloc[0]
        left = row_to_canonical(a_row, schema, "left")
        right = row_to_canonical(b_row, schema, "right")
        rows.append(compare_records(left, right, schema))
    df = pd.DataFrame(rows)
    stats = {"num_pairs": int(len(df))}
    for col in df.columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "p50": float(df[col].quantile(0.5)),
            "p90": float(df[col].quantile(0.9)),
            "p95": float(df[col].quantile(0.95)),
        }
    return stats



def compute_final_metrics(preds: List[int], golds: List[int]) -> Dict:
    return {
        "num_cases": len(golds),
        "accuracy": float(accuracy_score(golds, preds)),
        "precision": float(precision_score(golds, preds, zero_division=0)),
        "recall": float(recall_score(golds, preds, zero_division=0)),
        "f1": float(f1_score(golds, preds, zero_division=0)),
    }



def run_full_pipeline(
    table_a_path: str,
    table_b_path: str,
    pairs_path: str,
    save_dir: str,
    generator_cfg: LLMConfig,
    gray_cfg: LLMConfig,
    dataset_name: str = "auto",
    lang: str = "en",
    desc_style: str = "kv",
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    table_a, table_b, pairs = load_dataset(table_a_path, table_b_path, pairs_path)
    schema = resolve_schema(dataset_name, table_a.columns, table_b.columns)

    raw_rules = fallback_seed_rules(schema)
    stats = {}
    try:
        stats = summarize_feature_stats_unlabeled_from_pairs(table_a, table_b, pairs, schema)
        raw_rules = propose_rules(build_client(generator_cfg), stats, schema, lang=lang)
        validate_rules(raw_rules)
    except Exception:
        raw_rules = fallback_seed_rules(schema)

    final_rules, sanity_report = sanitize_rules(raw_rules, schema, stats=stats)
    gray_client = build_client(gray_cfg)
    a_id_col, b_id_col = table_a.columns[0], table_b.columns[0]

    preds, golds, outputs = [], [], []
    rule_hits, gray_hits = 0, 0
    rule_hit_counter: Dict[str, int] = {}

    for row in pairs.itertuples(index=False):
        l_id = str(getattr(row, "ltable_id"))
        r_id = str(getattr(row, "rtable_id"))
        gold = int(getattr(row, "label"))
        a_row = table_a[table_a[a_id_col] == l_id].iloc[0]
        b_row = table_b[table_b[b_id_col] == r_id].iloc[0]
        left = row_to_canonical(a_row, schema, "left")
        right = row_to_canonical(b_row, schema, "right")
        comp = compare_records(left, right, schema)
        entity1_desc = build_entity_desc_from_canonical(left, schema.desc_order)
        entity2_desc = build_entity_desc_from_canonical(right, schema.desc_order)

        pred_rule, rule_src = apply_rules_to_row(pd.Series(comp), final_rules)
        if pred_rule is not None:
            pred = int(pred_rule)
            source = f"RULE::{rule_src}"
            stage = "RULE"
            rule_name = rule_src
            model_answer = ""
            rule_hits += 1
            rule_hit_counter[rule_src] = rule_hit_counter.get(rule_src, 0) + 1
        else:
            try:
                model_answer, pred_gray = judge_gray_pair(gray_client, schema, entity1_desc, entity2_desc, comp, lang=lang)
            except Exception as exc:
                model_answer, pred_gray = f"gray model call failed: {exc}", 0
            pred = 0 if pred_gray is None else int(pred_gray)
            source = "GRAY_LLM"
            stage = "GRAY_LLM"
            rule_name = ""
            gray_hits += 1

        preds.append(pred)
        golds.append(gold)
        outputs.append(
            {
                "ltable_id": l_id,
                "rtable_id": r_id,
                "label": gold,
                "pred": pred,
                "source": source,
                "model_answer": model_answer,
                "entity1_desc": entity1_desc,
                "entity2_desc": entity2_desc,
                "rule_name": rule_name,
                "stage": stage,
                "feature_profile": schema.feature_profile,
                **comp,
            }
        )

    total_time = time.time() - t0
    metrics = {
        "dataset": dataset_name,
        "method": "strict_zero_shot_rule_gray_llm_unified_v4",
        "model_generator": generator_cfg.model,
        "model_gray": gray_cfg.model,
        **compute_final_metrics(preds, golds),
        "total_time_sec": round(float(total_time), 6),
        "avg_time_per_case_sec": round(float(total_time / len(golds)), 6) if golds else 0.0,
        "rule_coverage": float(rule_hits / len(golds)) if golds else 0.0,
        "gray_ratio": float(gray_hits / len(golds)) if golds else 0.0,
        "num_final_rules": len(final_rules),
    }

    run_config = {
        "dataset": dataset_name,
        "method": "strict_zero_shot_rule_gray_llm_unified_v4",
        "strict_zero_shot": True,
        "generator_provider": generator_cfg.provider,
        "generator_model": generator_cfg.model,
        "gray_provider": gray_cfg.provider,
        "gray_model": gray_cfg.model,
        "schema_mode": "registry" if dataset_name in {"dblp_acm", "dblp_scholar", "abt_buy", "walmart_amazon", "beer", "fodors_zagat", "itunes_amazon"} else "auto",
        "domain": schema.domain,
        "feature_profile": schema.feature_profile,
        "desc_style": desc_style,
        "lang": lang,
        "desc_order": schema.desc_order,
        "schema": {
            "left_map": schema.left_map,
            "right_map": schema.right_map,
            "numeric_kind": schema.numeric_kind,
            "time_kind": schema.time_kind,
            "code_kind": schema.code_kind,
        },
        "table_a_path": table_a_path,
        "table_b_path": table_b_path,
        "pairs_path": pairs_path,
        "save_dir": save_dir,
    }

    coverage_summary = {
        "num_cases": len(golds),
        "rule_hits": rule_hits,
        "gray_hits": gray_hits,
        "rule_coverage": float(rule_hits / len(golds)) if golds else 0.0,
        "gray_ratio": float(gray_hits / len(golds)) if golds else 0.0,
        "rule_hit_counter": rule_hit_counter,
    }

    pred_df = pd.DataFrame(outputs)
    error_df = pred_df[pred_df["pred"] != pred_df["label"]][
        [
            "ltable_id",
            "rtable_id",
            "label",
            "pred",
            "source",
            "model_answer",
            "entity1_desc",
            "entity2_desc",
            "rule_name",
            "stage",
            "feature_profile",
        ]
    ].copy()

    (save_path / "generated_rules_raw.json").write_text(json.dumps(raw_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_path / "final_rules.json").write_text(json.dumps(final_rules, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_path / "rule_sanity_report.json").write_text(json.dumps(sanity_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_path / "rule_coverage_summary.json").write_text(json.dumps(coverage_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_path / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_df.to_csv(save_path / "predictions.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(save_path / "error_analysis.csv", index=False, encoding="utf-8-sig")
    return metrics
