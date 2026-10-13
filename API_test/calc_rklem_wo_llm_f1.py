#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calc_rklem_wo_llm_f1.py

功能：
1. 计算 RKLEM w/o LLM 的 F1 分数。
2. 对 source == GRAY_LLM 的样本，强制视为匹配错误：
   - label == 0 时，将 pred 视为 1
   - label == 1 时，将 pred 视为 0
3. 对 source 以 RULE:: 开头，或 stage == RULE 的样本，保留原 pred。
4. 输出表格：
   Dataset, RKLEM w/o LLM, RKLEM

用法一：批量计算七个数据集
python calc_rklem_wo_llm_f1.py --result_root "D:/bysj/API_test/result" --run_tag "n1n_deepseekV3.2thinking" --output "rklem_wo_llm_f1.csv"

用法二：只计算单个 predictions.csv
python calc_rklem_wo_llm_f1.py --predictions "D:/bysj/API_test/result/abt_buy/n1n_deepseekV3.2thinking/predictions.csv" --dataset_abbr AB --output "single_result.csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DATASETS: List[Tuple[str, str]] = [
    ("abt_buy", "AB"),
    ("walmart_amazon", "WA"),
    ("dblp_acm", "DA"),
    ("dblp_scholar", "DS"),
    ("beer", "BE"),
    ("fodors_zagat", "FZ"),
    ("itunes_amazon", "IA"),
]


def to_binary_series(s: pd.Series) -> pd.Series:
    """将 label / pred 列统一转换为 0/1。"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "non-match": 0,
        "nonmatch": 0,
        "match": 1,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).astype("Int64")


def calc_confusion_and_f1(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """根据 TP、TN、FP、FN 手动计算 F1。"""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_gray_mask(df: pd.DataFrame) -> pd.Series:
    """
    判断哪些样本来自灰区 LLM 层。
    优先使用 source == GRAY_LLM；
    如果没有 source，则使用 stage == GRAY_LLM。
    """
    if "source" in df.columns:
        return df["source"].astype(str).str.upper().eq("GRAY_LLM")

    if "stage" in df.columns:
        return df["stage"].astype(str).str.upper().eq("GRAY_LLM")

    raise ValueError("predictions.csv 中缺少 source/stage 列，无法判断灰区 LLM 样本。")


def calc_one_file(predictions_path: Path, dataset_abbr: str) -> Dict[str, object]:
    """计算单个 predictions.csv 的 RKLEM w/o LLM F1 和完整 RKLEM F1。"""
    df = pd.read_csv(predictions_path)

    required_cols = {"label", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{predictions_path} 缺少必要列：{missing}")

    y_true = to_binary_series(df["label"])
    y_pred_full = to_binary_series(df["pred"])
    gray_mask = get_gray_mask(df)

    valid = y_true.notna() & y_pred_full.notna()
    y_true = y_true.loc[valid].astype(int)
    y_pred_full = y_pred_full.loc[valid].astype(int)
    gray_mask = gray_mask.loc[valid]

    # 完整 RKLEM：直接使用原始 pred
    full_metrics = calc_confusion_and_f1(y_true, y_pred_full)

    # RKLEM w/o LLM：
    # 对 GRAY_LLM 样本全部视为错误：
    # label=0 -> pred=1；label=1 -> pred=0
    # 对 RULE 样本保留原始 pred
    y_pred_wo_llm = y_pred_full.copy()
    y_pred_wo_llm.loc[gray_mask] = 1 - y_true.loc[gray_mask]

    wo_llm_metrics = calc_confusion_and_f1(y_true, y_pred_wo_llm)

    return {
        "Dataset": dataset_abbr,
        "RKLEM w/o LLM": round(wo_llm_metrics["f1"] * 100, 1),
        "RKLEM": round(full_metrics["f1"] * 100, 1),
        "TP": wo_llm_metrics["TP"],
        "TN": wo_llm_metrics["TN"],
        "FP": wo_llm_metrics["FP"],
        "FN": wo_llm_metrics["FN"],
        "Gray_LLM_Num": int(gray_mask.sum()),
        "Rule_Num": int((~gray_mask).sum()),
        "predictions_path": str(predictions_path),
    }


def calc_batch(result_root: Path, run_tag: str) -> List[Dict[str, object]]:
    """批量计算七个数据集。目录格式：result_root / dataset_name / run_tag / predictions.csv"""
    rows: List[Dict[str, object]] = []

    for dataset_name, dataset_abbr in DATASETS:
        pred_path = result_root / dataset_name / run_tag / "predictions.csv"
        if not pred_path.exists():
            print(f"[WARN] 未找到文件，已跳过：{pred_path}")
            continue

        rows.append(calc_one_file(pred_path, dataset_abbr))

    return rows


def save_main_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """保存论文画图用主表。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "RKLEM w/o LLM", "RKLEM"])
        for r in rows:
            writer.writerow([r["Dataset"], r["RKLEM w/o LLM"], r["RKLEM"]])

    print(f"[OK] 主表已保存：{output_path}")


def save_detail_csv(rows: List[Dict[str, object]], detail_path: Path) -> None:
    """保存包含 TP/TN/FP/FN 的详细表，便于检查。"""
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "Dataset",
        "RKLEM w/o LLM",
        "RKLEM",
        "TP",
        "TN",
        "FP",
        "FN",
        "Gray_LLM_Num",
        "Rule_Num",
        "predictions_path",
    ]

    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})

    print(f"[OK] 详细表已保存：{detail_path}")


def print_table(rows: List[Dict[str, object]]) -> None:
    """在终端输出表格。"""
    print()
    print("Dataset\tRKLEM w/o LLM\tRKLEM")
    for r in rows:
        print(f"{r['Dataset']}\t{r['RKLEM w/o LLM']}\t{r['RKLEM']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, default=None, help="结果根目录，例如 D:/bysj/API_test/result")
    parser.add_argument("--run_tag", type=str, default="n1n_deepseekV3.2thinking", help="RKLEM 方法对应的结果文件夹名")
    parser.add_argument("--predictions", type=str, default=None, help="单个 predictions.csv 路径")
    parser.add_argument("--dataset_abbr", type=str, default="AB", help="单文件模式下的数据集简称，例如 AB/FZ")
    parser.add_argument("--output", type=str, default="rklem_wo_llm_f1.csv", help="输出主表 CSV")
    parser.add_argument("--detail_output", type=str, default="rklem_wo_llm_f1_detail.csv", help="输出详细 CSV，包含 TP/TN/FP/FN")
    args = parser.parse_args()

    if args.predictions:
        rows = [calc_one_file(Path(args.predictions), args.dataset_abbr)]
    elif args.result_root:
        rows = calc_batch(Path(args.result_root), args.run_tag)
    else:
        raise ValueError("请提供 --result_root 或 --predictions。")

    if not rows:
        raise ValueError("没有成功读取任何 predictions.csv，请检查路径和 run_tag。")

    print_table(rows)
    save_main_csv(rows, Path(args.output))
    save_detail_csv(rows, Path(args.detail_output))


if __name__ == "__main__":
    main()
