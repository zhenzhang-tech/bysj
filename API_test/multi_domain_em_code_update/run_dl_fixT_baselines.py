import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "abt_buy",
    "beer",
    "dblp_acm",
    "dblp_scholar",
    "fodors_zagat",
    "itunes_amazon",
    "walmart_amazon",
]

MODEL_REGISTRY = {
    "bert": r"D:\bysj\API_test\hf_models\bert-base-uncased",
    "roberta": r"D:\bysj\API_test\hf_models\roberta-base",
    "albert": r"D:\bysj\API_test\hf_models\albert-base-v2",
    "distilbert": r"D:\bysj\API_test\hf_models\distilbert-base-uncased",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return re.sub(r"\s+", " ", s)


def id_index(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("tableA/tableB must contain an 'id' column.")
    out = df.copy()
    out["_id_key"] = out["id"].astype(str)
    return out.set_index("_id_key", drop=False)


def infer_pair_columns(pairs: pd.DataFrame) -> Tuple[str, str, str]:
    left_candidates = ["ltable_id", "left_id", "id_left", "tableA_id", "A_id"]
    right_candidates = ["rtable_id", "right_id", "id_right", "tableB_id", "B_id"]
    label_candidates = ["label", "gold", "target", "match"]

    left_col = next((c for c in left_candidates if c in pairs.columns), None)
    right_col = next((c for c in right_candidates if c in pairs.columns), None)
    label_col = next((c for c in label_candidates if c in pairs.columns), None)

    if left_col is None or right_col is None:
        raise ValueError("Pair file must contain left/right id columns, e.g. ltable_id and rtable_id.")
    if label_col is None:
        raise ValueError("Pair file must contain a label column.")
    return left_col, right_col, label_col


def row_to_record_text(row: pd.Series, side_name: str = "Record", max_field_chars: int = 120) -> str:
    parts = []
    for col, val in row.items():
        if col in {"_id_key"} or col.lower() == "id":
            continue
        v = clean_value(val)
        if not v:
            continue
        if max_field_chars and len(v) > max_field_chars:
            v = v[:max_field_chars] + "..."
        parts.append(f"{col}: {v}")
    return f"{side_name}: " + " ; ".join(parts)


def build_pair_texts(
    pairs: pd.DataFrame,
    table_a_idx: pd.DataFrame,
    table_b_idx: pd.DataFrame,
    max_field_chars: int = 120,
):
    left_col, right_col, label_col = infer_pair_columns(pairs)
    left_texts, right_texts, labels, metas = [], [], [], []
    missing = 0

    for pair_idx, r in pairs.iterrows():
        lid = clean_value(r[left_col])
        rid = clean_value(r[right_col])

        if lid not in table_a_idx.index or rid not in table_b_idx.index:
            missing += 1
            continue

        left_row = table_a_idx.loc[lid]
        right_row = table_b_idx.loc[rid]

        left_text = row_to_record_text(left_row, "L", max_field_chars=max_field_chars)
        right_text = row_to_record_text(right_row, "R", max_field_chars=max_field_chars)
        label = int(float(clean_value(r[label_col])))

        left_texts.append(left_text)
        right_texts.append(right_text)
        labels.append(label)
        metas.append({
            "pair_index": int(pair_idx),
            "ltable_id": lid,
            "rtable_id": rid,
            "label": label,
            "left_text": left_text,
            "right_text": right_text,
        })

    if missing:
        print(f"[Warning] skipped {missing} pairs because table ids were missing.")

    return left_texts, right_texts, labels, metas


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(preds, dtype=int)

    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())

    accuracy = float((tp + tn) / len(y)) if len(y) else math.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "num_cases": int(len(y)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def mean_pooling(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, device, batch_size: int, max_length: int):
    import torch
    from torch.nn import functional as F

    model.eval()
    all_vecs = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)

            # Use mean pooling for all models to keep the four baselines comparable.
            vec = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
            vec = F.normalize(vec, p=2, dim=1)

            all_vecs.append(vec.detach().cpu().numpy())

    return np.vstack(all_vecs) if all_vecs else np.empty((0, 0), dtype=float)


def run_one_dataset(args, method: str, dataset: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = args.model_name or MODEL_REGISTRY[method]
    method_name = args.method_name or f"{method}_fixT_baseline"

    dataset_dir = Path(args.data_root) / dataset
    table_a_path = dataset_dir / args.table_a
    table_b_path = dataset_dir / args.table_b
    test_path = dataset_dir / args.test_pairs

    print(f"\n[Dataset] {dataset}")
    print(f"[Method] {method}")
    print(f"[Backbone] {model_name}")
    print(f"[Mode] no training, cosine similarity > {args.threshold} => match")
    print(f"[Data] tableA: {table_a_path}")
    print(f"[Data] tableB: {table_b_path}")
    print(f"[Data] test:  {test_path}")

    save_dir = Path(args.save_root) /dataset / method_name
    ensure_dir(save_dir)

    table_a = id_index(read_csv(table_a_path))
    table_b = id_index(read_csv(table_b_path))
    test_pairs = read_csv(test_path)

    if args.limit and args.limit > 0:
        test_pairs = test_pairs.head(args.limit)

    left_texts, right_texts, labels, metas = build_pair_texts(
        test_pairs,
        table_a,
        table_b,
        max_field_chars=args.max_field_chars,
    )

    if not labels:
        raise ValueError(f"No valid test pairs for dataset {dataset}.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    left_vecs = encode_texts(
        left_texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    right_vecs = encode_texts(
        right_texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    similarities = np.sum(left_vecs * right_vecs, axis=1)
    ########################################################################
    #查看相似度集中度
    #sim_debug = pd.DataFrame({"label": labels,"similarity": similarities,})
    #print("\n[Similarity Distribution - Overall]")
    #print(sim_debug["similarity"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    #print("\n[Similarity Distribution - By Label]")
    #print(sim_debug.groupby("label")["similarity"].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    #########################################################################
    preds = [1 if float(s) > args.threshold else 0 for s in similarities]

    metrics_detail = compute_metrics(labels, preds)
    total_time = time.time() - start_time

    pred_rows = []
    for meta, pred, sim in zip(metas, preds, similarities):
        pred_rows.append({
            "pair_index": meta["pair_index"],
            "ltable_id": meta["ltable_id"],
            "rtable_id": meta["rtable_id"],
            "label": int(meta["label"]),
            "pred": int(pred),
            "similarity": float(sim),
            "threshold": float(args.threshold),
            "correct": int(int(pred) == int(meta["label"])),
            "method": method,
            "model_name": model_name,
            "left_text": meta["left_text"],
            "right_text": meta["right_text"],
        })

    pred_df = pd.DataFrame(pred_rows)
    err_df = pred_df[pred_df["correct"] == 0].copy()

    metrics = {
        "dataset": dataset,
        "method": method,
        "method_name": method_name,
        "model_name": model_name,
        "mode": "pretrained_embedding_cosine_similarity",
        "threshold_rule": f"similarity > {args.threshold}",
        "threshold": float(args.threshold),
        "num_cases": int(metrics_detail["num_cases"]),
        "accuracy": metrics_detail["accuracy"],
        "precision": metrics_detail["precision"],
        "recall": metrics_detail["recall"],
        "f1": metrics_detail["f1"],
        "tp": metrics_detail["tp"],
        "tn": metrics_detail["tn"],
        "fp": metrics_detail["fp"],
        "fn": metrics_detail["fn"],
        "total_time_sec": round(float(total_time), 6),
        "avg_time_per_case_sec": round(float(total_time / len(labels)), 6) if labels else math.nan,
    }

    run_config = {
        "dataset": dataset,
        "method": method,
        "method_name": method_name,
        "model_name": model_name,
        "mode": "no_training_similarity_baseline",
        "table_a_path": str(table_a_path),
        "table_b_path": str(table_b_path),
        "test_pairs_path": str(test_path),
        "save_dir": str(save_dir),
        "max_length": args.max_length,
        "max_field_chars": args.max_field_chars,
        "batch_size": args.batch_size,
        "threshold": float(args.threshold),
        "device": str(device),
        "limit": args.limit,
    }

    pred_df.to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    err_df.to_csv(save_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    write_json(metrics, save_dir / "metrics.json")
    write_json(run_config, save_dir / "run_config.json")

    print(f"[Done] {dataset} / {method}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run no-training DL similarity baselines for entity matching."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="bert, roberta, albert, distilbert, or all.",
    )
    parser.add_argument("--model-name", type=str, default=None, help="Override HuggingFace model name/path.")
    parser.add_argument("--method-name", type=str, default=None, help="Output folder name. Only valid when running one method.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset", type=str, default=None, help="Single dataset name, e.g. walmart_amazon.")
    group.add_argument("--datasets", nargs="*", default=None, help='Dataset list, or "all".')

    parser.add_argument("--data-root", type=str, default=r"D:/bysj/API_test/data")
    parser.add_argument("--save-root", type=str, default=r"D:/bysj/API_test/result")

    parser.add_argument("--table-a", type=str, default="tableA.csv")
    parser.add_argument("--table-b", type=str, default="tableB.csv")
    parser.add_argument("--test-pairs", type=str, default="test.csv")

    parser.add_argument("--threshold", type=float, default=0.5, help="Predict match if cosine similarity > threshold.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-field-chars", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: use first N rows of test.csv.")

    return parser.parse_args()


def resolve_datasets(args) -> List[str]:
    if args.dataset:
        return [args.dataset]
    if args.datasets:
        if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
            return DEFAULT_DATASETS
        return args.datasets
    return ["walmart_amazon"]


def resolve_methods(args) -> List[str]:
    if args.method == "all":
        if args.model_name:
            raise ValueError("--model-name can only be used when --method is a single method.")
        if args.method_name:
            raise ValueError("--method-name can only be used when --method is a single method.")
        return list(MODEL_REGISTRY.keys())
    return [args.method]


def main():
    args = parse_args()
    methods = resolve_methods(args)
    datasets = resolve_datasets(args)

    for method in methods:
        for dataset in datasets:
            run_one_dataset(args, method, dataset)


if __name__ == "__main__":
    main()
