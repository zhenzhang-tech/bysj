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


def clean_value(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return re.sub(r"\s+", " ", s)


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


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset:
    def __init__(self, left_texts, right_texts, labels, tokenizer, max_length):
        self.left_texts = left_texts
        self.right_texts = right_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(
            self.left_texts[idx],
            self.right_texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 去掉 batch 维度（因为一次只编码一个样本）
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return enc


def evaluate_model(model, dataloader, device, threshold=0.5):
    import torch
    from torch.nn import functional as F

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=-1)
            prob_match = probs[:, 1]
            pred = (prob_match >= threshold).long()
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
            all_preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
            all_probs.extend(probs[:, 1].detach().cpu().numpy().astype(float).tolist())

    return compute_metrics(all_labels, all_preds), all_preds, all_probs, all_labels

def find_best_threshold(labels, probs):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.05, 0.96, 0.01):
        preds = [1 if p >= t else 0 for p in probs]
        metrics = compute_metrics(labels, preds)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(t)

    return best_t, best_f1

def train_one_dataset(args, dataset: str) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(args.seed)

    method = args.method.lower()
    model_name = args.model_name or MODEL_REGISTRY[method]
    method_name = args.method_name or f"{method}_baseline"

    dataset_dir = Path(args.data_root) / dataset
    table_a_path = dataset_dir / args.table_a
    table_b_path = dataset_dir / args.table_b
    train_path = dataset_dir / args.train_pairs
    valid_path = dataset_dir / args.valid_pairs
    test_path = dataset_dir / args.test_pairs

    print(f"\n[Dataset] {dataset}")
    print(f"[Method] {method}")
    print(f"[Backbone] {model_name}")
    print(f"[Data] tableA: {table_a_path}")
    print(f"[Data] tableB: {table_b_path}")
    print(f"[Data] train: {train_path}")
    print(f"[Data] valid: {valid_path}")
    print(f"[Data] test:  {test_path}")

    save_dir = Path(args.save_root) / dataset / method_name
    ensure_dir(save_dir)
    best_model_dir = save_dir / "best_model"
    ensure_dir(best_model_dir)

    table_a = id_index(read_csv(table_a_path))
    table_b = id_index(read_csv(table_b_path))
    train_pairs = read_csv(train_path)
    valid_pairs = read_csv(valid_path)
    test_pairs = read_csv(test_path)

    if args.limit and args.limit > 0:
        train_pairs = train_pairs.head(args.limit)
        valid_pairs = valid_pairs.head(args.limit)
        test_pairs = test_pairs.head(args.limit)

    train_l, train_r, train_y, _ = build_pair_texts(train_pairs, table_a, table_b, args.max_field_chars)
    valid_l, valid_r, valid_y, _ = build_pair_texts(valid_pairs, table_a, table_b, args.max_field_chars)
    test_l, test_r, test_y, test_meta = build_pair_texts(test_pairs, table_a, table_b, args.max_field_chars)

    if not train_y:
        raise ValueError(f"No valid train pairs for dataset {dataset}.")
    if not valid_y:
        raise ValueError(f"No valid valid pairs for dataset {dataset}.")
    if not test_y:
        raise ValueError(f"No valid test pairs for dataset {dataset}.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = PairDataset(train_l, train_r, train_y, tokenizer, args.max_length)
    valid_ds = PairDataset(valid_l, valid_r, valid_y, tokenizer, args.max_length)
    test_ds = PairDataset(test_l, test_r, test_y, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_valid_f1 = -1.0
    best_epoch = -1
    history = []
    start = time.time()
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = labels

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu().item())

            if args.log_every and step % args.log_every == 0:
                avg_loss = epoch_loss / step
                print(f"[Train] {dataset} epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} loss={avg_loss:.4f}")

        train_loss = epoch_loss / max(1, len(train_loader))
        valid_metrics_raw, _, valid_probs, valid_labels = evaluate_model(model, valid_loader, device, threshold=0.5)
        best_threshold_epoch, best_f1_epoch = find_best_threshold(valid_labels, valid_probs)

        valid_preds = [1 if p >= best_threshold_epoch else 0 for p in valid_probs]
        valid_metrics = compute_metrics(valid_labels, valid_preds)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        print(f"[Valid] epoch={epoch} loss={train_loss:.4f} f1={valid_metrics['f1']:.4f} acc={valid_metrics['accuracy']:.4f}")

        if valid_metrics["f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["f1"]
            best_epoch = epoch
            best_threshold = best_threshold_epoch
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"[Best] saved epoch={epoch}, valid_f1={best_valid_f1:.4f}, threshold={best_threshold:.2f}")

    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    test_metrics_detail, test_preds, test_probs, test_labels = evaluate_model(
        model, test_loader, device, threshold=best_threshold
    )

    total_time = time.time() - start
    metrics = {
        "dataset": dataset,
        "method": method,
        "method_name": method_name,
        "model_name": model_name,
        "best_epoch": int(best_epoch),
        "best_valid_f1": float(best_valid_f1),
        "num_cases": int(test_metrics_detail["num_cases"]),
        "accuracy": test_metrics_detail["accuracy"],
        "precision": test_metrics_detail["precision"],
        "recall": test_metrics_detail["recall"],
        "f1": test_metrics_detail["f1"],
        "total_time_sec": round(total_time, 6),
        "avg_time_per_case_sec": round(total_time / len(test_labels), 6) if test_labels else math.nan,
        "best_threshold": float(best_threshold),
    }

    pred_rows = []
    for meta, pred, prob in zip(test_meta, test_preds, test_probs):
        pred_rows.append({
            "pair_index": meta["pair_index"],
            "ltable_id": meta["ltable_id"],
            "rtable_id": meta["rtable_id"],
            "label": int(meta["label"]),
            "pred": int(pred),
            "prob_match": float(prob),
            "correct": int(int(pred) == int(meta["label"])),
            "method": method,
            "model_name": model_name,
            "left_text": meta["left_text"],
            "right_text": meta["right_text"],
        })

    pred_df = pd.DataFrame(pred_rows)
    err_df = pred_df[pred_df["correct"] == 0].copy()

    pred_df.to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    err_df.to_csv(save_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    write_json(metrics, save_dir / "metrics.json")

    run_config = {
        "dataset": dataset,
        "method": method,
        "method_name": method_name,
        "model_name": model_name,
        "table_a_path": str(table_a_path),
        "table_b_path": str(table_b_path),
        "train_pairs_path": str(train_path),
        "valid_pairs_path": str(valid_path),
        "test_pairs_path": str(test_path),
        "save_dir": str(save_dir),
        "best_model_dir": str(best_model_dir),
        "max_length": args.max_length,
        "max_field_chars": args.max_field_chars,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "device": str(device),
        "limit": args.limit,
        "tp": test_metrics_detail["tp"],
        "tn": test_metrics_detail["tn"],
        "fp": test_metrics_detail["fp"],
        "fn": test_metrics_detail["fn"],
        "best_threshold": float(best_threshold),
    }
    write_json(run_config, save_dir / "run_config.json")

    print(f"[Done] {dataset}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] {save_dir}")



def parse_args():
    parser = argparse.ArgumentParser(description="Run deep learning baselines for entity matching: BERT, RoBERTa, ALBERT, DistilBERT.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset", type=str, default=None, help="Single dataset name, e.g. walmart_amazon.")
    group.add_argument("--datasets", nargs="*", default=None, help='Dataset list, or "all".')
    parser.add_argument("--method", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model-name", type=str, default=None, help="Override HuggingFace model name/path.")
    parser.add_argument("--method-name", type=str, default=None, help="Output folder name. Default: METHOD_baseline.")
    parser.add_argument("--data-root", type=str, default=r"D:/bysj/API_test/data")
    parser.add_argument("--save-root", type=str, default=r"D:/bysj/API_test/result")
    parser.add_argument("--table-a", type=str, default="tableA.csv")
    parser.add_argument("--table-b", type=str, default="tableB.csv")
    parser.add_argument("--train-pairs", type=str, default="train.csv")
    parser.add_argument("--valid-pairs", type=str, default="valid.csv")
    parser.add_argument("--test-pairs", type=str, default="test.csv")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-field-chars", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: use first N rows of train/valid/test.")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def resolve_datasets(args) -> List[str]:
    if args.dataset:
        return [args.dataset]
    if args.datasets:
        if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
            return DEFAULT_DATASETS
        return args.datasets
    return ["walmart_amazon"]


def main():
    args = parse_args()
    datasets = resolve_datasets(args)
    for dataset in datasets:
        train_one_dataset(args, dataset)


if __name__ == "__main__":
    main()
