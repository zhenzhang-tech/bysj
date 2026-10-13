import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


DEFAULT_DATASETS = [
    "abt_buy",
    "beer",
    "dblp_acm",
    "dblp_scholar",
    "fodors_zagat",
    "itunes_amazon",
    "walmart_amazon",
]

METHOD_CHOICES = [
    "zero_shot",
    "jaccard_top6",
    "st_top6",
    "st_pos3_neg3",
]


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_pairs_file(dataset_dir: Path, preferred_name: Optional[str] = None) -> Path:
    if preferred_name:
        p = dataset_dir / preferred_name
        if not p.exists():
            raise FileNotFoundError(f"Requested pair file does not exist: {p}")
        return p

    for name in ["test.csv", "valid.csv", "validation.csv", "train.csv"]:
        p = dataset_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No pair file found in {dataset_dir}")


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


# ---------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------

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
        parts.append(f"{col}={v}")
    return f"{side_name}: " + " | ".join(parts)


def pair_to_text(left_row: pd.Series, right_row: pd.Series, max_field_chars: int = 120) -> str:
    left = row_to_record_text(left_row, "L", max_field_chars=max_field_chars)
    right = row_to_record_text(right_row, "R", max_field_chars=max_field_chars)
    return f"{left}\n{right}"


def pair_to_retrieval_text(left_row: pd.Series, right_row: pd.Series) -> str:
    return pair_to_text(left_row, right_row, max_field_chars=200).lower()


def get_pair_rows(
    pairs: pd.DataFrame,
    table_a_idx: pd.DataFrame,
    table_b_idx: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> List[Tuple[int, str, str, pd.Series, pd.Series]]:
    rows = []
    missing = 0
    for i, r in pairs.iterrows():
        lid = clean_value(r[left_col])
        rid = clean_value(r[right_col])
        if lid not in table_a_idx.index or rid not in table_b_idx.index:
            missing += 1
            continue
        rows.append((i, lid, rid, table_a_idx.loc[lid], table_b_idx.loc[rid]))
    if missing:
        print(f"[Warning] skipped {missing} pairs because table ids were missing.")
    return rows


def build_items(
    pairs: pd.DataFrame,
    table_a_idx: pd.DataFrame,
    table_b_idx: pd.DataFrame,
    left_col: str,
    right_col: str,
    label_col: str,
) -> List[dict]:
    items = []
    for pair_idx, lid, rid, left_row, right_row in get_pair_rows(
        pairs, table_a_idx, table_b_idx, left_col, right_col
    ):
        label = int(float(clean_value(pairs.loc[pair_idx, label_col])))
        items.append({
            "pair_idx": int(pair_idx),
            "ltable_id": lid,
            "rtable_id": rid,
            "left_row": left_row,
            "right_row": right_row,
            "label": label,
            "retrieval_text": pair_to_retrieval_text(left_row, right_row),
            "pair_key": f"{lid}::{rid}",
        })
    return items


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

class Retriever:
    def __init__(self, backend: str = "jaccard", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.demo_vectors = None
        self.query_vectors = None
        self.demo_texts = []
        self.query_texts = []

    def fit(self, demo_texts: List[str], query_texts: List[str]) -> None:
        self.demo_texts = demo_texts
        self.query_texts = query_texts

        if self.backend == "jaccard":
            return

        if self.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.demo_vectors = self.model.encode(
                demo_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
            self.query_vectors = self.model.encode(
                query_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
            )
            return

        raise ValueError(f"Unknown retrieval backend: {self.backend}")

    @staticmethod
    def _token_set(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def similarity_scores(self, query_idx: int) -> np.ndarray:
        if self.backend == "sentence_transformer":
            return np.asarray(self.demo_vectors @ self.query_vectors[query_idx], dtype=float)

        q = self._token_set(self.query_texts[query_idx])
        scores = []
        for t in self.demo_texts:
            d = self._token_set(t)
            if not q and not d:
                scores.append(1.0)
            elif not q or not d:
                scores.append(0.0)
            else:
                scores.append(len(q & d) / len(q | d))
        return np.asarray(scores, dtype=float)


def top_k(
        scores: np.ndarray,
        k: int,
        forbidden_mask: Optional[np.ndarray] = None,
        min_score: float = 0.0,  # 新增参数
) -> List[int]:
    if forbidden_mask is None:
        forbidden_mask = np.zeros(len(scores), dtype=bool)
    allowed_idx = np.where(~forbidden_mask)[0]
    if len(allowed_idx) == 0:
        return []

    # 按分数降序排序
    sorted_idx = allowed_idx[np.argsort(-scores[allowed_idx])]

    # 依次选取前 k 个且分数 ≥ min_score 的索引
    selected = []
    for idx in sorted_idx:
        if len(selected) >= k:
            break
        if scores[idx] >= min_score:
            selected.append(int(idx))  # 确保是 Python int
    return selected


def top_k_by_label(
    scores: np.ndarray,
    demo_labels: np.ndarray,
    k_pos: int,
    k_neg: int,
    forbidden_mask: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int]]:
    if forbidden_mask is None:
        forbidden_mask = np.zeros(len(scores), dtype=bool)
    allowed = ~forbidden_mask

    pos_idx = np.where((demo_labels == 1) & allowed)[0]
    neg_idx = np.where((demo_labels == 0) & allowed)[0]

    pos_sorted = pos_idx[np.argsort(-scores[pos_idx])] if len(pos_idx) else np.array([], dtype=int)
    neg_sorted = neg_idx[np.argsort(-scores[neg_idx])] if len(neg_idx) else np.array([], dtype=int)
    return pos_sorted[:k_pos].tolist(), neg_sorted[:k_neg].tolist()


# ---------------------------------------------------------------------
# Prompt and Ollama
# ---------------------------------------------------------------------

def build_zero_shot_prompt(left_row: pd.Series, right_row: pd.Series, dataset: str, max_field_chars: int = 240) -> str:
    query_block = pair_to_text(left_row, right_row, max_field_chars=max_field_chars)
    return f"""
You are doing entity matching for dataset: {dataset}.
Task:Determine whether the two records refer to the same real-world entity.
Labels:
1 means MATCH.
0 means NON-MATCH.
Query pair:
{query_block}
Decision rules:
- Return only one character: 1 or 0.
- Do not explain.
- If the two records are the same entity, return 1.
- If they are different entities, return 0.
Answer:
""".strip()


def build_example_block(example_indices: List[int], demo_items: List[dict], title: str, max_field_chars: int = 240) -> str:
    lines = [title]
    if not example_indices:
        lines.append("(none)")
        return "\n".join(lines)
    for idx in example_indices:
        item = demo_items[idx]
        text = pair_to_text(item["left_row"], item["right_row"], max_field_chars=max_field_chars)
        text = text.replace("\n", " ; ")
        lines.append(f"[{int(item['label'])}] {text}")
    return "\n".join(lines).strip()


def build_top6_prompt(
    left_row: pd.Series,
    right_row: pd.Series,
    example_indices: List[int],
    demo_items: List[dict],
    dataset: str,
    max_field_chars: int = 240,
) -> str:
    example_block = build_example_block(
        example_indices,
        demo_items,
        "Retrieved examples from test set, selected by similarity:",
        max_field_chars=max_field_chars,
    )
    query_block = pair_to_text(left_row, right_row, max_field_chars=max_field_chars)

    return f"""
You are doing entity matching for dataset: {dataset}.
Task:Determine whether the two records refer to the same real-world entity.
Labels:
1 means MATCH.
0 means NON-MATCH.
Use the retrieved labeled examples below. They are selected by similarity to the query pair.
{example_block}
Query pair:
{query_block}
Decision rules:
- Return only one character: 1 or 0.
- Do not explain.
- If the two records are the same entity, return 1.
- If they are different entities, return 0.
Answer:
""".strip()


def build_pos_neg_prompt(
    left_row: pd.Series,
    right_row: pd.Series,
    pos_indices: List[int],
    neg_indices: List[int],
    demo_items: List[dict],
    dataset: str,
    max_field_chars: int = 240,
) -> str:
    pos_block = build_example_block(pos_indices, demo_items, "Positive examples, label = 1:", max_field_chars=max_field_chars)
    neg_block = build_example_block(neg_indices, demo_items, "Negative examples, label = 0:", max_field_chars=max_field_chars)
    query_block = pair_to_text(left_row, right_row, max_field_chars=max_field_chars)

    return f"""
You are doing entity matching for dataset: {dataset}.
Task:Determine whether the two records refer to the same real-world entity.
Labels:
1 means MATCH.
0 means NON-MATCH.
Use the in-context examples below. They are selected by sentence-transformer similarity to the query pair.
{pos_block}

{neg_block}

Query pair:
{query_block}
Decision rules:
- Return only one character: 1 or 0.
- Do not explain.
- If the two records are the same entity, return 1.
- If they are different entities, return 0.

Answer:
""".strip()


def call_ollama(
    prompt: str,
    model: str,
    host: str,
    timeout: int = 120,
    temperature: float = 0.0,
    num_gpu: Optional[int] = None,
    num_thread: Optional[int] = None,
    num_ctx: Optional[int] = None,
    keep_alive: Optional[str] = None,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    options = {
        "temperature": temperature,
        "num_predict": 4,
    }
    if num_gpu is not None:
        options["num_gpu"] = int(num_gpu)
    if num_thread is not None:
        options["num_thread"] = int(num_thread)
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if keep_alive:
        payload["keep_alive"] = keep_alive

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip()


def parse_binary_response(text: str) -> int:
    s = str(text).strip()
    m = re.search(r"\b([01])\b", s)
    if m:
        return int(m.group(1))
    if s.startswith("1"):
        return 1
    if s.startswith("0"):
        return 0

    lower = s.lower()
    no_words = ["non-match", "nonmatch", "different", "not same", "no", "false"]
    yes_words = ["same", "match", "yes", "true"]
    if any(w in lower for w in no_words):
        return 0
    if any(w in lower for w in yes_words):
        return 1
    return 0


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def build_prompt_for_method(
    method: str,
    item: dict,
    dataset: str,
    args,
    demo_items: Optional[List[dict]] = None,
    scores: Optional[np.ndarray] = None,
    demo_labels: Optional[np.ndarray] = None,
) -> Tuple[str, dict]:
    extra = {
        "retrieved_indices": "[]",
        "retrieved_scores": "[]",
        "top_pos_indices": "[]",
        "top_neg_indices": "[]",
        "top_pos_scores": "[]",
        "top_neg_scores": "[]",
    }

    if method == "zero_shot":
        prompt = build_zero_shot_prompt(
            item["left_row"],
            item["right_row"],
            dataset,
            max_field_chars=args.max_field_chars,
        )
        return prompt, extra

    if demo_items is None or scores is None:
        raise ValueError(f"Method {method} requires demo_items and scores.")

    forbidden = np.asarray([d["pair_key"] == item["pair_key"] for d in demo_items], dtype=bool)

    if method in {"jaccard_top6", "st_top6"}:
        idx = top_k(scores, k=6, forbidden_mask=forbidden, min_score=args.min_jaccard_score)
        prompt = build_top6_prompt(
            item["left_row"],
            item["right_row"],
            idx,
            demo_items,
            dataset,
            max_field_chars=args.max_field_chars,
        )
        extra["retrieved_indices"] = json.dumps(idx)
        extra["retrieved_scores"] = json.dumps([float(scores[i]) for i in idx])
        return prompt, extra

    if method == "st_pos3_neg3":
        if demo_labels is None:
            raise ValueError("st_pos3_neg3 requires demo_labels.")
        pos_idx, neg_idx = top_k_by_label(
            scores,
            demo_labels,
            k_pos=3,
            k_neg=3,
            forbidden_mask=forbidden,
        )
        prompt = build_pos_neg_prompt(
            item["left_row"],
            item["right_row"],
            pos_idx,
            neg_idx,
            demo_items,
            dataset,
            max_field_chars=args.max_field_chars,
        )
        extra["top_pos_indices"] = json.dumps(pos_idx)
        extra["top_neg_indices"] = json.dumps(neg_idx)
        extra["top_pos_scores"] = json.dumps([float(scores[i]) for i in pos_idx])
        extra["top_neg_scores"] = json.dumps([float(scores[i]) for i in neg_idx])
        return prompt, extra

    raise ValueError(f"Unsupported method: {method}")


def predict_one_item(qi: int, item: dict, dataset: str, args, demo_items=None, demo_labels=None, scores=None) -> dict:
    prompt, extra = build_prompt_for_method(
        args.method,
        item,
        dataset,
        args,
        demo_items=demo_items,
        scores=scores,
        demo_labels=demo_labels,
    )

    if args.dry_run:
        raw = "0"
        pred = 0
    else:
        try:
            raw = call_ollama(
                prompt=prompt,
                model=args.model,
                host=args.ollama_host,
                timeout=args.timeout,
                temperature=args.temperature,
                num_gpu=args.ollama_num_gpu,
                num_thread=args.ollama_num_thread,
                num_ctx=args.ollama_num_ctx,
                keep_alive=args.ollama_keep_alive,
            )
            pred = parse_binary_response(raw)

        except Exception as e:
            raise RuntimeError(f"Ollama call failed for pair {item['pair_key']}: {repr(e)}") from e

    label = int(item["label"])
    out = {
        "pair_index": item["pair_idx"],
        "ltable_id": item["ltable_id"],
        "rtable_id": item["rtable_id"],
        "label": label,
        "pred": int(pred),
        "correct": int(pred == label),
        "method": args.method,
        "stage": "LLM_BASELINE",
        "raw_response": raw,
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def method_to_retrieval_backend(method: str) -> Optional[str]:
    if method == "zero_shot":
        return None
    if method == "jaccard_top6":
        return "jaccard"
    if method in {"st_top6", "st_pos3_neg3"}:
        return "sentence_transformer"
    raise ValueError(f"Unsupported method: {method}")


def run_one_dataset(args, dataset: str) -> None:
    dataset_dir = Path(args.data_root) / dataset
    table_a_path = dataset_dir / args.table_a
    table_b_path = dataset_dir / args.table_b
    test_path = find_pairs_file(dataset_dir, args.test_pairs)
    train_path = dataset_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")


    print(f"\n[Dataset] {dataset}")
    print(f"[Method] {args.method}")
    print(f"[Data] tableA: {table_a_path}")
    print(f"[Data] tableB: {table_b_path}")
    print(f"[Data] test pairs / demo pool: {test_path}")

    table_a = id_index(read_csv(table_a_path))
    table_b = id_index(read_csv(table_b_path))
    test_pairs = read_csv(test_path)
    left_col, right_col, label_col = infer_pair_columns(test_pairs)
    test_items = build_items(test_pairs, table_a, table_b, left_col, right_col, label_col)
    train_pairs = read_csv(train_path)
    train_left_col, train_right_col, train_label_col = infer_pair_columns(train_pairs)
    train_items = build_items(train_pairs, table_a, table_b, train_left_col, train_right_col, train_label_col)

    if args.limit and args.limit > 0:
        test_items = test_items[:args.limit]

    if not test_items:
        raise ValueError(f"No valid test pairs for dataset {dataset}.")

    backend = method_to_retrieval_backend(args.method)
    retriever = None
    demo_items = None
    demo_labels = None

    if backend is not None:
        # The user explicitly requested retrieval within the train set.
        demo_items = train_items  # ← 现在从训练集检索
        demo_labels = np.asarray([int(x["label"]) for x in demo_items], dtype=int)

        if args.method == "st_pos3_neg3":
            if (demo_labels == 1).sum() == 0 or (demo_labels == 0).sum() == 0:
                raise ValueError("st_pos3_neg3 requires both positive and negative examples in the test set.")

        retriever = Retriever(backend=backend, model_name=args.embedding_model)
        retriever.fit(
            [x["retrieval_text"] for x in demo_items],
            [x["retrieval_text"] for x in test_items],
        )

    method_name = args.method_name or f"{args.method}_{args.model.replace(':', '_')}"
    save_dir = Path(args.save_root) / dataset / method_name
    ensure_dir(save_dir)

    predictions = []
    existing_path = save_dir / "predictions.csv"
    done_keys = set()

    if args.resume and existing_path.exists():
        old_df = pd.read_csv(existing_path, dtype=str, keep_default_na=False)
        if "ltable_id" in old_df.columns and "rtable_id" in old_df.columns:
            old_df["_key"] = old_df["ltable_id"].astype(str) + "::" + old_df["rtable_id"].astype(str)
            done_keys = set(old_df["_key"].tolist())
            predictions.extend(old_df.drop(columns=["_key"], errors="ignore").to_dict("records"))
            print(f"[Resume] loaded {len(done_keys)} existing predictions.")

    pending = [(qi, item) for qi, item in enumerate(test_items) if item["pair_key"] not in done_keys]
    print(f"[Run] pending cases: {len(pending)} / {len(test_items)}, workers={args.num_workers}")

    start = time.time()

    def flush_partial() -> None:
        if predictions:
            pd.DataFrame(predictions).to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    if args.num_workers <= 1:
        for done_count, (qi, item) in enumerate(pending, start=1):
            scores = retriever.similarity_scores(qi) if retriever is not None else None
            row = predict_one_item(
                qi,
                item,
                dataset,
                args,
                demo_items=demo_items,
                demo_labels=demo_labels,
                scores=scores,
            )
            predictions.append(row)

            if done_count % args.save_every == 0:
                flush_partial()

            if done_count % args.log_every == 0 or done_count == len(pending):
                elapsed = time.time() - start
                total_done = len(done_keys) + done_count
                print(f"[Progress] {dataset}: {total_done}/{len(test_items)} done, elapsed={elapsed:.1f}s")
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            for qi, item in pending:
                scores = retriever.similarity_scores(qi) if retriever is not None else None
                futures.append(ex.submit(
                    predict_one_item,
                    qi,
                    item,
                    dataset,
                    args,
                    demo_items,
                    demo_labels,
                    scores,
                ))

            for done_count, fut in enumerate(as_completed(futures), start=1):
                predictions.append(fut.result())

                if done_count % args.save_every == 0:
                    flush_partial()

                if done_count % args.log_every == 0 or done_count == len(futures):
                    elapsed = time.time() - start
                    total_done = len(done_keys) + done_count
                    print(f"[Progress] {dataset}: {total_done}/{len(test_items)} done, elapsed={elapsed:.1f}s")

    flush_partial()

    total_time = time.time() - start
    pred_df = pd.DataFrame(predictions)
    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["pred"] = pred_df["pred"].astype(int)
    pred_df["correct"] = pred_df["correct"].astype(int)

    detailed_metrics = compute_metrics(pred_df["label"].tolist(), pred_df["pred"].tolist())
    metrics = {
        "dataset": dataset,
        "method": args.method,
        "method_name": method_name,
        "model": args.model,
        "num_cases": int(detailed_metrics["num_cases"]),
        "accuracy": detailed_metrics["accuracy"],
        "precision": detailed_metrics["precision"],
        "recall": detailed_metrics["recall"],
        "f1": detailed_metrics["f1"],
        "total_time_sec": round(total_time, 6),
        "avg_time_per_case_sec": round(total_time / len(pred_df), 6) if len(pred_df) else math.nan,
    }

    errors = pred_df[pred_df["correct"] == 0].copy()
    pred_df.to_csv(save_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(save_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    write_json(metrics, save_dir / "metrics.json")

    run_config = {
        "dataset": dataset,
        "method": args.method,
        "method_name": method_name,
        "model": args.model,
        "ollama_host": args.ollama_host,
        "table_a_path": str(table_a_path),
        "table_b_path": str(table_b_path),
        "test_pairs_path": str(test_path),
        "demo_source": "test.csv",
        "current_pair_excluded_from_retrieval": backend is not None,
        "save_dir": str(save_dir),
        "retrieval_backend": backend,
        "embedding_model": args.embedding_model if backend == "sentence_transformer" else None,
        "temperature": args.temperature,
        "limit": args.limit,
        "num_workers": args.num_workers,
        "max_field_chars": args.max_field_chars,
        "resume": args.resume,
        "ollama_num_gpu": args.ollama_num_gpu,
        "ollama_num_thread": args.ollama_num_thread,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_keep_alive": args.ollama_keep_alive,
        "tp": detailed_metrics["tp"],
        "tn": detailed_metrics["tn"],
        "fp": detailed_metrics["fp"],
        "fn": detailed_metrics["fn"],
    }
    write_json(run_config, save_dir / "run_config.json")

    print(f"[Done] {dataset}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[Saved] {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LLM baselines for entity matching: zero-shot, Jaccard top-6 ICL, sentence-transformer top-6 ICL, and sentence-transformer pos3/neg3 ICL."
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--dataset", type=str, default=None, help="Single dataset name, e.g. walmart_amazon.")
    group.add_argument("--datasets", nargs="*", default=None, help='Dataset list, or "all".')

    parser.add_argument("--method", type=str, required=True, choices=METHOD_CHOICES)
    parser.add_argument("--data-root", type=str, default=r"D:/bysj/API_test/data")
    parser.add_argument("--save-root", type=str, default=r"D:/bysj/API_test/result")
    parser.add_argument("--method-name", type=str, default=None)

    parser.add_argument("--table-a", type=str, default="tableA.csv")
    parser.add_argument("--table-b", type=str, default="tableB.csv")
    parser.add_argument("--test-pairs", type=str, default="test.csv")

    parser.add_argument("--model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--ollama-num-gpu", type=int, default=None, help="Forwarded to Ollama options.num_gpu. Try -1 for all available GPU layers, 0 for CPU.")
    parser.add_argument("--ollama-num-thread", type=int, default=None, help="Forwarded to Ollama options.num_thread.")
    parser.add_argument("--ollama-num-ctx", type=int, default=None, help="Forwarded to Ollama options.num_ctx.")
    parser.add_argument("--ollama-keep-alive", type=str, default="30m", help="Forwarded to Ollama keep_alive, e.g. 30m.")

    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--limit", type=int, default=0, help="Debug only: run first N test pairs.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel Ollama requests. Use 1 for stability.")
    parser.add_argument("--max-field-chars", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Debug output without calling Ollama.")
    parser.add_argument("--min-jaccard-score", type=float, default=0.75,
                        help="Minimum Jaccard similarity to keep a retrieved example (for jaccard_top6).")
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
        run_one_dataset(args, dataset)


if __name__ == "__main__":
    main()
