import argparse
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Experiment configuration
# =========================

DATASETS = [
    ("abt_buy", "AB"),
    ("walmart_amazon", "WA"),
    ("dblp_acm", "DA"),
    ("dblp_scholar", "DS"),
    ("beer", "BE"),
    ("fodors_zagat", "FZ"),
    ("itunes_amazon", "IA"),
]

MAIN_METHODS = [
    ("bert_fixT_baseline", "bert"),
    ("roberta_fixT_baseline", "roberta"),
    ("albert_fixT_baseline", "albert"),
    ("distilbert_fixT_baseline", "distilbert"),
    ("zero_shot_qwen2.5_1.5b", "Zero-shot"),
    ("jaccard_top6_qwen2.5_1.5b", "J6"),
    ("st_top6_qwen2.5_1.5b", "ST6"),
    ("st_pos3_neg3_qwen2.5_1.5b", "ST+-3"),
    ("n1n_deepseekV3.2thinking", "RKLEM"),
]

RKLEM_METHOD = "n1n_deepseekV3.2thinking"

ROBUSTNESS_METHODS = [
    ("n1n_deepseekV3.2thinking", "deepseek"),
    ("n1n_qwen3-235b-a22b-thinking-2507", "qwen"),
    ("n1n_gpt-5.4", "gpt"),
]

METRIC_FILES = [
    "metrics.json", "metric.json", "test_metrics.json", "eval_metrics.json",
    "eval_results.json", "test_results.json", "results.json", "result.json",
    "scores.json", "score.json", "classification_report.json",
]
PRED_FILES = [
    "predictions.csv", "prediction.csv", "test_predictions.csv", "eval_predictions.csv",
    "preds.csv", "test_preds.csv", "result.csv", "results.csv", "test_results.csv", "eval_results.csv",
]

METRIC_KEYS = {
    "f1": ["f1", "overall_f1", "f1_score", "f1score", "test_f1", "eval_f1", "macro_f1", "micro_f1", "weighted_f1", "f1_macro", "f1_micro", "f1_weighted"],
    "accuracy": ["accuracy", "acc", "overall_accuracy", "test_accuracy", "eval_accuracy"],
    "rule_coverage": ["rule_coverage", "rulecoverage", "coverage"],
    "gray_ratio": ["gray_ratio", "grayratio", "gray_rate"],
    "rule_accuracy": ["rule_accuracy", "ruleaccuracy", "rule_acc"],
    "gray_accuracy": ["gray_accuracy", "grayaccuracy", "gray_acc"],
    "total_time": ["total_time_sec", "total_time", "totaltime", "runtime", "time_sec", "elapsed_sec"],
    "avg_time": ["avg_time_per_case_sec", "avg_time_per_case", "avgtime", "avg_inference_time", "average_time", "avg_latency", "latency"],
    "error_rule": ["error_rule", "errorrule", "rule_error", "rule_errors"],
    "error_gray": ["error_gray", "errorgray", "gray_error", "gray_errors"],
}

LABEL_COLS = ["label", "labels", "gold", "gold_label", "ground_truth", "true_label", "target", "targets", "y_true", "y"]
PRED_COLS = ["pred", "prediction", "predictions", "pred_label", "predicted_label", "predicted", "predict", "output", "y_pred"]
SCORE_COLS = ["score", "prob", "proba", "probability", "match_score", "match_prob", "positive_prob", "pos_prob"]
RATIO_METRICS = {"f1", "accuracy", "rule_coverage", "gray_ratio", "rule_accuracy", "gray_accuracy"}


# =========================
# Basic utilities
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default


def norm_name(x: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(x).lower())


def flatten_dict(d: dict, prefix: str = "") -> Dict[str, object]:
    """Flatten nested dictionaries to support fields such as test.f1."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk))
        else:
            out[nk] = v
    return out


def load_json(path: Path) -> Optional[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"[Warn] Skip JSON: {path} ({e})")
        return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[Warn] Skip CSV: {path} ({e})")
        return None


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    """If a CSV is opened by Excel, save a timestamped copy instead of crashing."""
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"[Warn] Locked file: {path}; saved as {alt}")
        return alt


def safe_write_text(path: Path, text: str) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        alt.write_text(text, encoding="utf-8")
        print(f"[Warn] Locked file: {path}; saved as {alt}")
        return alt


# =========================
# Result loading
# =========================

def exp_dir(result_root: Path, dataset: str, method: str) -> Path:
    """Return experiment directory."""
    return result_root / dataset / method


def candidate_files(folder: Path, names: List[str], patterns: List[str]) -> List[Path]:
    """Find direct candidate files first, then recursive wildcard matches."""
    if not folder.exists():
        return []
    seen, out = set(), []

    def add(p: Path) -> None:
        if not p.is_file():
            return
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            out.append(p)

    for name in names:
        add(folder / name)
    for pat in patterns:
        for p in sorted(folder.rglob(pat), key=lambda x: (len(x.parts), str(x).lower())):
            add(p)
    return out


def normalize_ratio(value, key: str) -> float:
    """Internally keep ratio metrics in 0-1 form. 80.7 is converted to 0.807."""
    v = safe_float(value)
    if pd.isna(v):
        return np.nan
    if key in RATIO_METRICS and 1.5 < v <= 100:
        return v / 100.0
    return v


def get_metric(metrics: Optional[dict], key: str, default=np.nan) -> float:
    """Read one metric from JSON using alias fields."""
    if not isinstance(metrics, dict):
        return default
    flat = flatten_dict(metrics)
    aliases = [norm_name(a) for a in METRIC_KEYS.get(key, [key])]
    items = [(norm_name(k), v) for k, v in flat.items()]

    for k, v in items:
        if k in aliases:
            val = normalize_ratio(v, key)
            return default if pd.isna(val) else val
    for k, v in items:
        if any(k.endswith(a) for a in aliases):
            val = normalize_ratio(v, key)
            return default if pd.isna(val) else val
    return default


def metric_file_score(data: Optional[dict]) -> int:
    if not isinstance(data, dict):
        return 0
    return sum(0 if pd.isna(get_metric(data, k, np.nan)) else 1 for k in ["f1", "accuracy", "total_time", "avg_time", "rule_coverage"])


def find_metrics(folder: Path) -> Tuple[Optional[dict], str]:
    """Pick the JSON file that contains the most useful metric fields."""
    best_data, best_path, best_score = None, "", -1
    for p in candidate_files(folder, METRIC_FILES, ["*metric*.json", "*result*.json", "*score*.json", "*report*.json"]):
        data = load_json(p)
        score = metric_file_score(data)
        if score > best_score:
            best_data, best_path, best_score = data, str(p), score
        if score >= 4:
            break
    return (best_data, best_path) if best_score > 0 else (None, best_path)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    by_norm = {norm_name(c): c for c in df.columns}
    for c in candidates:
        if norm_name(c) in by_norm:
            return by_norm[norm_name(c)]
    cand = [norm_name(c) for c in candidates]
    for c in df.columns:
        nc = norm_name(c)
        if any(nc.endswith(x) or nc.startswith(x) for x in cand):
            return c
    return None


def find_predictions(folder: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Find a prediction CSV to compute fallback F1/Accuracy and stage metrics."""
    best_df, best_path, best_score = None, "", -1
    patterns = ["*pred*.csv", "*prediction*.csv", "*result*.csv", "*test*.csv", "*eval*.csv"]
    for p in candidate_files(folder, PRED_FILES, patterns):
        df = load_csv(p)
        if df is None:
            continue
        score = int(find_col(df, LABEL_COLS) is not None) + int(find_col(df, PRED_COLS) is not None) + int(find_col(df, SCORE_COLS) is not None)
        if score > best_score:
            best_df, best_path, best_score = df, str(p), score
        if score >= 2:
            break
    return best_df, best_path


def binary_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "match": 1, "non-match": 0, "nonmatch": 0}
    return s.astype(str).str.strip().str.lower().map(mapping)


def stage_mask(df: pd.DataFrame, stage: str) -> pd.Series:
    """Identify RULE or GRAY rows from stage/source column."""
    target = stage.upper()
    if "stage" in df.columns:
        s = df["stage"].astype(str).str.upper()
        return s.eq("RULE") if target == "RULE" else s.str.contains(target, na=False)
    if "source" in df.columns:
        s = df["source"].astype(str).str.upper()
        return s.str.startswith("RULE", na=False) if target == "RULE" else s.str.contains(target, na=False)
    return pd.Series([False] * len(df), index=df.index)


def summarize_predictions(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Fallback metrics from predictions.csv."""
    out = {"f1": np.nan, "accuracy": np.nan, "rule_accuracy": np.nan, "gray_accuracy": np.nan, "error_rule": np.nan, "error_gray": np.nan}
    if df is None or df.empty:
        return out

    label_col = find_col(df, LABEL_COLS)
    pred_col = find_col(df, PRED_COLS)
    if pred_col is None:
        score_col = find_col(df, SCORE_COLS)
        if score_col is not None:
            score = pd.to_numeric(df[score_col], errors="coerce")
            if score.notna().any():
                df = df.copy()
                df["__pred_from_score__"] = (score >= 0.5).astype(int)
                pred_col = "__pred_from_score__"

    if label_col is None or pred_col is None:
        return out

    y = binary_series(df[label_col])
    p = binary_series(df[pred_col])
    valid = y.notna() & p.notna()
    if not valid.any():
        return out

    dfv = df.loc[valid].copy()
    y = y.loc[valid].astype(int)
    p = p.loc[valid].astype(int)
    correct = y == p

    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and precision + recall else np.nan

    rule = stage_mask(dfv, "RULE")
    gray = stage_mask(dfv, "GRAY")
    wrong = ~correct

    def acc(mask: pd.Series) -> float:
        return float(correct.loc[mask].mean()) if int(mask.sum()) else np.nan

    out.update({
        "f1": f1,
        "accuracy": float(correct.mean()),
        "rule_accuracy": acc(rule),
        "gray_accuracy": acc(gray),
        "error_rule": int(wrong.loc[rule].sum()) if rule.any() else np.nan,
        "error_gray": int(wrong.loc[gray].sum()) if gray.any() else np.nan,
    })
    return out


def summarize_error_csv(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Prefer error_analysis.csv for Error_rule/Error_gray if available."""
    out = {"error_rule": np.nan, "error_gray": np.nan}
    if df is None or df.empty:
        return out
    rule = stage_mask(df, "RULE")
    gray = stage_mask(df, "GRAY")
    out["error_rule"] = int(rule.sum()) if rule.any() else np.nan
    out["error_gray"] = int(gray.sum()) if gray.any() else np.nan
    return out


def collect_one(result_root: Path, dataset: str, method: str, method_display: str) -> Dict[str, object]:
    """Collect all fields needed by the requested tables."""
    folder = exp_dir(result_root, dataset, method)
    metrics, metrics_path = find_metrics(folder)
    pred_df, pred_path = find_predictions(folder)
    err_df = load_csv(folder / "error_analysis.csv")
    coverage = load_json(folder / "rule_coverage_summary.json")

    pred = summarize_predictions(pred_df)
    err = summarize_error_csv(err_df)

    cov_rule = normalize_ratio(safe_float((coverage or {}).get("rule_coverage"), np.nan), "rule_coverage")
    cov_gray = normalize_ratio(safe_float((coverage or {}).get("gray_ratio"), np.nan), "gray_ratio")

    error_rule_default = err["error_rule"] if not pd.isna(err["error_rule"]) else pred["error_rule"]
    error_gray_default = err["error_gray"] if not pd.isna(err["error_gray"]) else pred["error_gray"]

    return {
        "dataset": dataset,
        "Dataset": dict(DATASETS).get(dataset, dataset),
        "method": method,
        "method_display": method_display,
        "exists": folder.exists(),
        "has_metrics": metrics is not None,
        "has_predictions": pred_df is not None,
        "experiment_dir": str(folder),
        "metrics_path": metrics_path,
        "predictions_path": pred_path,
        "f1": get_metric(metrics, "f1", pred["f1"]),
        "accuracy": get_metric(metrics, "accuracy", pred["accuracy"]),
        "rule_coverage": get_metric(metrics, "rule_coverage", cov_rule),
        "gray_ratio": get_metric(metrics, "gray_ratio", cov_gray),
        "rule_accuracy": get_metric(metrics, "rule_accuracy", pred["rule_accuracy"]),
        "gray_accuracy": get_metric(metrics, "gray_accuracy", pred["gray_accuracy"]),
        "total_time": get_metric(metrics, "total_time", np.nan),
        "avg_time": get_metric(metrics, "avg_time", np.nan),
        "error_rule": get_metric(metrics, "error_rule", error_rule_default),
        "error_gray": get_metric(metrics, "error_gray", error_gray_default),
    }


def collect_all(result_root: Path, methods: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for dataset, _abbr in DATASETS:
        for method, name in methods:
            rows.append(collect_one(result_root, dataset, method, name))
    return pd.DataFrame(rows)


# =========================
# Table building and writing
# =========================

def make_matrix(df: pd.DataFrame, metric: str, methods: List[Tuple[str, str]], final_row: str) -> pd.DataFrame:
    """Build dataset x method table and append Average or Sum."""
    method_ids = [m for m, _ in methods]
    method_names = [n for _, n in methods]
    dataset_ids = [d for d, _ in DATASETS]
    dataset_names = [a for _, a in DATASETS]

    pivot = df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="first")
    mat = pivot.reindex(index=dataset_ids, columns=method_ids)
    out = mat.copy()
    out.insert(0, "Dataset", dataset_names)

    if final_row.lower() == "sum":
        final = mat.sum(axis=0, skipna=True)
        final[mat.notna().sum(axis=0) == 0] = np.nan
        label = "Sum"
    else:
        final = mat.mean(axis=0, skipna=True)
        label = "Average"

    out.loc[len(out)] = [label] + [final.get(m, np.nan) for m in method_ids]
    return out.rename(columns={method_ids[i]: method_names[i] for i in range(len(method_ids))})


def ratio_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Convert internal 0-1 ratio table into percentage-point table."""
    out = df.copy()
    for col in out.columns[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out


def rklem_pair(df: pd.DataFrame, metric_a: str, name_a: str, metric_b: str, name_b: str, final_row: str = "Average") -> pd.DataFrame:
    """RKLEM-only table with two columns."""
    sub = df[df["method"] == RKLEM_METHOD].set_index("dataset").reindex([d for d, _ in DATASETS])
    out = pd.DataFrame({
        "Dataset": [a for _, a in DATASETS],
        name_a: pd.to_numeric(sub[metric_a], errors="coerce").to_numpy(dtype=float),
        name_b: pd.to_numeric(sub[metric_b], errors="coerce").to_numpy(dtype=float),
    })
    if final_row.lower() == "sum":
        out.loc[len(out)] = ["Sum", out[name_a].sum(skipna=True), out[name_b].sum(skipna=True)]
        if out[[name_a, name_b]].iloc[:-1].notna().sum().sum() == 0:
            out.loc[len(out) - 1, [name_a, name_b]] = np.nan
    else:
        out.loc[len(out)] = ["Average", out[name_a].mean(skipna=True), out[name_b].mean(skipna=True)]
    return out


def fmt_cell(x, kind: str) -> str:
    v = safe_float(x)
    if pd.isna(v) or not np.isfinite(v):
        return "—"
    if kind == "percent":
        return f"{v:.1f}"
    if kind == "count":
        return f"{int(round(v))}"
    if kind == "time_total":
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    if kind == "time_avg":
        return f"{v:.4f}"
    return f"{v:.4f}"


def rank_styles(vals: List[float], higher: bool) -> Dict[int, str]:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {}
    uniq = sorted(set(round(float(v), 10) for v in valid), reverse=higher)
    styles = {}
    for i, v in enumerate(arr):
        if np.isfinite(v) and np.isclose(v, uniq[0]):
            styles[i] = "best"
    if len(uniq) > 1:
        for i, v in enumerate(arr):
            if np.isfinite(v) and np.isclose(v, uniq[1]):
                styles[i] = "second"
    return styles


def style_text(text: str, style: Optional[str], fmt: str) -> str:
    if text == "—":
        return text
    if fmt == "html":
        text = escape(text)
        return f"<strong>{text}</strong>" if style == "best" else f"<u>{text}</u>" if style == "second" else text
    if fmt == "latex":
        return r"\textbf{" + text + "}" if style == "best" else r"\underline{" + text + "}" if style == "second" else text
    return f"<b>{text}</b>" if style == "best" else f"<u>{text}</u>" if style == "second" else text


def to_md(df: pd.DataFrame, kind: str, rank: bool, higher: bool) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        value_cols = cols[1:]
        styles = rank_styles([row[c] for c in value_cols], higher) if rank else {}
        cells = [str(row["Dataset"])] + [style_text(fmt_cell(row[c], kind), styles.get(i), "markdown") for i, c in enumerate(value_cols)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_html(df: pd.DataFrame, title: str, kind: str, rank: bool, higher: bool) -> str:
    cols = list(df.columns)
    lines = [
        "<!doctype html>", "<html lang=\"zh-CN\">", "<head>", "<meta charset=\"utf-8\">",
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:'Times New Roman','Microsoft YaHei',SimSun,serif;margin:24px;}",
        "table{border-collapse:collapse;width:100%;font-size:14px;}",
        "caption{caption-side:top;font-weight:600;margin-bottom:10px;font-size:16px;}",
        "th,td{border-top:1px solid #333;border-bottom:1px solid #333;padding:6px 8px;text-align:center;}",
        "th{font-weight:700;} tbody tr:last-child td{border-top:1.5px solid #333;}",
        ".note{margin-top:10px;font-size:12px;color:#555;}",
        "</style></head><body><table>", f"<caption>{escape(title)}</caption>", "<thead><tr>",
    ]
    lines += [f"<th>{escape(str(c))}</th>" for c in cols]
    lines += ["</tr></thead><tbody>"]
    for _, row in df.iterrows():
        value_cols = cols[1:]
        styles = rank_styles([row[c] for c in value_cols], higher) if rank else {}
        lines.append("<tr>")
        lines.append(f"<td>{escape(str(row['Dataset']))}</td>")
        for i, c in enumerate(value_cols):
            lines.append(f"<td>{style_text(fmt_cell(row[c], kind), styles.get(i), 'html')}</td>")
        lines.append("</tr>")
    note = "Missing results are shown as —."
    if rank:
        note = ("The best value in each row is bold; the second best is underlined. " if higher else
                "The lowest value in each row is bold; the second lowest is underlined. ") + note
    lines += ["</tbody></table>", f"<div class=\"note\">{escape(note)}</div>", "</body></html>"]
    return "\n".join(lines) + "\n"


def to_tex(df: pd.DataFrame, title: str, kind: str, rank: bool, higher: bool) -> str:
    cols = list(df.columns)
    lines = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{title}}}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
        " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule",
    ]
    for _, row in df.iterrows():
        value_cols = cols[1:]
        styles = rank_styles([row[c] for c in value_cols], higher) if rank else {}
        cells = [str(row["Dataset"]).replace("_", r"\_")]
        cells += [style_text(fmt_cell(row[c], kind), styles.get(i), "latex") for i, c in enumerate(value_cols)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, folder: Path, stem: str, title: str, kind: str, rank: bool = False, higher: bool = True, float_format: str = "%.6f") -> None:
    safe_to_csv(df, folder / f"{stem}.csv", index=False, encoding="utf-8-sig", float_format=float_format)
    safe_write_text(folder / f"{stem}.html", to_html(df, title, kind, rank, higher))


def save_all_tables(summary: pd.DataFrame, robust: pd.DataFrame, tables_dir: Path) -> None:
    ensure_dir(tables_dir)

    # Diagnostic files help locate missing cells during experiments.
    safe_to_csv(summary, tables_dir / "00_experiment_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    diag_cols = ["Dataset", "method_display", "exists", "has_metrics", "has_predictions", "f1", "accuracy", "experiment_dir", "metrics_path", "predictions_path"]
    safe_to_csv(summary[diag_cols], tables_dir / "00_loading_diagnostics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 1. Overall performance.
    save_table(ratio_to_percent(make_matrix(summary, "f1", MAIN_METHODS, "Average")), tables_dir, "01_overall_f1", "F1 scores across datasets", "percent", rank=True, higher=True, float_format="%.1f")
    save_table(ratio_to_percent(make_matrix(summary, "accuracy", MAIN_METHODS, "Average")), tables_dir, "02_overall_accuracy", "Accuracy across datasets", "percent", rank=True, higher=True, float_format="%.1f")

    # 2. Hierarchical decision effects, RKLEM only.
    save_table(ratio_to_percent(rklem_pair(summary, "rule_coverage", "RuleCoverage", "gray_ratio", "GrayRatio")), tables_dir, "03_rklem_rule_coverage_gray_ratio", "RKLEM RuleCoverage and GrayRatio", "percent", float_format="%.1f")
    save_table(ratio_to_percent(rklem_pair(summary, "rule_accuracy", "RuleAccuracy", "gray_accuracy", "GrayAccuracy")), tables_dir, "04_rklem_rule_gray_accuracy", "RKLEM RuleAccuracy and GrayAccuracy", "percent", float_format="%.1f")
    save_table(rklem_pair(summary, "error_rule", "Error_rule", "error_gray", "Error_gray", final_row="Sum"), tables_dir, "05_rklem_rule_gray_errors", "RKLEM Error_rule and Error_gray", "count", float_format="%.0f")

    # 3. Runtime efficiency. TotalTime uses Sum; AvgTime uses Average.
    save_table(make_matrix(summary, "total_time", MAIN_METHODS, "Sum"), tables_dir, "06_runtime_total_time", "TotalTime across datasets", "time_total", rank=True, higher=False, float_format="%.4f")
    save_table(make_matrix(summary, "avg_time", MAIN_METHODS, "Average"), tables_dir, "07_runtime_avg_time", "AvgTime across datasets", "time_avg", rank=True, higher=False, float_format="%.6f")

    # 4. Robustness analysis.
    save_table(ratio_to_percent(make_matrix(robust, "f1", ROBUSTNESS_METHODS, "Average")), tables_dir, "08_robustness_f1", "RKLEM F1 scores with different LLM backbones", "percent", rank=True, higher=True, float_format="%.1f")


def parse_args():
    p = argparse.ArgumentParser(description="Generate table-only outputs for REG-LLM-EM experiments.")
    p.add_argument("--result-root", type=str, default=r"D:/bysj/API_test/result", help="Root result directory.")
    p.add_argument("--output-dir", type=str, default="visualization_output", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    tables_dir = Path(args.output_dir) / "tables"

    summary = collect_all(result_root, MAIN_METHODS)
    robust = collect_all(result_root, ROBUSTNESS_METHODS)
    save_all_tables(summary, robust, tables_dir)

    print(f"[Done] Tables directory: {tables_dir.resolve()}")
    print(f"[Done] Main rows: {len(summary)}")
    print(f"[Done] Robustness rows: {len(robust)}")


if __name__ == "__main__":
    main()
