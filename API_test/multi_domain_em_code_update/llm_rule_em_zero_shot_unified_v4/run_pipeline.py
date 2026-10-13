from __future__ import annotations

import argparse
import json

from .llm_client import LLMConfig
from .pipeline import run_full_pipeline

ALL_DATASETS_EXCEPT_AMAZON_GOOGLE = [
    "dblp_acm",
    "dblp_scholar",
    "abt_buy",
    "walmart_amazon",
    "beer",
    "fodors_zagat",
    "itunes_amazon",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_a", required=None)
    parser.add_argument("--table_b", required=None)
    parser.add_argument("--pairs", required=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default="auto")
    parser.add_argument("--datasets",default=None,help="Run multiple datasets. Use 'all' to run all registered datasets except amazon_google, or comma-separated names.",)
    parser.add_argument("--data_root",default="data",help="Root directory containing dataset folders.",)
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--desc_style", default="kv", choices=["kv", "flat"])
    parser.add_argument("--run_tag",default="default_run",help="Subfolder name for each dataset (e.g., model name)",)
    parser.add_argument("--generator_provider", required=True, choices=["ollama", "openai_compatible"])
    parser.add_argument("--generator_model", required=True)
    parser.add_argument("--generator_api_base", default=None)
    parser.add_argument("--generator_api_key", default=None)
    parser.add_argument("--generator_max_tokens", type=int, default=2048)

    parser.add_argument("--gray_provider", required=True, choices=["ollama", "openai_compatible"])
    parser.add_argument("--gray_model", required=True)
    parser.add_argument("--gray_api_base", default=None)
    parser.add_argument("--gray_api_key", default=None)
    parser.add_argument("--gray_max_tokens", type=int, default=1024)

    return parser.parse_args()
args = parse_args()

if not args.datasets:
    missing = []
    if not args.table_a:
        missing.append("--table_a")
    if not args.table_b:
        missing.append("--table_b")
    if not args.pairs:
        missing.append("--pairs")
    if missing:
        raise ValueError(
            "Single-dataset mode requires: " + ", ".join(missing)
        )


def main():
    args = parse_args()
    generator_cfg = LLMConfig(
        args.generator_provider,
        args.generator_model,
        args.generator_api_base,
        args.generator_api_key,
        0.0,
        120,
        args.generator_max_tokens,
    )
    gray_cfg = LLMConfig(
        args.gray_provider,
        args.gray_model,
        args.gray_api_base,
        args.gray_api_key,
        0.0,
        120,
        args.gray_max_tokens,
    )
    if args.datasets:
        if args.datasets.lower() == "all":
            dataset_names = ALL_DATASETS_EXCEPT_AMAZON_GOOGLE
        else:
            dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]

        all_metrics = {}

        for dataset_name in dataset_names:
            print(f"\n===== Running dataset: {dataset_name} =====")

            table_a_path = f"{args.data_root}/{dataset_name}/tableA.csv"
            table_b_path = f"{args.data_root}/{dataset_name}/tableB.csv"
            pairs_path = f"{args.data_root}/{dataset_name}/test.csv"
            save_dir = f"{args.save_dir}/{dataset_name}/{args.run_tag}"

            metrics = run_full_pipeline(
                table_a_path=table_a_path,
                table_b_path=table_b_path,
                pairs_path=pairs_path,
                save_dir=save_dir,
                generator_cfg=generator_cfg,
                gray_cfg=gray_cfg,
                dataset_name=dataset_name,
                lang=args.lang,
                desc_style=args.desc_style,
            )

            all_metrics[dataset_name] = metrics

        print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
        return

    metrics = run_full_pipeline(
        table_a_path=args.table_a,
        table_b_path=args.table_b,
        pairs_path=args.pairs,
        save_dir=args.save_dir,
        generator_cfg=generator_cfg,
        gray_cfg=gray_cfg,
        dataset_name=args.dataset_name,
        lang=args.lang,
        desc_style=args.desc_style,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
