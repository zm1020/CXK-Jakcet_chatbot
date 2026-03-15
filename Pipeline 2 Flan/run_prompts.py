"""
Run the jacket recommender on prompts from a text file.
Prompts are separated by a blank line (one or more empty lines between prompts).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from cs175_eval.config import DEFAULT_METHODS
from cs175_eval.pipeline import JacketRecommenderSystem


def load_prompts(path: str) -> list[str]:
    """Read prompts from file; split on blank lines (one or more newlines)."""
    text = Path(path).read_text(encoding="utf-8")
    # Split on one or more blank lines (newlines with optional spaces)
    blocks = re.split(r"\n\s*\n", text)
    prompts = [s.strip() for s in blocks if s.strip()]
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run recommender on prompts from a file (prompts separated by blank line)."
    )
    parser.add_argument("--db", required=True, help="Path to canada_goose.db")
    parser.add_argument(
        "--prompts",
        default="prompts.txt",
        help="Path to prompts file (default: prompts.txt)",
    )
    parser.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "flan", "hybrid"],
        help="Parsing method (default: baseline)",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run all methods (baseline, flan, hybrid) for each prompt",
    )
    parser.add_argument("--output-dir", default="project_outputs", help="Directory for output CSVs")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query")
    parser.add_argument("--load-flan", action="store_true", help="Pre-load FLAN model at startup")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found in", args.prompts)
        return

    print(f"Loaded {len(prompts)} prompt(s) from {args.prompts}")
    methods = DEFAULT_METHODS if args.all_methods else [args.method]

    system = JacketRecommenderSystem(db_path=args.db, load_flan=args.load_flan)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, query in enumerate(prompts):
        for method in methods:
            result, ranked = system.query(query, method=method, top_k=args.top_k)
            row = {
                "prompt_index": i + 1,
                "query": query,
                "method": method,
                "constraints": json.dumps(result["constraints"], ensure_ascii=False),
                "num_results": result["num_results"],
                "runtime_sec": result["runtime_sec"],
                "parse_error": result.get("parse_error", ""),
                "top_names": " | ".join(result.get("top_names", [])),
            }
            if not ranked.empty:
                row["top1_name"] = ranked.iloc[0]["name"]
                row["top1_price"] = ranked.iloc[0]["price"]
                row["top1_type"] = ranked.iloc[0]["product_type"]
            else:
                row["top1_name"] = row["top1_price"] = row["top1_type"] = ""
            rows.append(row)

            print(f"\n[{i + 1}/{len(prompts)}] {method}: {query[:60]}{'...' if len(query) > 60 else ''}")
            print(f"  runtime: {result['runtime_sec']}s, results: {result['num_results']}")
            if not ranked.empty:
                print(ranked[["name", "price", "product_type", "score"]].head(args.top_k).to_string(index=False))

    df = pd.DataFrame(rows)
    out_csv = out_dir / "prompts_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
