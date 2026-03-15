from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import get_eval_df

def export_project_outputs(output_dir: str, outputs: dict[str, pd.DataFrame]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    outputs["results_df"].to_csv(path / "full_results.csv", index=False)
    outputs["summary_table"].to_csv(path / "summary_table.csv", index=False)
    outputs["constraint_table"].to_csv(path / "constraint_table.csv", index=False)
    outputs["parse_table"].to_csv(path / "parse_table.csv", index=False)
    outputs["category_table"].to_csv(path / "category_table.csv", index=False)
    outputs["error_cases"].to_csv(path / "error_cases.csv", index=False)

def export_expected_vs_actual(system, output_prefix: str = "expected_vs_actual", methods: list[str] | None = None, top_k: int = 5) -> pd.DataFrame:
    if methods is None:
        methods = ["baseline", "flan", "hybrid"]

    eval_df = get_eval_df()
    rows = []

    for _, gold_row in eval_df.iterrows():
        gold = gold_row.to_dict()
        query = gold["query"]

        expected_payload = {
            "expected_type": gold.get("expected_type"),
            "expected_price_max": gold.get("expected_price_max"),
            "expected_use_case": gold.get("expected_use_case"),
            "expected_insulation": gold.get("expected_insulation"),
            "expected_gender": gold.get("expected_gender"),
            "expected_outerwear": gold.get("expected_outerwear"),
        }

        for method in methods:
            result, ranked = system.query(query, method=method, top_k=top_k)

            actual_top_results = []
            for rank_idx, (_, row) in enumerate(ranked.head(top_k).iterrows(), start=1):
                actual_top_results.append({
                    "rank": rank_idx,
                    "name": row.get("name"),
                    "price": None if pd.isna(row.get("price")) else float(row.get("price")),
                    "product_type": row.get("product_type"),
                    "gender_norm": row.get("gender_norm"),
                    "tei_level": None if pd.isna(row.get("tei_level")) else float(row.get("tei_level")),
                    "score": None if pd.isna(row.get("score")) else float(row.get("score")),
                })

            rows.append({
                "query": query,
                "method": method,
                "expected": json.dumps(expected_payload, ensure_ascii=False),
                "actual_constraints": json.dumps(result.get("constraints", {}), ensure_ascii=False),
                "actual_top1_name": ranked.iloc[0]["name"] if len(ranked) > 0 else "",
                "actual_top1_price": None if len(ranked) == 0 or pd.isna(ranked.iloc[0]["price"]) else float(ranked.iloc[0]["price"]),
                "actual_top1_type": ranked.iloc[0]["product_type"] if len(ranked) > 0 else "",
                "actual_top_results": json.dumps(actual_top_results, ensure_ascii=False),
                "num_results": result.get("num_results", 0),
                "runtime_sec": result.get("runtime_sec", np.nan),
                "parse_error": result.get("parse_error", ""),
            })

    df = pd.DataFrame(rows)
    csv_path = f"{output_prefix}.csv"
    db_path = f"{output_prefix}.db"

    df.to_csv(csv_path, index=False)
    conn = sqlite3.connect(db_path)
    df.to_sql("expected_vs_actual", conn, if_exists="replace", index=False)
    conn.close()

    return df
