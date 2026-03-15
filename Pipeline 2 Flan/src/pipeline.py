from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .benchmark import get_eval_df, query_category
from .config import DEFAULT_METHODS, DEFAULT_POOL_K, DEFAULT_TOP_K
from .db import connect_db, load_product_tables
from .evaluation import build_summary_tables, evaluate_constraints, evaluate_retrieval
from .parsing import FlanParser, baseline_constraints, merge_constraints
from .preprocess import dataset_summary, preprocess_products
from .retrieval import Retriever, rank_products

@dataclass
class JacketRecommenderSystem:
    db_path: str
    load_flan: bool = False
    conn: Any = field(init=False)
    products_df: pd.DataFrame = field(init=False)
    keywords_df: pd.DataFrame = field(init=False)
    products: pd.DataFrame = field(init=False)
    retriever: Retriever = field(init=False)
    flan_parser: Optional[FlanParser] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.conn = connect_db(self.db_path)
        self.products_df, self.keywords_df = load_product_tables(self.conn)
        self.products = preprocess_products(self.products_df, self.keywords_df)
        self.retriever = Retriever(self.products)
        if self.load_flan:
            self.flan_parser = FlanParser()
            self.flan_parser.load()

    def get_constraints(self, user_query: str, method: str = "baseline") -> tuple[dict, str, str]:
        flan_raw = ""
        parse_error = ""

        if method == "baseline":
            constraints = baseline_constraints(user_query)
        elif method == "flan":
            try:
                parser = self.flan_parser or FlanParser()
                constraints, flan_raw = parser.parse(user_query)
                self.flan_parser = parser
            except Exception as exc:
                constraints = {}
                parse_error = str(exc)
        elif method == "hybrid":
            rule_c = baseline_constraints(user_query)
            try:
                parser = self.flan_parser or FlanParser()
                flan_c, flan_raw = parser.parse(user_query)
                self.flan_parser = parser
            except Exception as exc:
                flan_c = {}
                parse_error = str(exc)
            constraints = merge_constraints(rule_c, flan_c)
        else:
            raise ValueError("method must be baseline, flan, or hybrid")

        return constraints, flan_raw, parse_error

    def query(self, user_query: str, method: str = "baseline", top_k: int = DEFAULT_TOP_K) -> tuple[dict, pd.DataFrame]:
        t0 = time.time()
        constraints, flan_raw, parse_error = self.get_constraints(user_query, method=method)
        ranked = rank_products(self.retriever, constraints, user_query=user_query, top_k=top_k, pool_k=DEFAULT_POOL_K)
        runtime_sec = round(time.time() - t0, 4)

        result = {
            "method": method,
            "query": user_query,
            "constraints": constraints,
            "num_results": len(ranked),
            "top_names": ranked["name"].tolist() if not ranked.empty else [],
            "runtime_sec": runtime_sec,
            "parse_error": parse_error,
            "flan_raw": flan_raw,
        }
        return result, ranked

    def run_evaluation(self, methods: list[str] | None = None) -> dict[str, pd.DataFrame]:
        if methods is None:
            methods = DEFAULT_METHODS

        eval_df = get_eval_df()
        all_rows = []
        ranked_examples: dict[tuple[str, str], pd.DataFrame] = {}

        for _, gold_row in eval_df.iterrows():
            gold = gold_row.to_dict()
            query = gold["query"]

            for method in methods:
                result, ranked = self.query(query, method=method, top_k=5)

                row = {
                    "query": query,
                    "method": method,
                    "constraints": json.dumps(result["constraints"], ensure_ascii=False),
                    "num_results": result["num_results"],
                    "runtime_sec": result["runtime_sec"],
                    "parse_error": result["parse_error"],
                    "top1_name": ranked.iloc[0]["name"] if len(ranked) > 0 else "",
                    "top1_type": ranked.iloc[0]["product_type"] if len(ranked) > 0 else "",
                    "top1_price": ranked.iloc[0]["price"] if len(ranked) > 0 else pd.NA,
                }
                row.update(evaluate_constraints(result["constraints"], gold))
                row.update(evaluate_retrieval(ranked, gold))
                all_rows.append(row)
                ranked_examples[(query, method)] = ranked.copy()

        results_df = pd.DataFrame(all_rows)
        tables = build_summary_tables(results_df)

        results_df["query_category"] = results_df["query"].apply(query_category)
        category_table = results_df.groupby(["query_category", "method"]).agg(
            top1_type_match=("top1_type_match", "mean"),
            hit_at_3=("hit_at_3", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            avg_relevance_at_5=("avg_relevance_at_5", "mean"),
        ).reset_index()

        error_cases = results_df[
            (results_df["top1_type_match"] < 1) |
            (results_df["constraint_exact_match"] < 1) |
            (results_df["ndcg_at_5"] < 1)
        ].copy()

        return {
            "results_df": results_df,
            "summary_table": tables["summary_table"],
            "constraint_table": tables["constraint_table"],
            "parse_table": tables["parse_table"],
            "category_table": category_table,
            "error_cases": error_cases,
            "ranked_examples": ranked_examples,
        }

    def summary_stats(self) -> dict:
        return dataset_summary(self.products, self.keywords_df)
