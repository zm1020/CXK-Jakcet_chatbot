from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import INSULATION_TERMS, USE_CASE_TERMS
from .utils import safe_lower

def type_family_match(pred_type: str, expected_type: str) -> bool:
    pred_type = safe_lower(pred_type)
    expected_type = safe_lower(expected_type)
    if pred_type == expected_type:
        return True
    if expected_type in {"jacket", "parka"} and pred_type in {"jacket", "parka"}:
        return True
    return False

def evaluate_constraints(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    expected_fields = {
        "price_max": gold.get("expected_price_max"),
        "product_type": gold.get("expected_type"),
        "use_case": gold.get("expected_use_case"),
        "insulation": gold.get("expected_insulation"),
        "gender": gold.get("expected_gender"),
    }

    correct_fields = 0
    total_expected = 0

    if expected_fields["price_max"] is not None:
        total_expected += 1
        metrics["price_detect"] = float("price_max" in pred)
        metrics["price_exact"] = float(pred.get("price_max") == expected_fields["price_max"])
        correct_fields += int(metrics["price_exact"])
    else:
        metrics["price_detect"] = np.nan
        metrics["price_exact"] = np.nan

    if expected_fields["product_type"] is not None:
        total_expected += 1
        metrics["type_detect"] = float("product_type" in pred)
        metrics["type_match"] = float(type_family_match(pred.get("product_type", ""), expected_fields["product_type"]))
        correct_fields += int(metrics["type_match"])
    else:
        metrics["type_detect"] = np.nan
        metrics["type_match"] = np.nan

    if expected_fields["use_case"] is not None:
        total_expected += 1
        metrics["use_case_detect"] = float("use_case" in pred)
        metrics["use_case_match"] = float(pred.get("use_case") == expected_fields["use_case"])
        correct_fields += int(metrics["use_case_match"])
    else:
        metrics["use_case_detect"] = np.nan
        metrics["use_case_match"] = np.nan

    if expected_fields["insulation"] is not None:
        total_expected += 1
        metrics["ins_detect"] = float("insulation" in pred)
        metrics["ins_match"] = float(pred.get("insulation") == expected_fields["insulation"])
        correct_fields += int(metrics["ins_match"])
    else:
        metrics["ins_detect"] = np.nan
        metrics["ins_match"] = np.nan

    if expected_fields["gender"] is not None:
        total_expected += 1
        metrics["gender_detect"] = float("gender" in pred)
        metrics["gender_match"] = float(pred.get("gender") == expected_fields["gender"])
        correct_fields += int(metrics["gender_match"])
    else:
        metrics["gender_detect"] = np.nan
        metrics["gender_match"] = np.nan

    metrics["constraint_exact_match"] = float(correct_fields == total_expected) if total_expected > 0 else np.nan
    metrics["constraint_field_accuracy"] = float(correct_fields / total_expected) if total_expected > 0 else np.nan
    metrics["num_predicted_fields"] = len(pred)
    return metrics

def relevance_label(row: pd.Series, gold: Dict[str, Any]) -> int:
    score = 0

    if gold.get("expected_outerwear", False):
        if row.get("is_outerwear", False):
            score += 1
        else:
            return 0

    if "expected_type" in gold and pd.notna(gold["expected_type"]):
        if type_family_match(row.get("product_type", ""), gold["expected_type"]):
            score += 1

    if "expected_price_max" in gold and pd.notna(gold["expected_price_max"]):
        if row.get("price", 1e9) <= gold["expected_price_max"]:
            score += 1

    if "expected_gender" in gold and pd.notna(gold["expected_gender"]):
        if row.get("gender_norm", "") == gold["expected_gender"]:
            score += 1

    text = safe_lower(row.get("full_text", ""))
    if "expected_use_case" in gold and pd.notna(gold["expected_use_case"]):
        uc = gold["expected_use_case"]
        if any(term in text for term in USE_CASE_TERMS.get(uc, [])):
            score += 1

    if "expected_insulation" in gold and pd.notna(gold["expected_insulation"]):
        ins = gold["expected_insulation"]
        if any(term in text for term in INSULATION_TERMS.get(ins, [])):
            score += 1

    if score >= 5:
        return 3
    if score >= 3:
        return 2
    if score >= 1:
        return 1
    return 0

def dcg_at_k(relevances: List[int], k: int) -> float:
    rels = relevances[:k]
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(relevances: List[int], k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0

def mrr_at_k(relevances: List[int], k: int) -> float:
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0

def hit_at_k(relevances: List[int], k: int) -> float:
    return float(any(rel > 0 for rel in relevances[:k]))

def evaluate_retrieval(ranked: pd.DataFrame, gold: Dict[str, Any]) -> Dict[str, float]:
    if ranked.empty:
        return {
            "has_results": 0.0,
            "outerwear_precision_at_5": 0.0,
            "price_satisfaction_at_5": 0.0,
            "top1_type_match": 0.0,
            "top1_gender_match": np.nan,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
            "avg_relevance_at_5": 0.0,
        }

    topk = ranked.head(5).copy()
    rels = [relevance_label(row, gold) for _, row in topk.iterrows()]
    top1 = topk.iloc[0]

    price_sat = (
        (topk["price"] <= gold["expected_price_max"]).mean()
        if "expected_price_max" in gold and pd.notna(gold["expected_price_max"])
        else np.nan
    )
    top1_type_ok = (
        float(type_family_match(top1["product_type"], gold["expected_type"]))
        if "expected_type" in gold and pd.notna(gold["expected_type"])
        else np.nan
    )
    top1_gender_ok = (
        float(top1["gender_norm"] == gold["expected_gender"])
        if "expected_gender" in gold and pd.notna(gold["expected_gender"])
        else np.nan
    )

    return {
        "has_results": 1.0,
        "outerwear_precision_at_5": float(topk["is_outerwear"].mean()),
        "price_satisfaction_at_5": float(price_sat) if not pd.isna(price_sat) else np.nan,
        "top1_type_match": float(top1_type_ok) if not pd.isna(top1_type_ok) else np.nan,
        "top1_gender_match": float(top1_gender_ok) if not pd.isna(top1_gender_ok) else np.nan,
        "hit_at_1": hit_at_k(rels, 1),
        "hit_at_3": hit_at_k(rels, 3),
        "mrr_at_5": mrr_at_k(rels, 5),
        "ndcg_at_5": ndcg_at_k(rels, 5),
        "avg_relevance_at_5": float(np.mean(rels)),
    }

def build_summary_tables(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_table = results_df.groupby("method").agg(
        avg_runtime_sec=("runtime_sec", "mean"),
        avg_num_results=("num_results", "mean"),
        has_results_rate=("has_results", "mean"),
        outerwear_precision_at_5=("outerwear_precision_at_5", "mean"),
        price_satisfaction_at_5=("price_satisfaction_at_5", "mean"),
        top1_type_match=("top1_type_match", "mean"),
        hit_at_1=("hit_at_1", "mean"),
        hit_at_3=("hit_at_3", "mean"),
        mrr_at_5=("mrr_at_5", "mean"),
        ndcg_at_5=("ndcg_at_5", "mean"),
        avg_relevance_at_5=("avg_relevance_at_5", "mean"),
    ).reset_index()

    constraint_table = results_df.groupby("method").agg(
        price_detect=("price_detect", "mean"),
        price_exact=("price_exact", "mean"),
        type_detect=("type_detect", "mean"),
        type_match=("type_match", "mean"),
        use_case_detect=("use_case_detect", "mean"),
        use_case_match=("use_case_match", "mean"),
        ins_detect=("ins_detect", "mean"),
        ins_match=("ins_match", "mean"),
        gender_detect=("gender_detect", "mean"),
        gender_match=("gender_match", "mean"),
        constraint_exact_match=("constraint_exact_match", "mean"),
        constraint_field_accuracy=("constraint_field_accuracy", "mean"),
        avg_num_predicted_fields=("num_predicted_fields", "mean"),
    ).reset_index()

    parse_table = results_df.groupby("method").apply(
        lambda g: pd.Series({"parse_failure_rate": (g["parse_error"].fillna("") != "").mean()})
    ).reset_index()

    return {
        "summary_table": summary_table,
        "constraint_table": constraint_table,
        "parse_table": parse_table,
    }
