from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .config import (
    INSULATION_TERMS,
    LENGTH_TERMS,
    STOPWORDS,
    TYPE_TERMS,
    USE_CASE_TERMS,
    WARMTH_QUERY_TERMS,
)
from .utils import safe_lower

def count_term_hits(text: str, terms: List[str]) -> int:
    t = safe_lower(text)
    return sum(1 for term in terms if term in t)

def tei_score(tei, warmth_level: Optional[str]) -> int:
    if pd.isna(tei) or warmth_level is None:
        return 0
    try:
        tei = int(tei)
    except Exception:
        return 0

    if warmth_level == "high":
        return 2 if tei >= 3 else 0
    if warmth_level == "medium":
        return 1 if tei >= 2 else 0
    if warmth_level == "low":
        return 1 if tei <= 2 else 0
    return 0

def simple_query_terms(user_query: str) -> List[str]:
    toks = re.findall(r"[a-z]+", safe_lower(user_query))
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]

def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", safe_lower(text))

def constraints_to_retrieval_query(user_query: str, constraints: Dict[str, Any]) -> str:
    parts = [safe_lower(user_query)]

    if "product_type" in constraints:
        type_terms = TYPE_TERMS.get(constraints["product_type"], [constraints["product_type"]])
        parts.extend(type_terms)
        parts.extend(type_terms)

    if "use_case" in constraints:
        parts.extend(USE_CASE_TERMS.get(constraints["use_case"], []))

    if "insulation" in constraints:
        parts.extend(INSULATION_TERMS.get(constraints["insulation"], []))

    if "length" in constraints:
        parts.extend(LENGTH_TERMS.get(constraints["length"], []))

    if "warmth_level" in constraints:
        parts.extend(WARMTH_QUERY_TERMS.get(constraints["warmth_level"], []))

    if "gender" in constraints:
        g = constraints["gender"]
        if g == "men":
            parts.extend(["men", "mens", "male"])
        elif g == "women":
            parts.extend(["women", "womens", "female"])
        elif g == "kids":
            parts.extend(["kids", "children"])

    return " ".join(parts).strip()

def normalized_family_name(name: str) -> str:
    name = safe_lower(name)
    name = re.sub(r"\b(men'?s|women'?s|kids?)\b", "", name)
    name = re.sub(r"\b(updated|classic|lite|lightweight|fusion fit|black label)\b", "", name)
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def dedup_candidates(df: pd.DataFrame, per_family_cap: int = 2) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["family_name"] = out["name"].apply(normalized_family_name)
    out["family_rank"] = out.groupby("family_name").cumcount() + 1
    out = out[out["family_rank"] <= per_family_cap].copy()
    return out.drop(columns=["family_rank"])

def apply_common_filters(products: pd.DataFrame, constraints: Dict[str, Any], price_multiplier: float = 1.0) -> pd.DataFrame:
    df = products.copy()

    if "gender" in constraints:
        g = constraints["gender"]
        if g in {"men", "women", "kids"}:
            df = df[(df["gender_norm"] == g) | (df["gender_norm"] == "unknown")].copy()

    query_has_outerwear_intent = any(
        k in constraints for k in ["product_type", "length", "insulation", "warmth_level", "use_case"]
    )
    if query_has_outerwear_intent:
        df = df[df["is_outerwear"]].copy()

    if "price_max" in constraints:
        df = df[df["price"] <= constraints["price_max"] * price_multiplier].copy()

    return df

@dataclass
class Retriever:
    products: pd.DataFrame

    def __post_init__(self) -> None:
        self.bm25_corpus = [tokenize_for_bm25(txt) for txt in self.products["retrieval_text"].fillna("")]
        self.bm25_index = BM25Okapi(self.bm25_corpus)

    def structured_candidate_retrieval(
        self, constraints: Dict[str, Any], user_query: str = "", pool_k: int = 20
    ) -> pd.DataFrame:
        df = apply_common_filters(self.products, constraints, price_multiplier=1.15)

        target_type = constraints.get("product_type")
        if target_type:
            if target_type in {"jacket", "parka"}:
                typed_df = df[df["product_type"].isin({"jacket", "parka"})].copy()
            else:
                typed_df = df[df["product_type"] == target_type].copy()
            if len(typed_df) >= max(6, pool_k // 2):
                df = typed_df

        if "insulation" in constraints:
            target_terms = INSULATION_TERMS.get(constraints["insulation"], [])
            ins_mask = df["full_text"].fillna("").str.lower().apply(lambda txt: any(term in txt for term in target_terms))
            ins_df = df[ins_mask].copy()
            if len(ins_df) >= max(6, pool_k // 2):
                df = ins_df

        scores = []
        q_terms = simple_query_terms(user_query)

        for _, row in df.iterrows():
            text = safe_lower(row.get("full_text", ""))
            s = 0.0

            if "product_type" in constraints:
                target_type = constraints["product_type"]
                if row.get("product_type", "") == target_type:
                    s += 4.0
                elif target_type in {"jacket", "parka"} and row.get("product_type", "") in {"jacket", "parka"}:
                    s += 2.0

            if "use_case" in constraints:
                hits = count_term_hits(text, USE_CASE_TERMS.get(constraints["use_case"], []))
                s += min(3.0, 1.5 * hits)

            if "insulation" in constraints:
                hits = count_term_hits(text, INSULATION_TERMS.get(constraints["insulation"], []))
                s += 2.5 if hits > 0 else 0.0

            if "length" in constraints:
                hits = count_term_hits(text, LENGTH_TERMS.get(constraints["length"], []))
                s += 1.5 if hits > 0 else 0.0

            if "warmth_level" in constraints:
                s += float(tei_score(row.get("tei_level", np.nan), constraints["warmth_level"]))

            if q_terms:
                overlap = sum(1 for t in q_terms if t in text)
                s += min(2.0, overlap * 0.4)

            if "price_max" in constraints and pd.notna(row.get("price", np.nan)):
                if row["price"] <= constraints["price_max"]:
                    s += 0.5
                elif row["price"] <= constraints["price_max"] * 1.15:
                    s += 0.1

            scores.append(s)

        df = df.copy()
        df["structured_rank_score"] = scores
        df = df.sort_values(by=["structured_rank_score", "price"], ascending=[False, True]).reset_index(drop=True)
        return df.head(pool_k)

    def lexical_candidate_retrieval(
        self, constraints: Dict[str, Any], user_query: str = "", pool_k: int = 20
    ) -> pd.DataFrame:
        df = apply_common_filters(self.products, constraints, price_multiplier=1.20)
        if df.empty:
            return df.copy()

        query_text = constraints_to_retrieval_query(user_query, constraints)
        query_tokens = tokenize_for_bm25(query_text) or tokenize_for_bm25(user_query)
        raw_scores = np.asarray(self.bm25_index.get_scores(query_tokens))

        out = df.copy()
        out["bm25_score"] = raw_scores[out.index]
        out = out.sort_values(by=["bm25_score", "price"], ascending=[False, True]).reset_index(drop=True)
        return out.head(pool_k)

    @staticmethod
    def reciprocal_rank_fusion(structured_df: pd.DataFrame, lexical_df: pd.DataFrame, rrf_k: int = 60) -> pd.DataFrame:
        frames = []
        for source_name, df in [("structured", structured_df), ("lexical", lexical_df)]:
            if df.empty:
                continue
            tmp = df.copy().reset_index(drop=True)
            tmp["rank_position"] = np.arange(1, len(tmp) + 1)
            tmp["rrf_component"] = 1.0 / (rrf_k + tmp["rank_position"])
            tmp["source_name"] = source_name
            frames.append(tmp)

        if not frames:
            return structured_df.iloc[0:0].copy()

        combined = pd.concat(frames, ignore_index=True)
        agg = (
            combined.groupby("id", as_index=False)
            .agg({"rrf_component": "sum", "structured_rank_score": "max", "bm25_score": "max"})
            .rename(columns={"rrf_component": "pool_score"})
        )

        merged_source = pd.concat([structured_df, lexical_df], ignore_index=True)
        merged_source = merged_source.drop_duplicates(subset=["id"], keep="first")
        out = merged_source.merge(agg, on="id", how="inner", suffixes=("", "_agg"))

        for col in ["structured_rank_score", "bm25_score"]:
            agg_col = f"{col}_agg"
            if agg_col in out.columns:
                out[col] = out[agg_col].fillna(out.get(col, 0.0)).fillna(0.0)
                out = out.drop(columns=[agg_col])
            elif col not in out.columns:
                out[col] = 0.0

        out["pool_score"] = out["pool_score"].fillna(0.0)
        return out.sort_values(
            by=["pool_score", "structured_rank_score", "bm25_score", "price"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def candidate_retrieval(self, constraints: Dict[str, Any], user_query: str = "", pool_k: int = 40) -> pd.DataFrame:
        structured_k = max(10, pool_k // 2)
        lexical_k = max(10, pool_k // 2)

        structured = self.structured_candidate_retrieval(constraints, user_query=user_query, pool_k=structured_k)
        lexical = self.lexical_candidate_retrieval(constraints, user_query=user_query, pool_k=lexical_k)
        fused = self.reciprocal_rank_fusion(structured, lexical, rrf_k=60)
        fused = dedup_candidates(fused, per_family_cap=2)
        return fused.head(pool_k).reset_index(drop=True)

def compute_score(row: pd.Series, constraints: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    text = safe_lower(row.get("full_text", ""))
    score = 0.0
    detail: Dict[str, float] = {}

    if "product_type" in constraints:
        target_type = constraints["product_type"]
        if row["product_type"] == target_type:
            detail["type_match"] = 3.0
        elif target_type in {"parka", "jacket"} and row["product_type"] in {"parka", "jacket"}:
            detail["type_match"] = 2.0
        else:
            detail["type_match"] = 0.0
        score += detail["type_match"]
    else:
        detail["type_match"] = 0.0

    if "use_case" in constraints:
        hits = count_term_hits(text, USE_CASE_TERMS.get(constraints["use_case"], []))
        detail["use_case_match"] = float(hits)
        score += detail["use_case_match"]
    else:
        detail["use_case_match"] = 0.0

    if "insulation" in constraints:
        hits = count_term_hits(text, INSULATION_TERMS.get(constraints["insulation"], []))
        detail["insulation_match"] = 2.0 if hits > 0 else 0.0
        score += detail["insulation_match"]
    else:
        detail["insulation_match"] = 0.0

    if "length" in constraints:
        hits = count_term_hits(text, LENGTH_TERMS.get(constraints["length"], []))
        detail["length_match"] = 1.5 if hits > 0 else 0.0
        score += detail["length_match"]
    else:
        detail["length_match"] = 0.0

    if "warmth_level" in constraints:
        detail["tei_match"] = float(tei_score(row.get("tei_level", np.nan), constraints["warmth_level"]))
        score += detail["tei_match"]
    else:
        detail["tei_match"] = 0.0

    if "gender" in constraints:
        detail["gender_match"] = 1.0 if row.get("gender_norm", "unknown") == constraints["gender"] else 0.0
        score += detail["gender_match"]
    else:
        detail["gender_match"] = 0.0

    if "price_max" in constraints:
        detail["price_ok"] = 1.0 if row["price"] <= constraints["price_max"] else 0.0
        score += detail["price_ok"]
    else:
        detail["price_ok"] = 0.0

    detail["outerwear_bonus"] = 1.0 if row.get("is_outerwear", False) else 0.0
    score += detail["outerwear_bonus"]
    return score, detail

def rank_products(retriever: Retriever, constraints: Dict[str, Any], user_query: str, top_k: int, pool_k: int) -> pd.DataFrame:
    candidates = retriever.candidate_retrieval(constraints, user_query=user_query, pool_k=pool_k)
    if candidates.empty:
        return candidates.copy()

    scores = []
    details = []
    for _, row in candidates.iterrows():
        s, d = compute_score(row, constraints)
        scores.append(s)
        details.append(d)

    candidates = candidates.copy()
    candidates["score"] = scores
    candidates["score_detail"] = details

    for col in ["pool_score", "structured_rank_score", "bm25_score"]:
        if col not in candidates.columns:
            candidates[col] = 0.0

    candidates = candidates.sort_values(
        by=["score", "pool_score", "structured_rank_score", "bm25_score", "price"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return candidates.head(top_k)
