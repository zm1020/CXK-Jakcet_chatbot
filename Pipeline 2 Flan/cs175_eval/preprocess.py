from __future__ import annotations

import pandas as pd

from .config import OUTERWEAR_TYPES
from .utils import safe_lower

def infer_product_type(name: str, desc: str = "", kw: str = "") -> str:
    text = f"{name or ''} {desc or ''} {kw or ''}".lower()

    if "parka" in text:
        return "parka"
    if "jacket" in text:
        return "jacket"
    if "vest" in text:
        return "vest"
    if "hoody" in text or "hoodie" in text:
        return "hoody"
    if "sweater" in text:
        return "sweater"
    if "cap" in text or "hat" in text:
        return "cap"
    if "glove" in text:
        return "gloves"
    if "boot" in text:
        return "boots"
    if "beanie" in text:
        return "beanie"
    return "other"

def infer_gender(value) -> str:
    value = safe_lower(value)
    if "men" in value:
        return "men"
    if "women" in value:
        return "women"
    if "kid" in value:
        return "kids"
    return "unknown"

def tei_bucket(tei) -> str:
    if pd.isna(tei):
        return "tei_unknown"
    try:
        tei = int(tei)
    except Exception:
        return "tei_unknown"
    if tei >= 4:
        return "tei_high"
    if tei >= 2:
        return "tei_medium"
    return "tei_low"

def build_synthetic_tags(row: pd.Series) -> str:
    tags = [
        row.get("product_type", ""),
        row.get("gender_norm", ""),
        row.get("tei_bucket", ""),
        "outerwear" if bool(row.get("is_outerwear", False)) else "",
    ]
    return " ".join(str(x) for x in tags if str(x).strip())

def preprocess_products(products_df: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
    kw_agg = (
        keywords_df.groupby("product_id")["keyword"]
        .apply(lambda s: " | ".join(sorted(set(str(x) for x in s if pd.notna(x)))))
        .reset_index()
        .rename(columns={"product_id": "id", "keyword": "keyword_text"})
    )

    products = products_df.merge(kw_agg, on="id", how="left")
    products["keyword_text"] = products["keyword_text"].fillna("")

    products["full_text"] = (
        products["name"].fillna("") + " " +
        products["description"].fillna("") + " " +
        products["keyword_text"].fillna("")
    ).str.lower()

    products["product_type"] = products.apply(
        lambda r: infer_product_type(r.get("name", ""), r.get("description", ""), r.get("keyword_text", "")),
        axis=1,
    )
    products["is_outerwear"] = products["product_type"].isin(OUTERWEAR_TYPES)
    products["gender_norm"] = products["gender"].apply(infer_gender)
    products["tei_bucket"] = products["tei_level"].apply(tei_bucket)
    products["synthetic_tags"] = products.apply(build_synthetic_tags, axis=1)

    products["retrieval_text"] = (
        products["name"].fillna("").str.lower() + " " +
        products["name"].fillna("").str.lower() + " " +
        products["name"].fillna("").str.lower() + " " +
        products["product_type"].fillna("").str.lower() + " " +
        products["product_type"].fillna("").str.lower() + " " +
        products["keyword_text"].fillna("").str.lower() + " " +
        products["keyword_text"].fillna("").str.lower() + " " +
        products["synthetic_tags"].fillna("").str.lower() + " " +
        products["description"].fillna("").str.lower()
    )

    return products

def dataset_summary(products: pd.DataFrame, keywords_df: pd.DataFrame) -> dict:
    return {
        "num_products": len(products),
        "num_keywords_rows": len(keywords_df),
        "num_outerwear_products": int(products["is_outerwear"].sum()),
        "num_non_outerwear_products": int((~products["is_outerwear"]).sum()),
        "num_with_tei": int(products["tei_level"].notna().sum()),
        "min_price": float(products["price"].min()) if "price" in products.columns else None,
        "max_price": float(products["price"].max()) if "price" in products.columns else None,
        "avg_price": float(products["price"].mean()) if "price" in products.columns else None,
    }
