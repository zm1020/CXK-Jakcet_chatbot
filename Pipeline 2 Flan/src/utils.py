from __future__ import annotations

import pandas as pd

def safe_lower(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()
