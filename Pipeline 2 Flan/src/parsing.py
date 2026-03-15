from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import (
    MODEL_NAME,
    VALID_GENDERS,
    VALID_INSULATION,
    VALID_LENGTH,
    VALID_TYPES,
    VALID_USE_CASE,
    VALID_WARMTH,
)

ALLOWED_KEYS = [
    "warmth_level", "length", "insulation",
    "use_case", "price_max", "product_type", "gender",
]

def detect_price_max(text: str) -> Optional[int]:
    t = text.lower()
    patterns = [
        r"under\s+\$?(\d+)",
        r"below\s+\$?(\d+)",
        r"less than\s+\$?(\d+)",
        r"\$([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            return int(match.group(1))
    return None

def detect_target_type(text: str) -> Optional[str]:
    t = text.lower()
    if "parka" in t:
        return "parka"
    if "jacket" in t:
        return "jacket"
    if "vest" in t:
        return "vest"
    if "hoody" in t or "hoodie" in t:
        return "hoody"
    if "sweater" in t:
        return "sweater"
    return None

def detect_gender(text: str) -> Optional[str]:
    t = text.lower()
    if "men's" in t or "mens" in t or re.search(r"\bmen\b", t):
        return "men"
    if "women's" in t or "womens" in t or re.search(r"\bwomen\b", t):
        return "women"
    if "kids" in t or "kid" in t:
        return "kids"
    return None

def detect_warmth(text: str) -> Optional[str]:
    t = text.lower()
    if any(x in t for x in ["very warm", "extreme cold", "warmest", "arctic", "super warm"]):
        return "high"
    if any(x in t for x in ["not too warm", "light", "lightweight", "cool weather"]):
        return "low"
    if "warm" in t:
        return "medium"
    return None

def detect_length(text: str) -> Optional[str]:
    t = text.lower()
    if any(x in t for x in ["long", "parka", "full length", "extended"]):
        return "long"
    if any(x in t for x in ["mid", "mid-length"]):
        return "mid"
    if any(x in t for x in ["short", "cropped"]):
        return "short"
    return None

def detect_insulation(text: str) -> Optional[str]:
    t = text.lower()
    if "down" in t:
        return "down"
    if any(x in t for x in ["synthetic", "primaloft", "poly"]):
        return "synthetic"
    return None

def detect_use_case(text: str) -> Optional[str]:
    t = text.lower()
    if any(x in t for x in ["extreme cold", "arctic", "polar", "freezing"]):
        return "extreme_cold"
    if any(x in t for x in ["hiking", "trail", "outdoor"]):
        return "hiking"
    if any(x in t for x in ["travel", "packable", "trip"]):
        return "travel"
    if any(x in t for x in ["everyday", "daily", "commute", "casual"]):
        return "everyday"
    return None

def baseline_constraints(user_text: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}

    for key, detector in [
        ("warmth_level", detect_warmth),
        ("length", detect_length),
        ("insulation", detect_insulation),
        ("use_case", detect_use_case),
        ("product_type", detect_target_type),
        ("gender", detect_gender),
    ]:
        value = detector(user_text)
        if value:
            constraints[key] = value

    price_max = detect_price_max(user_text)
    if price_max is not None:
        constraints["price_max"] = price_max

    return constraints

def build_prompt(user_query: str) -> str:
    return f"""
Extract shopping constraints from the user request.

Return exactly one JSON object and nothing else.
The output must start with {{ and end with }}.

Allowed keys:
warmth_level, length, insulation, use_case, price_max, product_type, gender

Allowed values:
warmth_level: low, medium, high
length: short, mid, long
insulation: down, synthetic
use_case: everyday, hiking, travel, extreme_cold
product_type: parka, jacket, vest, hoody, sweater
gender: men, women, kids, unknown

Examples:
Input: very warm long down jacket for extreme cold under $1500
Output: {{"warmth_level":"high","length":"long","insulation":"down","use_case":"extreme_cold","price_max":1500,"product_type":"jacket"}}

Input: men's down vest under $800
Output: {{"insulation":"down","price_max":800,"product_type":"vest","gender":"men"}}

Input: lightweight hoody for casual everyday use
Output: {{"warmth_level":"low","use_case":"everyday","product_type":"hoody"}}

Input: women's travel jacket under $1000
Output: {{"use_case":"travel","price_max":1000,"product_type":"jacket","gender":"women"}}

Input: parka for extreme cold
Output: {{"warmth_level":"high","use_case":"extreme_cold","product_type":"parka","length":"long"}}

Input: {user_query}
Output:
""".strip()

def parse_json_strict(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    if ('":' in text or "':" in text) and not text.startswith("{"):
        candidate = "{" + text
        if not candidate.endswith("}"):
            candidate += "}"
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")

def regex_recover_constraints(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = text.strip().lower()

    patterns = {
        "warmth_level": r'"?warmth_level"?\s*[:=]\s*"?(low|medium|high)"?',
        "length": r'"?length"?\s*[:=]\s*"?(short|mid|long)"?',
        "insulation": r'"?insulation"?\s*[:=]\s*"?(down|synthetic)"?',
        "use_case": r'"?use_case"?\s*[:=]\s*"?(everyday|hiking|travel|extreme_cold)"?',
        "product_type": r'"?product_type"?\s*[:=]\s*"?(parka|jacket|vest|hoody|sweater)"?',
        "gender": r'"?gender"?\s*[:=]\s*"?(men|women|kids|unknown)"?',
        "price_max": r'"?price_max"?\s*[:=]\s*\$?(\d+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, t)
        if match:
            value = match.group(1)
            out[key] = int(value) if key == "price_max" else value
    return out

def normalize_constraints(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in ALLOWED_KEYS:
            continue

        if isinstance(value, str):
            value = value.strip().lower()

        if key == "warmth_level" and value in VALID_WARMTH:
            out[key] = value
        elif key == "length" and value in VALID_LENGTH:
            out[key] = value
        elif key == "insulation" and value in VALID_INSULATION:
            out[key] = value
        elif key == "use_case" and value in VALID_USE_CASE:
            out[key] = value
        elif key == "product_type" and value in VALID_TYPES:
            out[key] = value
        elif key == "gender" and value in VALID_GENDERS:
            out[key] = value
        elif key == "price_max":
            try:
                out[key] = int(str(value).replace("$", "").strip())
            except Exception:
                pass
    return out

@dataclass
class FlanParser:
    model_name: str = MODEL_NAME
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForSeq2SeqLM] = None
    device: Optional[str] = None

    def load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def parse(self, user_query: str, max_new_tokens: int = 80) -> Tuple[Dict[str, Any], str]:
        self.load()
        assert self.tokenizer is not None and self.model is not None and self.device is not None

        prompt = build_prompt(user_query)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

        try:
            data = normalize_constraints(parse_json_strict(text))
            if data:
                return data, text
        except Exception:
            pass

        recovered = normalize_constraints(regex_recover_constraints(text))
        if recovered:
            return recovered, text

        fallback: Dict[str, Any] = {}
        for key, detector in [
            ("product_type", detect_target_type),
            ("gender", detect_gender),
            ("use_case", detect_use_case),
            ("insulation", detect_insulation),
            ("warmth_level", detect_warmth),
            ("length", detect_length),
        ]:
            value = detector(user_query)
            if value:
                fallback[key] = value

        price_max = detect_price_max(user_query)
        if price_max is not None:
            fallback["price_max"] = price_max

        fallback = normalize_constraints(fallback)
        if fallback:
            return fallback, text

        raise ValueError(f"Could not parse JSON from model output: {text[:300]}")

def merge_constraints(rule_c: Dict[str, Any], flan_c: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rule_c)
    for key in ["product_type", "gender", "use_case", "insulation", "warmth_level", "length", "price_max"]:
        rule_val = out.get(key)
        flan_val = flan_c.get(key)
        if rule_val in [None, "", {}] and flan_val not in [None, "", {}]:
            out[key] = flan_val
    return out

def get_constraints(user_query: str, method: str = "baseline", flan_parser: Optional[FlanParser] = None) -> Dict[str, Any]:
    if method == "baseline":
        return baseline_constraints(user_query)
    if method == "flan":
        if flan_parser is None:
            flan_parser = FlanParser()
        try:
            constraints, _ = flan_parser.parse(user_query)
            return constraints
        except Exception:
            return {}
    if method == "hybrid":
        rule_c = baseline_constraints(user_query)
        if flan_parser is None:
            flan_parser = FlanParser()
        try:
            flan_c, _ = flan_parser.parse(user_query)
        except Exception:
            flan_c = {}
        return merge_constraints(rule_c, flan_c)
    raise ValueError("method must be one of: baseline, flan, hybrid")
