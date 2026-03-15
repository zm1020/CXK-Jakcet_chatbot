MODEL_NAME = "google/flan-t5-base"
DEFAULT_TOP_K = 5
DEFAULT_POOL_K = 40
DEFAULT_METHODS = ["baseline", "flan", "hybrid"]

VALID_WARMTH = {"low", "medium", "high"}
VALID_LENGTH = {"short", "mid", "long"}
VALID_INSULATION = {"down", "synthetic"}
VALID_USE_CASE = {"everyday", "hiking", "travel", "extreme_cold"}
VALID_TYPES = {"parka", "jacket", "vest", "hoody", "sweater"}
VALID_GENDERS = {"men", "women", "kids", "unknown"}

OUTERWEAR_TYPES = {"parka", "jacket", "vest", "hoody", "sweater"}

USE_CASE_TERMS = {
    "hiking": ["hiking", "trail", "outdoor", "packable"],
    "everyday": ["everyday", "daily", "commute", "casual", "urban"],
    "travel": ["travel", "packable", "lightweight"],
    "extreme_cold": ["extreme cold", "arctic", "expedition", "polar", "cold", "winter"],
}

TYPE_TERMS = {
    "parka": ["parka"],
    "jacket": ["jacket"],
    "vest": ["vest"],
    "hoody": ["hoody", "hoodie"],
    "sweater": ["sweater"],
}

INSULATION_TERMS = {
    "down": ["down"],
    "synthetic": ["synthetic", "primaloft", "poly"],
}

LENGTH_TERMS = {
    "long": ["long", "parka", "extended"],
    "mid": ["mid", "mid-length"],
    "short": ["short", "cropped"],
}

STOPWORDS = {
    "a", "an", "the", "for", "to", "of", "in", "on", "with", "and", "or",
    "i", "want", "need", "looking", "find", "me", "my", "that", "is",
    "very", "really", "something", "under", "over", "around",
}

WARMTH_QUERY_TERMS = {
    "high": ["very warm", "warm", "winter", "extreme cold", "freezing", "insulated", "heavy", "arctic"],
    "medium": ["warm", "cool weather", "fall", "layering"],
    "low": ["lightweight", "mild weather", "spring", "light"],
}
