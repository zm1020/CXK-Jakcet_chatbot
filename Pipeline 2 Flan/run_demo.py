from __future__ import annotations

import argparse
import json

from src.pipeline import JacketRecommenderSystem

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to canada_goose.db")
    parser.add_argument("--query", required=True, help="Natural-language user query")
    parser.add_argument("--method", default="baseline", choices=["baseline", "flan", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    system = JacketRecommenderSystem(db_path=args.db)
    result, ranked = system.query(args.query, method=args.method, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print(ranked[["name", "price", "product_type", "gender_norm", "tei_level", "score"]].head(args.top_k).to_string(index=False))

if __name__ == "__main__":
    main()
