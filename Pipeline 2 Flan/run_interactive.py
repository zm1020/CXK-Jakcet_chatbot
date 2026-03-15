"""
Interactive console app: type a query and see jacket recommendations.
"""
from __future__ import annotations

import argparse
import sys

from cs175_eval.pipeline import JacketRecommenderSystem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive jacket recommender — enter queries and see results."
    )
    parser.add_argument(
        "--db",
        default="canada_goose.db",
        help="Path to canada_goose.db (default: canada_goose.db)",
    )
    parser.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "flan", "hybrid"],
        help="Parsing method (default: baseline)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    args = parser.parse_args()

    print("Loading recommender system...")
    system = JacketRecommenderSystem(db_path=args.db)
    print(f"Ready. Method: {args.method}, top-k: {args.top_k}")
    print("Enter a query (or 'quit' / 'exit' to stop).\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            sys.exit(0)

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        result, ranked = system.query(query, method=args.method, top_k=args.top_k)

        print()
        if result.get("parse_error"):
            print("Parse note:", result["parse_error"])
        print(f"Constraints: {result['constraints']}")
        print(f"({result['runtime_sec']}s, {result['num_results']} result(s))\n")

        if ranked.empty:
            print("No matching products.")
        else:
            cols = ["name", "price", "product_type", "score"]
            available = [c for c in cols if c in ranked.columns]
            print(ranked[available].head(args.top_k).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
