from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.exporting import export_expected_vs_actual, export_project_outputs
from src.pipeline import JacketRecommenderSystem

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to canada_goose.db")
    parser.add_argument("--output-dir", default="project_outputs")
    parser.add_argument("--load-flan", action="store_true")
    args = parser.parse_args()

    system = JacketRecommenderSystem(db_path=args.db, load_flan=args.load_flan)
    print("Dataset stats:")
    print(json.dumps(system.summary_stats(), indent=2))

    outputs = system.run_evaluation()
    export_project_outputs(args.output_dir, outputs)
    export_expected_vs_actual(system, output_prefix=str(Path(args.output_dir) / "expected_vs_actual"))

    print("\nSystem-level comparison:")
    print(outputs["summary_table"].to_string(index=False))

    print(f"\nSaved outputs to {args.output_dir}")

if __name__ == "__main__":
    main()
