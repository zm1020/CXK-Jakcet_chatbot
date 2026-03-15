# CS175 Pipeline 2 Flan

## Structure

- `src/config.py` — constants and defaults
- `src/db.py` — SQLite loading helpers
- `src/preprocess.py` — product preprocessing
- `src/parsing.py` — baseline / FLAN / hybrid query parsing
- `src/retrieval.py` — candidate pooling and ranking
- `src/evaluation.py` — benchmark metrics
- `src/benchmark.py` — evaluation query set
- `src/pipeline.py` — end-to-end system wrapper
- `src/exporting.py` — CSV / SQLite export helpers
- `run_demo.py` — run one query
- `run_evaluation.py` — run the full benchmark and export outputs

## Quick start

```bash
pip install -r requirements.txt
python run_evaluation.py --db canada_goose.db --output-dir project_outputs
```

To use FLAN or hybrid methods, we must run this command to cache

```bash
python run_demo.py --db canada_goose.db --query "very warm long down jacket for extreme cold under $1500" --method hybrid
```
