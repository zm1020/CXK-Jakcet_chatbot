# CS175 Evaluation Refactor

This is a Python package version of the `cs175_Evaluation.ipynb` notebook.

## Structure

- `cs175_eval/config.py` — constants and defaults
- `cs175_eval/db.py` — SQLite loading helpers
- `cs175_eval/preprocess.py` — product preprocessing
- `cs175_eval/parsing.py` — baseline / FLAN / hybrid query parsing
- `cs175_eval/retrieval.py` — candidate pooling and ranking
- `cs175_eval/evaluation.py` — benchmark metrics
- `cs175_eval/benchmark.py` — evaluation query set
- `cs175_eval/pipeline.py` — end-to-end system wrapper
- `cs175_eval/exporting.py` — CSV / SQLite export helpers
- `run_demo.py` — run one query
- `run_evaluation.py` — run the full benchmark and export outputs

## Quick start

```bash
pip install -r requirements.txt
python run_evaluation.py --db canada_goose.db --output-dir project_outputs
```

To use FLAN or hybrid methods, make sure the Hugging Face model can be downloaded or is already cached.

```bash
python run_demo.py --db canada_goose.db --query "very warm long down jacket for extreme cold under $1500" --method hybrid
```
