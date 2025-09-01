# Dynamic Split Evaluation

This directory will contain evaluation artifacts produced by `dynamic_split.py`.

Running the script:

```bash
python dynamic_split.py --clinical_excel 临床.xlsx --habitat_csv 生境_new.csv --script 0827latest.py --results results
```

Outputs:
- `split_metrics.csv` – per-split metrics.
- `best_split_assignments.csv` – `train` flags for the best split.
- `split_*/` – raw outputs for each evaluated split.

These files are placeholders until the script is executed in an environment
with the required dependencies (pandas, scikit-learn, etc.).
