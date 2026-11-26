# Experiments

- `experiment_log.csv` — one row per run. Columns: run_id, date (YYYY-MM-DD), author
  (GitHub username), pipeline, stage, method, params (JSON), seed, time_s, metric_name,
  metric_value_dev, notes.
- `runs/` — ignored per-image evaluator output and failure summaries; do not commit raw data.
- `figures/` — figures used in the report (stage-by-stage images, spectra, plots).
- Tuning is one factor at a time from a fixed reference configuration, on the dev set only.
- Evaluate the held-out test manifest once, after the final configuration has been selected.

## Studies

- Restoration PSNR per noise type: `python -m src.psnr_study --out experiments/psnr_<name>.csv`.
  Synthetic runs are dry runs only; log real dev-set runs in `experiment_log.csv`.
