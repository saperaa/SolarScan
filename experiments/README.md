# Experiments

- `experiment_log.csv` — one row per run. Columns: run_id, date (YYYY-MM-DD), author
  (GitHub username), pipeline, stage, method, params (JSON), seed, time_s, metric_name,
  metric_value_dev, notes.
- `runs/` — ignored per-image evaluator output and failure summaries; do not commit raw data.
- `figures/` — figures used in the report (stage-by-stage images, spectra, plots).
- Tuning is one factor at a time from a fixed reference configuration, on the dev set only.
- Evaluate the held-out test manifest once, after the final configuration has been selected.

## Baselines (Phase 2)

Run both baselines on the dev split and append their rows to `experiment_log.csv`:

```
python -m src.run_baselines --image-dir <trainval JPEGImages> \
    --annotation-dir <trainval Annotations> --author <github-username>
```

Per-image CSVs and summaries go to `experiments/runs/<date>-<baseline>/` (git-ignored);
commit only the log rows.

## Studies

- Restoration PSNR per noise type: `python -m src.psnr_study --out experiments/psnr_<name>.csv`.
  Synthetic runs are dry runs only; log real dev-set runs in `experiment_log.csv`.

## Dev baseline results (26 Nov 2026)

`python -m src.run_baselines` on all 3,150 dev images (8 in-scope classes). Full rows in
`experiment_log.csv`; per-image CSVs were written to `experiments/runs/` (git-ignored).

| Baseline | Box-union IoU (mean ± std) | F1 @ IoU 0.3 | Count error | Boxes per image (median) | Time / image |
|---|---|---|---|---|---|
| 1 — threshold 127 | 0.273 ± 0.371 | 0.001 | 1,789.6 | 1,544 | 0.05 s |
| 2 — Otsu + 3×3 opening | 0.273 ± 0.370 | 0.008 | 93.0 | 106 | 0.03 s |

Reading: only 4 dev images are defect-free, and both baselines mark most of the cell dark,
so the union box covers the defect (IoU ≈ 0.27) while almost every individual box is a false
positive (F1 ≈ 0). Opening cuts the box count 17×, but doesn't fix localisation. These are
the numbers Pipelines A–C must beat: fewer, tighter boxes.
