"""Run both baselines on the development split and log the results.

Phase 2 deliverable. One command evaluates Baseline 1 and Baseline 2 with
``src.evaluate`` on the IDs in ``data/splits/dev.txt``, writes the per-image CSV and
summary JSON for each under ``experiments/runs/``, and appends one row per metric to
``experiments/experiment_log.csv``.

Usage (from the repo root):

    python -m src.run_baselines --image-dir <trainval JPEGImages> \\
        --annotation-dir <trainval Annotations> --author <github-username>

Never pass test-set folders or ``test.txt`` here: tuning and baselines use dev only.
"""

import argparse
import csv
import datetime as dt
import json
import time
from collections.abc import Sequence
from pathlib import Path

from src.config import load_pipeline_config
from src.evaluate import EvaluationSummary, evaluate_dataset, read_ids_file
from src.evaluate import write_evaluation

DEFAULT_CONFIGS = ("configs/baseline_trivial.yaml", "configs/baseline_simple.yaml")
LOG_COLUMNS = (
    "run_id", "date", "author", "pipeline", "stage", "method", "params", "seed",
    "time_s", "metric_name", "metric_value_dev", "notes",
)


def summary_rows(
    summary: EvaluationSummary,
    config: dict,
    *,
    run_id: str,
    author: str,
    date: str,
    time_s: float,
) -> list[dict[str, str]]:
    """Turn one evaluation summary into experiment-log rows, one per metric.

    Args:
        summary: Result of ``evaluate_dataset`` on the dev split.
        config: The pipeline configuration that was evaluated.
        run_id: Identifier shared by every row of this run.
        author: GitHub username of the person who ran it.
        date: ISO date (YYYY-MM-DD).
        time_s: Wall-clock seconds for the whole evaluation.

    Returns:
        List of dicts with the ``experiment_log.csv`` columns.
    """
    d03, d05 = summary.detection_at_0_3, summary.detection_at_0_5
    metrics = {
        "box_union_iou_mean": summary.mean_iou,
        "box_union_iou_std": summary.std_iou,
        "f1_iou0.3": d03["f1"],
        "precision_iou0.3": d03["precision"],
        "recall_iou0.3": d03["recall"],
        "f1_iou0.5": d05["f1"],
        "count_error_mean": summary.mean_count_error,
        "time_per_image_s": time_s / summary.image_count,
    }
    params = {k: v for k, v in config.items() if k not in ("name", "seed")}
    note = f"dev split, {summary.image_count} images; worst {summary.worst_image_id}"
    return [
        {
            "run_id": run_id,
            "date": date,
            "author": author,
            "pipeline": str(config.get("name", "unnamed")),
            "stage": "full",
            "method": str(config.get("segmentation", {}).get("method", "")),
            "params": json.dumps(params, sort_keys=True),
            "seed": str(config.get("seed", 0)),
            "time_s": f"{time_s:.2f}",
            "metric_name": name,
            "metric_value_dev": f"{value:.4f}",
            "notes": note,
        }
        for name, value in metrics.items()
    ]


def append_log(log_path: str | Path, rows: list[dict[str, str]]) -> None:
    """Append rows to the experiment log, writing the header if the file is new."""
    path = Path(log_path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate each baseline config on dev, save outputs and append log rows."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, help="trainval image folder")
    parser.add_argument("--annotation-dir", required=True, help="trainval XML folder")
    parser.add_argument("--ids-file", default="data/splits/dev.txt")
    parser.add_argument("--author", required=True, help="your GitHub username")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--log", default="experiments/experiment_log.csv")
    args = parser.parse_args(argv)

    if Path(args.ids_file).name != "dev.txt":
        parser.error("baselines run on the dev split only (data/splits/dev.txt)")

    ids = read_ids_file(args.ids_file)
    date = dt.date.today().isoformat()
    for config_path in args.configs:
        config = load_pipeline_config(config_path)
        name = str(config.get("name", Path(config_path).stem))
        run_id = f"{date}-{name}"
        start = time.perf_counter()
        summary = evaluate_dataset(args.image_dir, args.annotation_dir, config, ids=ids)
        elapsed = time.perf_counter() - start
        write_evaluation(Path(args.output_root) / run_id, summary)
        rows = summary_rows(summary, config, run_id=run_id, author=args.author,
                            date=date, time_s=elapsed)
        append_log(args.log, rows)
        print(f"{name}: IoU {summary.mean_iou:.4f} ± {summary.std_iou:.4f}, "
              f"F1@0.3 {summary.detection_at_0_3['f1']:.4f}, "
              f"count error {summary.mean_count_error:.3f}, {elapsed:.1f}s")
    print(f"Logged to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
