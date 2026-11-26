"""Dataset evaluation and failure summaries for SolarScan pipelines."""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from src.config import load_pipeline_config
from src.data import IN_SCOPE_CLASSES, load_annotations, load_image
from src.metrics import box_union_iou, count_error, detection_prf
from src.pipeline import PipelineResult, run

PipelineRunner = Callable[[np.ndarray, dict[str, Any]], PipelineResult]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IOU_THRESHOLDS = (0.3, 0.5)


@dataclass(frozen=True)
class EvaluationRecord:
    """Per-image metrics produced by :func:`evaluate_dataset`."""

    image_id: str
    box_union_iou: float
    count_error: int
    predicted_count: int
    ground_truth_count: int
    predicted_boxes: str
    ground_truth_boxes: str
    ground_truth_classes: str
    time_s: float
    tp_iou_0_3: int
    fp_iou_0_3: int
    fn_iou_0_3: int
    precision_0_3: float
    recall_0_3: float
    f1_0_3: float
    tp_iou_0_5: int
    fp_iou_0_5: int
    fn_iou_0_5: int
    precision_0_5: float
    recall_0_5: float
    f1_0_5: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics and the worst/best images for one evaluation run."""

    image_count: int
    mean_iou: float
    std_iou: float
    mean_count_error: float
    best_image_id: str
    best_image_iou: float
    worst_image_id: str
    worst_image_iou: float
    detection_at_0_3: dict[str, int | float]
    detection_at_0_5: dict[str, int | float]
    records: tuple[EvaluationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate metrics and the ten lowest-IoU failure cases."""
        failures = sorted(
            self.records, key=lambda record: (record.box_union_iou, record.image_id)
        )[:10]
        return {
            "image_count": self.image_count,
            "mean_iou": self.mean_iou,
            "std_iou": self.std_iou,
            "mean_count_error": self.mean_count_error,
            "best_image_id": self.best_image_id,
            "best_image_iou": self.best_image_iou,
            "worst_image_id": self.worst_image_id,
            "worst_image_iou": self.worst_image_iou,
            "detection_at_0_3": self.detection_at_0_3,
            "detection_at_0_5": self.detection_at_0_5,
            "lowest_iou_cases": [
                {
                    "image_id": record.image_id,
                    "box_union_iou": record.box_union_iou,
                    "count_error": record.count_error,
                    "ground_truth_count": record.ground_truth_count,
                    "predicted_count": record.predicted_count,
                }
                for record in failures
            ],
        }


def _index_directory(
    directory: Path, suffixes: set[str], description: str
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {directory}")

    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in indexed:
            raise ValueError(f"duplicate {description} ID: {path.stem}")
        indexed[path.stem] = path
    if not indexed:
        raise ValueError(f"no {description} files found in {directory}")
    return indexed


def _select_ids(
    image_ids: set[str],
    annotation_ids: set[str],
    ids: Sequence[str] | None,
) -> list[str]:
    missing_images = sorted(annotation_ids - image_ids)
    missing_annotations = sorted(image_ids - annotation_ids)
    if missing_images or missing_annotations:
        details = []
        if missing_images:
            details.append(f"missing images for annotations: {missing_images[:5]}")
        if missing_annotations:
            details.append(f"missing annotations for images: {missing_annotations[:5]}")
        raise ValueError("image/annotation IDs do not match; " + "; ".join(details))

    if ids is None:
        return sorted(image_ids)
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise ValueError("ids must be a sequence of image IDs")
    if not ids:
        raise ValueError("ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    unknown_ids = sorted(set(ids) - image_ids)
    if unknown_ids:
        raise ValueError(f"requested image IDs were not found: {unknown_ids[:5]}")
    return list(ids)


def _precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _build_summary(records: list[EvaluationRecord]) -> EvaluationSummary:
    if not records:
        raise ValueError("evaluation requires at least one image")
    ious = np.asarray([record.box_union_iou for record in records], dtype=np.float64)
    best = min(records, key=lambda record: (-record.box_union_iou, record.image_id))
    worst = min(records, key=lambda record: (record.box_union_iou, record.image_id))

    detection_metrics = []
    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        true_positives = sum(
            getattr(record, f"tp_iou_{suffix}") for record in records
        )
        false_positives = sum(
            getattr(record, f"fp_iou_{suffix}") for record in records
        )
        false_negatives = sum(
            getattr(record, f"fn_iou_{suffix}") for record in records
        )
        precision, recall, f1 = _precision_recall_f1(
            true_positives, false_positives, false_negatives
        )
        detection_metrics.append(
            {
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return EvaluationSummary(
        image_count=len(records),
        mean_iou=float(np.mean(ious)),
        std_iou=float(np.std(ious)),
        mean_count_error=float(
            np.mean([record.count_error for record in records], dtype=np.float64)
        ),
        best_image_id=best.image_id,
        best_image_iou=best.box_union_iou,
        worst_image_id=worst.image_id,
        worst_image_iou=worst.box_union_iou,
        detection_at_0_3=detection_metrics[0],
        detection_at_0_5=detection_metrics[1],
        records=tuple(records),
    )


def evaluate_dataset(
    image_dir: str | Path,
    annotation_dir: str | Path,
    config: Mapping[str, Any],
    *,
    ids: Sequence[str] | None = None,
    pipeline_runner: PipelineRunner | None = None,
) -> EvaluationSummary:
    """Evaluate configured predictions against Pascal VOC box annotations.

    Image and annotation IDs must match exactly. Detection metrics aggregate
    true/false positives and negatives over the selected images (micro average).
    Only ground-truth boxes of in-scope classes are scored: ``src.data.IN_SCOPE_CLASSES``
    by default, or ``config["evaluation"]["classes"]`` if the config sets it.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    images = _index_directory(Path(image_dir), IMAGE_SUFFIXES, "image")
    annotations = _index_directory(Path(annotation_dir), {".xml"}, "annotation")
    selected_ids = _select_ids(set(images), set(annotations), ids)
    runner = run if pipeline_runner is None else pipeline_runner
    evaluation = config.get("evaluation") or {}
    classes = frozenset(evaluation.get("classes", IN_SCOPE_CLASSES))

    records = []
    for image_id in selected_ids:
        image = load_image(images[image_id])
        ground_truth = [
            (name, box)
            for name, box in load_annotations(annotations[image_id])
            if name in classes
        ]
        ground_truth_boxes = [box for _, box in ground_truth]

        start = time.perf_counter()
        result = runner(image, dict(config))
        elapsed = time.perf_counter() - start
        if not isinstance(result, PipelineResult):
            raise TypeError("pipeline runner must return PipelineResult")
        if (
            not isinstance(result.mask, np.ndarray)
            or result.mask.dtype != np.bool_
            or result.mask.shape != image.shape
        ):
            raise ValueError(
                f"pipeline mask for {image_id} must be a boolean array matching "
                "the input image shape"
            )

        per_threshold = [
            detection_prf(result.boxes, ground_truth_boxes, iou_thr=threshold)
            for threshold in IOU_THRESHOLDS
        ]
        threshold_metrics = [
            (*counts, *_precision_recall_f1(*counts)) for counts in per_threshold
        ]
        (
            (tp_03, fp_03, fn_03, precision_03, recall_03, f1_03),
            (tp_05, fp_05, fn_05, precision_05, recall_05, f1_05),
        ) = threshold_metrics
        records.append(
            EvaluationRecord(
                image_id=image_id,
                box_union_iou=box_union_iou(
                    result.boxes, ground_truth_boxes, image.shape
                ),
                count_error=count_error(len(result.boxes), len(ground_truth_boxes)),
                predicted_count=len(result.boxes),
                ground_truth_count=len(ground_truth_boxes),
                predicted_boxes=json.dumps(result.boxes),
                ground_truth_boxes=json.dumps(ground_truth_boxes),
                ground_truth_classes=json.dumps([name for name, _ in ground_truth]),
                time_s=elapsed,
                tp_iou_0_3=tp_03,
                fp_iou_0_3=fp_03,
                fn_iou_0_3=fn_03,
                precision_0_3=precision_03,
                recall_0_3=recall_03,
                f1_0_3=f1_03,
                tp_iou_0_5=tp_05,
                fp_iou_0_5=fp_05,
                fn_iou_0_5=fn_05,
                precision_0_5=precision_05,
                recall_0_5=recall_05,
                f1_0_5=f1_05,
            )
        )
    return _build_summary(records)


def read_ids_file(path: str | Path) -> list[str]:
    """Read unique image IDs from a newline-delimited split manifest."""
    ids_path = Path(path)
    if not ids_path.is_file():
        raise FileNotFoundError(f"image ID manifest does not exist: {ids_path}")
    ids = [
        line.split("#", maxsplit=1)[0].strip()
        for line in ids_path.read_text(encoding="utf-8").splitlines()
    ]
    ids = [image_id for image_id in ids if image_id]
    if not ids:
        raise ValueError(f"image ID manifest is empty: {ids_path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"image ID manifest contains duplicate IDs: {ids_path}")
    return ids


def write_evaluation(
    output_dir: str | Path,
    summary: EvaluationSummary,
) -> tuple[Path, Path]:
    """Write per-image CSV metrics and a JSON aggregate/failure summary."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "per_image.csv"
    summary_path = destination / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[field.name for field in fields(EvaluationRecord)],
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in summary.records)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, help="directory containing images")
    parser.add_argument(
        "--annotation-dir", required=True, help="directory containing Pascal VOC XML"
    )
    parser.add_argument("--config", required=True, help="pipeline YAML configuration")
    parser.add_argument(
        "--ids-file",
        help="optional newline-delimited manifest; evaluate only these image IDs",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/evaluation",
        help="directory for per-image CSV and summary JSON",
    )
    args = parser.parse_args(argv)

    summary = evaluate_dataset(
        args.image_dir,
        args.annotation_dir,
        load_pipeline_config(args.config),
        ids=read_ids_file(args.ids_file) if args.ids_file else None,
    )
    csv_path, summary_path = write_evaluation(args.output_dir, summary)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print(f"Per-image metrics: {csv_path}")
    print(f"Aggregate summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
