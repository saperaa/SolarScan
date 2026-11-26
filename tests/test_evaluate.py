"""Tests for dataset-level evaluation using small synthetic fixtures."""

import csv
import json

import cv2
import numpy as np
import pytest

from src.evaluate import evaluate_dataset, read_ids_file, write_evaluation
from src.pipeline import PipelineResult


def _write_annotation(path, box=None):
    objects = ""
    if box is not None:
        x_min, y_min, x_max, y_max = box
        objects = (
            "<object><name>crack</name><bndbox>"
            f"<xmin>{x_min}</xmin><ymin>{y_min}</ymin>"
            f"<xmax>{x_max}</xmax><ymax>{y_max}</ymax>"
            "</bndbox></object>"
        )
    path.write_text(f"<annotation>{objects}</annotation>", encoding="utf-8")


def _make_dataset(tmp_path, *, with_annotation=True):
    image_dir = tmp_path / "images"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir()
    annotation_dir.mkdir()
    image = np.zeros((4, 4), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "cell.jpg"), image)
    if with_annotation:
        _write_annotation(annotation_dir / "cell.xml", (2, 2, 3, 3))
    return image_dir, annotation_dir


def test_evaluate_dataset_reports_box_iou_detection_and_count_metrics(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path)
    expected_box = (1, 1, 2, 2)

    def runner(image, config):
        assert image.shape == (4, 4)
        assert config == {"name": "synthetic"}
        return PipelineResult(
            mask=np.zeros(image.shape, dtype=bool),
            boxes=[expected_box],
        )

    summary = evaluate_dataset(
        image_dir,
        annotation_dir,
        {"name": "synthetic"},
        pipeline_runner=runner,
    )

    assert summary.image_count == 1
    assert summary.mean_iou == 1.0
    assert summary.std_iou == 0.0
    assert summary.mean_count_error == 0.0
    assert summary.best_image_id == summary.worst_image_id == "cell"
    assert summary.detection_at_0_3 == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert summary.detection_at_0_5["f1"] == 1.0


def test_evaluate_dataset_can_select_ids_from_matching_directories(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path)
    selected_ids = ["cell"]

    summary = evaluate_dataset(
        image_dir,
        annotation_dir,
        {},
        ids=selected_ids,
        pipeline_runner=lambda image, config: PipelineResult(
            mask=np.zeros(image.shape, dtype=bool),
            boxes=[],
        ),
    )

    assert summary.image_count == 1
    assert summary.records[0].image_id == "cell"
    assert summary.records[0].box_union_iou == 0.0
    assert summary.records[0].count_error == 1


def test_evaluate_dataset_handles_clean_images_with_no_predictions(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path, with_annotation=False)
    _write_annotation(annotation_dir / "cell.xml")

    summary = evaluate_dataset(
        image_dir,
        annotation_dir,
        {},
        pipeline_runner=lambda image, config: PipelineResult(
            mask=np.zeros(image.shape, dtype=bool),
            boxes=[],
        ),
    )

    assert summary.mean_iou == 1.0
    assert summary.detection_at_0_3["tp"] == 0
    assert summary.detection_at_0_3["f1"] == 0.0


def test_evaluate_dataset_rejects_mismatched_image_annotation_ids(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path, with_annotation=False)
    _write_annotation(annotation_dir / "other.xml")

    with pytest.raises(ValueError, match="IDs do not match"):
        evaluate_dataset(
            image_dir,
            annotation_dir,
            {},
            pipeline_runner=lambda image, config: PipelineResult(
                mask=np.zeros(image.shape, dtype=bool),
            ),
        )


def test_evaluate_dataset_rejects_unknown_requested_id(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path)

    with pytest.raises(ValueError, match="requested image IDs"):
        evaluate_dataset(
            image_dir,
            annotation_dir,
            {},
            ids=["missing"],
            pipeline_runner=lambda image, config: PipelineResult(
                mask=np.zeros(image.shape, dtype=bool),
            ),
        )


def test_evaluate_dataset_rejects_runner_with_wrong_mask_shape(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path)

    with pytest.raises(ValueError, match="mask"):
        evaluate_dataset(
            image_dir,
            annotation_dir,
            {},
            pipeline_runner=lambda image, config: PipelineResult(
                mask=np.zeros((1, 1), dtype=bool),
            ),
        )


def test_write_evaluation_creates_per_image_csv_and_summary_json(tmp_path):
    image_dir, annotation_dir = _make_dataset(tmp_path)
    summary = evaluate_dataset(
        image_dir,
        annotation_dir,
        {},
        pipeline_runner=lambda image, config: PipelineResult(
            mask=np.zeros(image.shape, dtype=bool),
            boxes=[(1, 1, 2, 2)],
        ),
    )

    csv_path, summary_path = write_evaluation(tmp_path / "runs" / "sample", summary)

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["image_id"] == "cell"
    assert saved_summary["image_count"] == 1
    assert saved_summary["lowest_iou_cases"][0]["image_id"] == "cell"


def test_read_ids_file_ignores_comments_and_blank_lines(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("cell-a\n\n# ignored\ncell-b # retained\n", encoding="utf-8")

    assert read_ids_file(ids_file) == ["cell-a", "cell-b"]


def test_read_ids_file_rejects_duplicates_and_empty_files(tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("cell-a\ncell-a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_ids_file(ids_file)

    ids_file.write_text("# no IDs\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_ids_file(ids_file)


def _write_two_class_dataset(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    cv2.imwrite(str(images / "cell.png"), np.full((20, 20), 180, dtype=np.uint8))
    (annotations / "cell.xml").write_text(
        "<annotation>"
        "<object><name>crack</name><bndbox><xmin>3</xmin><ymin>3</ymin>"
        "<xmax>6</xmax><ymax>6</ymax></bndbox></object>"
        "<object><name>scratch</name><bndbox><xmin>11</xmin><ymin>11</ymin>"
        "<xmax>15</xmax><ymax>15</ymax></bndbox></object>"
        "</annotation>",
        encoding="utf-8",
    )
    return images, annotations


def _predict_crack_only(image, config):
    mask = np.zeros(image.shape, dtype=bool)
    mask[2:6, 2:6] = True
    return PipelineResult(mask=mask, boxes=[(2, 2, 5, 5)])


def test_evaluate_ignores_out_of_scope_classes(tmp_path):
    images, annotations = _write_two_class_dataset(tmp_path)

    summary = evaluate_dataset(images, annotations, {}, pipeline_runner=_predict_crack_only)

    record = summary.records[0]
    assert record.ground_truth_count == 1          # the scratch box is not scored
    assert json.loads(record.ground_truth_classes) == ["crack"]
    assert summary.mean_iou == pytest.approx(1.0)


def test_evaluate_class_list_can_be_set_in_config(tmp_path):
    images, annotations = _write_two_class_dataset(tmp_path)
    config = {"evaluation": {"classes": ["crack", "scratch"]}}

    summary = evaluate_dataset(
        images, annotations, config, pipeline_runner=_predict_crack_only
    )

    assert summary.records[0].ground_truth_count == 2
