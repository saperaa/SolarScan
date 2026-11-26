"""Tests for SolarScan image and detection metrics."""

import numpy as np
import pytest

import src.pipeline as pipeline
from src.metrics import (
    box_union_iou,
    count_error,
    detection_prf,
    mask_iou,
    mse,
    psnr,
)


def test_mse_and_psnr_hand_worked_example():
    reference = np.zeros((2, 2), dtype=np.uint8)
    degraded = reference.copy()
    degraded[0, 0] = 10

    assert mse(reference, degraded) == 25.0
    assert psnr(reference, degraded) == pytest.approx(34.15, abs=0.01)


def test_psnr_is_infinite_for_identical_images():
    image = np.full((2, 2), 128, dtype=np.uint8)

    assert psnr(image, image) == float("inf")


def test_mse_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        mse(np.zeros((2, 2)), np.zeros((2, 3)))


def test_mask_iou_hand_worked_example_and_empty_cases():
    pred = np.zeros((4, 4), dtype=bool)
    gt = np.zeros((4, 4), dtype=bool)
    pred[0:2, 0:2] = True
    gt[0:2, 1:3] = True

    assert mask_iou(pred, gt) == pytest.approx(2 / 6)
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0
    assert mask_iou(np.zeros((4, 4), bool), gt) == 0.0
    assert mask_iou(pred, np.zeros((4, 4), bool)) == 0.0


def test_box_union_iou_uses_shared_boxes_to_mask(monkeypatch):
    def rasterize(boxes, shape):
        mask = np.zeros(shape, dtype=bool)
        for x_min, y_min, x_max, y_max in boxes:
            mask[y_min:y_max + 1, x_min:x_max + 1] = True
        return mask

    monkeypatch.setattr(pipeline, "boxes_to_mask", rasterize, raising=False)

    assert (
        box_union_iou([(0, 0, 1, 1)], [(1, 0, 2, 1)], (4, 4))
        == pytest.approx(2 / 6)
    )


def test_detection_prf_greedily_matches_each_box_once():
    true_box = (0, 0, 1, 1)
    pred = [true_box, true_box, (3, 3, 4, 4)]

    assert detection_prf(pred, [true_box], iou_thr=0.3) == (1, 2, 0)
    assert detection_prf([], [true_box]) == (0, 0, 1)
    assert detection_prf([], []) == (0, 0, 0)


def test_detection_prf_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="iou_thr"):
        detection_prf([], [], iou_thr=1.1)


def test_count_error():
    assert count_error(7, 4) == 3
    assert count_error(0, 0) == 0


def test_count_error_rejects_negative_counts():
    with pytest.raises(ValueError, match="nonnegative"):
        count_error(-1, 2)


def test_count_error_rejects_noninteger_counts():
    with pytest.raises(ValueError, match="integers"):
        count_error(1.5, 2)
