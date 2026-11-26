"""Tests for pipeline helpers owned by the Tech Lead."""

import cv2
import numpy as np

from src.pipeline import boxes_to_mask


def test_boxes_to_mask_hand_worked_6x6():
    # Box A covers x 0-2, y 0-1 (6 px); box B covers x 2-3, y 1-3 (6 px).
    # They share pixel (x=2, y=1), so the union has 6 + 6 - 1 = 11 pixels.
    mask = boxes_to_mask([(0, 0, 2, 1), (2, 1, 3, 3)], (6, 6))
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert mask.dtype == bool
    assert mask.sum() == 11
    np.testing.assert_array_equal(mask, expected)


def test_boxes_to_mask_matches_opencv_filled_rectangles():
    rng = np.random.default_rng(0)
    boxes = []
    for _ in range(5):
        x0, y0 = rng.integers(0, 30, 2)
        boxes.append((int(x0), int(y0), int(x0 + rng.integers(0, 10)),
                      int(y0 + rng.integers(0, 10))))
    drawn = np.zeros((32, 40), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(drawn, (x0, y0), (x1, y1), 255, thickness=-1)
    np.testing.assert_array_equal(boxes_to_mask(boxes, (32, 40)), drawn == 255)


def test_boxes_to_mask_empty_and_clipped():
    assert not boxes_to_mask([], (4, 4)).any()
    clipped = boxes_to_mask([(2, 2, 10, 10)], (4, 4))
    assert clipped.sum() == 4 and clipped[2:, 2:].all()
