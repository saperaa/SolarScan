"""Tests for pure-NumPy baseline operations."""

import cv2
import numpy as np
import pytest

from src.baselines import global_threshold


def test_global_threshold_matches_opencv_and_hand_worked_example():
    image = np.array(
        [
            [0, 64, 127],
            [128, 200, 255],
            [126, 127, 128],
        ],
        dtype=np.uint8,
    )

    result = global_threshold(image, threshold=127)
    _, expected_cv = cv2.threshold(
        image, 127, 255, cv2.THRESH_BINARY_INV
    )
    expected_hand = np.array(
        [
            [255, 255, 255],
            [0, 0, 0],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(result, expected_cv)
    np.testing.assert_array_equal(result, expected_hand)


@pytest.mark.parametrize("threshold", [-1, 256, 1.5, True])
def test_global_threshold_rejects_invalid_threshold(threshold):
    image = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="threshold"):
        global_threshold(image, threshold)
