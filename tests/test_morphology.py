"""Tests for morphology operations on binary masks."""

import cv2
import numpy as np
import pytest
from scipy import ndimage

from src.morphology import (
    apply_operations,
    closing,
    dilation,
    erosion,
    fill_holes,
    opening,
)


def test_erosion_hand_worked_example():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    expected = np.zeros_like(mask)
    expected[2, 2] = True

    np.testing.assert_array_equal(erosion(mask), expected)


def test_dilation_hand_worked_disk_example():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    expected = np.zeros_like(mask)
    expected[1:4, 2] = True
    expected[2, 1:4] = True

    np.testing.assert_array_equal(dilation(mask, shape="disk", size=3), expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1)])
def test_reflect_morphology_supports_single_pixel_dimensions(shape):
    mask = np.zeros(shape, dtype=bool)
    mask.flat[0] = True

    assert erosion(mask).shape == shape
    assert dilation(mask).shape == shape


@pytest.mark.parametrize("shape,size", [("square", 3), ("disk", 3), ("disk", 5)])
@pytest.mark.parametrize(
    "operation,cv_operation",
    [(erosion, cv2.erode), (dilation, cv2.dilate)],
)
def test_numpy_morphology_matches_opencv_reflect(
    operation, cv_operation, shape, size
):
    rng = np.random.default_rng(42)
    mask = rng.random((13, 17)) > 0.6
    radius = size // 2
    rows, columns = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    if shape == "square":
        kernel = np.ones((size, size), dtype=np.uint8)
    else:
        kernel = (rows * rows + columns * columns <= radius * radius).astype(
            np.uint8
        )

    expected = cv_operation(
        mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )

    np.testing.assert_array_equal(
        operation(mask, shape=shape, size=size),
        expected.astype(bool),
    )


def test_opening_and_closing_compose_the_expected_operations():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[0, 0] = True

    np.testing.assert_array_equal(
        opening(mask, shape="square", size=3),
        dilation(erosion(mask, shape="square", size=3), shape="square", size=3),
    )
    np.testing.assert_array_equal(
        closing(mask, shape="disk", size=3),
        erosion(dilation(mask, shape="disk", size=3), shape="disk", size=3),
    )


def test_fill_holes_matches_scipy_and_preserves_exterior_background():
    mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, False, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    expected = ndimage.binary_fill_holes(mask)

    np.testing.assert_array_equal(fill_holes(mask), expected)
    assert fill_holes(mask)[2, 2]
    assert not fill_holes(mask)[0, 0]


def test_fill_holes_does_not_fill_background_connected_to_border():
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 0:3] = False
    mask[2, 2] = False

    np.testing.assert_array_equal(fill_holes(mask), mask)


def test_apply_operations_uses_configured_order_and_options():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    operations = [
        {"op": "dilation", "shape": "disk", "size": 3},
        {"op": "fill_holes"},
    ]

    np.testing.assert_array_equal(
        apply_operations(mask, operations),
        fill_holes(dilation(mask, shape="disk", size=3)),
    )


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=bool),
        np.zeros((0, 2), dtype=bool),
    ],
)
def test_morphology_rejects_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        erosion(mask)


@pytest.mark.parametrize("size", [0, 2, -3, 3.0, True])
def test_morphology_rejects_invalid_structuring_element_size(size):
    with pytest.raises(ValueError, match="size"):
        dilation(np.zeros((3, 3), dtype=bool), size=size)


def test_morphology_rejects_unknown_shapes_and_operations():
    mask = np.zeros((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        erosion(mask, shape="triangle")
    with pytest.raises(ValueError, match="unsupported morphology operation"):
        apply_operations(mask, [{"op": "skeletonize"}])
    with pytest.raises(ValueError, match="does not accept"):
        apply_operations(mask, [{"op": "fill_holes", "size": 3}])
