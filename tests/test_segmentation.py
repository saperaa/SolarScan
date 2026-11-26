"""Tests for src/segmentation.py (owner: Mohamed Alaa, role 5)."""

import cv2
import numpy as np
import pytest
from skimage.filters import threshold_otsu

from src.segmentation import adaptive_threshold, otsu_threshold, threshold_mask


def test_otsu_hand_worked_4x4():
    # 4x4 image: eight pixels at 20 and eight at 200 (two clean classes).
    # For any k in [20, 199]: w0 = w1 = 0.5, mu0 = 20, mu1 = 200,
    #   sigma_B^2 = 0.5 * 0.5 * (20 - 200)^2 = 8100 (the maximum).
    # The first k reaching the maximum is 20, so Otsu returns 20:
    # every 20 is dark (<= 20), every 200 is bright.
    img = np.array(
        [[20, 20, 200, 200],
         [20, 20, 200, 200],
         [20, 20, 200, 200],
         [20, 20, 200, 200]],
        dtype=np.uint8,
    )
    t = otsu_threshold(img)
    assert t == 20
    mask = threshold_mask(img, t)
    assert mask[:, :2].all() and not mask[:, 2:].any()


def test_otsu_hand_worked_three_levels():
    # Values 10 (x4), 50 (x8), 240 (x4); 16 pixels, mean = 87.5.
    # k in [10, 49]: w0 = 4/16, mu0 = 10; w1 = 12/16, mu1 = (400 + 960) / 12 = 113.33
    #   sigma_B^2 = 0.25 * 0.75 * 103.33^2 = 2002.1
    # k in [50, 239]: w0 = 12/16, mu0 = 36.67; w1 = 4/16, mu1 = 240
    #   sigma_B^2 = 0.75 * 0.25 * 203.33^2 = 7752.1  -> larger, so k = 50
    img = np.array([10] * 4 + [50] * 8 + [240] * 4, dtype=np.uint8).reshape(4, 4)
    assert otsu_threshold(img) == 50


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_otsu_matches_skimage_and_opencv(seed):
    rng = np.random.default_rng(seed)
    dark = rng.normal(60, 15, 400)
    bright = rng.normal(170, 20, 1200)
    img = np.clip(np.concatenate([dark, bright]), 0, 255).astype(np.uint8).reshape(40, 40)

    t = otsu_threshold(img)
    assert t == int(threshold_otsu(img))
    t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    assert t == int(t_cv)


def test_otsu_constant_image():
    assert otsu_threshold(np.full((5, 5), 90, dtype=np.uint8)) == 90


def test_threshold_mask_dark_and_bright():
    img = np.array([[10, 127, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(threshold_mask(img, 127), [[True, True, False, False]])
    np.testing.assert_array_equal(
        threshold_mask(img, 127, dark_objects=False), [[False, False, True, True]]
    )
    assert threshold_mask(img, 127).dtype == bool


def test_adaptive_hand_worked_3x3():
    # Centre pixel 10 surrounded by 100s. 3x3 mean = (8 * 100 + 10) / 9 = 90.
    # With c = 5 the local threshold is 85: 10 <= 85 -> dark (True).
    # Every other pixel is 100 and its 3x3 window (replicate border) holds at most the
    # one 10, so its mean is >= 90 and 100 > 90 - 5 -> not dark (False).
    img = np.full((3, 3), 100, dtype=np.uint8)
    img[1, 1] = 10
    mask = adaptive_threshold(img, block_size=3, c=5)
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize(("block_size", "c"), [(3, 0), (11, 5), (31, 2), (51, -3)])
def test_adaptive_matches_opencv(block_size, c):
    rng = np.random.default_rng(block_size)
    y, x = np.mgrid[0:64, 0:80]
    base = 80 + 1.5 * x + rng.normal(0, 12, (64, 80))   # uneven brightness + noise
    img = np.clip(base, 0, 255).astype(np.uint8)

    expected = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    np.testing.assert_array_equal(adaptive_threshold(img, block_size, c), expected == 255)


def test_adaptive_finds_dark_line_under_uneven_light():
    # A global threshold fails on a brightness ramp; the adaptive one still finds the line.
    y, x = np.mgrid[0:40, 0:40]
    img = (60 + 4 * x).clip(0, 255).astype(np.uint8)
    img[20, :] = (img[20, :].astype(int) - 40).clip(0, 255).astype(np.uint8)
    mask = adaptive_threshold(img, block_size=9, c=10)
    assert mask[20, 2:-2].all()
    assert not mask[5, :].any()


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float64), np.zeros((2, 2, 3), np.uint8)])
def test_rejects_bad_images(bad):
    with pytest.raises(ValueError):
        otsu_threshold(bad)


def test_adaptive_rejects_even_block():
    with pytest.raises(ValueError):
        adaptive_threshold(np.zeros((5, 5), np.uint8), 4, 0)
