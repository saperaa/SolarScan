import cv2
import numpy as np
import pytest

from src import restoration as r


@pytest.mark.parametrize("size", [3, 5])
def test_median_matches_opencv(size):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 40), dtype=np.uint8)
    assert np.array_equal(r.median_filter(img, size), cv2.medianBlur(img, size))


def test_median_hand_example_5x5():
    # Flat patch of 10s, one salt (255) and one pepper (0) pixel.
    # Any 3x3 window holds at most one 255 and one 0 -> median is 10.
    img = np.full((5, 5), 10, dtype=np.uint8)
    img[1, 1] = 255
    img[3, 3] = 0
    out = r.median_filter(img, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.full((5, 5), 10, dtype=np.uint8))


def test_constant_image_unchanged_by_all_filters():
    img = np.full((9, 9), 80, dtype=np.uint8)
    for f in (r.arithmetic_mean, r.geometric_mean, r.harmonic_mean,
              r.contraharmonic_mean, r.median_filter, r.max_filter,
              r.min_filter, r.midpoint_filter, r.alpha_trimmed_mean):
        out = f(img, 3)
        assert out.dtype == np.uint8
        assert np.array_equal(out, img)


def test_contraharmonic_special_cases():
    rng = np.random.default_rng(1)
    img = rng.integers(1, 256, (20, 20), dtype=np.uint8)
    am = r.arithmetic_mean(img, 3).astype(int)
    hm = r.harmonic_mean(img, 3).astype(int)
    assert np.abs(r.contraharmonic_mean(img, 3, q=0) - am).max() <= 1
    assert np.abs(r.contraharmonic_mean(img, 3, q=-1) - hm).max() <= 1


def test_alpha_trimmed_special_cases():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    assert np.array_equal(r.alpha_trimmed_mean(img, 3, d=0),
                          r.arithmetic_mean(img, 3))
    assert np.array_equal(r.alpha_trimmed_mean(img, 3, d=8),
                          r.median_filter(img, 3))


def test_max_min_midpoint_order():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    lo, hi = r.min_filter(img, 3), r.max_filter(img, 3)
    mid = r.midpoint_filter(img, 3)
    assert (lo <= mid).all() and (mid <= hi).all()


def test_even_size_rejected():
    with pytest.raises(ValueError):
        r.median_filter(np.zeros((5, 5), dtype=np.uint8), 4)
