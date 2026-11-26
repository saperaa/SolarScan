import numpy as np
import pytest

from src import frequency as f


def _butterworth_loop(shape, d0, n):
    """Slow reference: one sample at a time."""
    m, k = shape
    h = np.zeros(shape)
    for u in range(m):
        for v in range(k):
            d = np.hypot(u - m // 2, v - k // 2)
            h[u, v] = 1.0 / (1.0 + (d / d0) ** (2 * n))
    return h


def test_dft_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (33, 40), dtype=np.uint8)
    back = f.inverse_dft(f.dft_centered(img))
    assert np.abs(back - img).max() < 1e-6


def test_butterworth_three_hand_points():
    shape, d0, n = (65, 65), 10, 2
    h = f.butterworth_lowpass(shape, d0, n)
    assert h[32, 32] == pytest.approx(1.0)                      # D = 0
    assert h[32, 32 + d0] == pytest.approx(0.5)                 # D = d0
    assert h[32, 32 + 2 * d0] == pytest.approx(1 / (1 + 2 ** (2 * n)))  # 1/17


def test_butterworth_matches_loop_reference():
    h = f.butterworth_lowpass((48, 64), 12, 3)
    assert np.allclose(h, _butterworth_loop((48, 64), 12, 3))


def test_butterworth_range_and_dtype():
    h = f.butterworth_lowpass((40, 40), 8, 2)
    assert h.dtype == np.float64 and h.min() >= 0 and h.max() <= 1


def test_gaussian_at_d0():
    h = f.gaussian_lowpass((65, 65), 10)
    assert h[32, 32] == pytest.approx(1.0)
    assert h[32, 42] == pytest.approx(np.exp(-0.5))


def test_ideal_is_binary():
    h = f.ideal_lowpass((65, 65), 10)
    assert set(np.unique(h)) == {0.0, 1.0}
    assert h[32, 32] == 1.0 and h[0, 0] == 0.0


def test_notch_rejects_peak_and_partner():
    h = f.notch_filter((65, 65), [(32, 52)], 3)
    assert h[32, 52] == 0.0 and h[32, 12] == 0.0
    assert h[0, 0] == 1.0
    assert np.array_equal(h, h[::-1, ::-1])      # symmetric


def test_identity_filter_returns_image():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    out = f.apply_frequency_filter(img, np.ones((32, 32)))
    assert out.dtype == np.uint8 and np.array_equal(out, img)


def test_lowpass_keeps_constant_image():
    img = np.full((32, 32), 100, dtype=np.uint8)
    h = f.butterworth_lowpass((32, 32), 5, 2)
    assert np.array_equal(f.apply_frequency_filter(img, h), img)


def test_log_spectrum_nonnegative():
    img = np.random.default_rng(2).integers(0, 256, (16, 16), dtype=np.uint8)
    assert (f.log_spectrum(img) >= 0).all()


def _stripes(size=64, k=8):
    x = np.arange(size)
    row = 128 + 50 * np.sin(2 * np.pi * k * x / size)
    return np.tile(np.rint(row), (size, 1)).astype(np.uint8)


def test_find_peak_of_pure_stripes():
    # Vertical stripes with 8 cycles: peak at (centre row, centre col - 8)
    peaks = f.find_spectrum_peaks(_stripes(), n_peaks=1, exclude=3)
    assert peaks == [(32, 24)]


def test_busbar_notch_removes_stripes():
    img = _stripes()
    out = f.busbar_notch(img, n_peaks=1, d0=3, exclude=3)
    assert out.dtype == np.uint8
    assert out.std() < 0.2 * img.std()
