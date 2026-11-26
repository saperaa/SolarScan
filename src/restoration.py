"""Spatial-domain restoration filters.

Every public filter takes and returns a uint8 (H, W) image in 0-255.
Neighbourhood operations pad the image first. The border mode is a
parameter: "replicate" (default), "reflect" or "zero".
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_PAD_MODES = {"replicate": "edge", "reflect": "reflect", "zero": "constant"}


def _windows(img, size, border):
    """Return all size x size neighbourhoods of img.

    Input: uint8 (H, W). Output: float64 array (H, W, size * size).
    The image is padded by size // 2 using the chosen border mode.
    """
    if img.ndim != 2:
        raise ValueError("img must be 2-D (H, W)")
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if border not in _PAD_MODES:
        raise ValueError("border must be replicate, reflect or zero")
    pad = size // 2
    padded = np.pad(img.astype(np.float64), pad, mode=_PAD_MODES[border])
    win = sliding_window_view(padded, (size, size))
    h, w = img.shape
    return win.reshape(h, w, size * size)


def _to_uint8(x):
    """Round a float array and clip it to uint8 0-255."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def arithmetic_mean(img, size=3, border="replicate"):
    """Arithmetic mean filter. uint8 (H, W) 0-255 -> uint8 (H, W) 0-255."""
    return _to_uint8(_windows(img, size, border).mean(axis=-1))


def geometric_mean(img, size=3, border="replicate"):
    """Geometric mean filter (product ** (1/mn)).

    uint8 (H, W) 0-255 -> uint8 (H, W) 0-255. A window containing a zero
    gives 0, as the product is 0.
    """
    w = _windows(img, size, border)
    has_zero = (w == 0).any(axis=-1)
    out = np.exp(np.log(np.where(w == 0, 1.0, w)).mean(axis=-1))
    out[has_zero] = 0.0
    return _to_uint8(out)


def harmonic_mean(img, size=3, border="replicate"):
    """Harmonic mean filter, mn / sum(1/g). Good for salt noise.

    uint8 (H, W) 0-255 -> uint8 (H, W) 0-255. A window with a zero gives 0.
    """
    w = _windows(img, size, border)
    has_zero = (w == 0).any(axis=-1)
    out = w.shape[-1] / (1.0 / np.where(w == 0, 1.0, w)).sum(axis=-1)
    out[has_zero] = 0.0
    return _to_uint8(out)


def contraharmonic_mean(img, size=3, q=1.5, border="replicate"):
    """Contraharmonic mean, sum(g^(q+1)) / sum(g^q).

    q > 0 removes pepper noise, q < 0 removes salt noise, q = 0 is the
    arithmetic mean, q = -1 is the harmonic mean.
    uint8 (H, W) 0-255 -> uint8 (H, W) 0-255. Zeros are replaced by 1e-6
    to avoid division by zero when q < 0.
    """
    w = np.maximum(_windows(img, size, border), 1e-6)
    return _to_uint8((w ** (q + 1)).sum(axis=-1) / (w ** q).sum(axis=-1))


def median_filter(img, size=3, border="replicate"):
    """Median filter (NumPy implementation).

    Slides a size x size window over the explicitly padded image and takes
    the median. Border: replicate by default, which matches cv2.medianBlur.
    uint8 (H, W) 0-255 -> uint8 (H, W) 0-255.
    """
    return _to_uint8(np.median(_windows(img, size, border), axis=-1))


def max_filter(img, size=3, border="replicate"):
    """Maximum filter (removes pepper). uint8 (H, W) -> uint8 (H, W)."""
    return _to_uint8(_windows(img, size, border).max(axis=-1))


def min_filter(img, size=3, border="replicate"):
    """Minimum filter (removes salt). uint8 (H, W) -> uint8 (H, W)."""
    return _to_uint8(_windows(img, size, border).min(axis=-1))


def midpoint_filter(img, size=3, border="replicate"):
    """Midpoint filter, (max + min) / 2. uint8 (H, W) -> uint8 (H, W)."""
    w = _windows(img, size, border)
    return _to_uint8((w.max(axis=-1) + w.min(axis=-1)) / 2.0)


def alpha_trimmed_mean(img, size=3, d=2, border="replicate"):
    """Alpha-trimmed mean: drop the d/2 lowest and d/2 highest, average rest.

    d must be even with 0 <= d < size * size. d = 0 is the arithmetic mean,
    d = size*size - 1 is the median. uint8 (H, W) -> uint8 (H, W).
    """
    w = _windows(img, size, border)
    n = w.shape[-1]
    if d % 2 != 0 or d < 0 or d >= n:
        raise ValueError("d must be even and satisfy 0 <= d < size * size")
    s = np.sort(w, axis=-1)
    return _to_uint8(s[..., d // 2:n - d // 2].mean(axis=-1))
