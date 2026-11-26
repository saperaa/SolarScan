"""Frequency-domain tools: centred DFT, transfer functions, notch filter.

Spectra are centred (fftshift), so the zero frequency sits at
(M // 2, N // 2). Transfer functions H(u, v) are float64 in [0, 1].
High-pass versions are 1 - H.
"""
import numpy as np


def dft_centered(img):
    """2-D DFT with the zero frequency in the centre.

    Input: (H, W) array, any numeric dtype. Output: complex128 (H, W).
    """
    return np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))


def inverse_dft(spectrum):
    """Inverse of dft_centered. Input: complex (H, W). Output: float64 (H, W)."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def log_spectrum(img):
    """Display spectrum log(1 + |F|). Input: (H, W). Output: float64 (H, W)."""
    return np.log1p(np.abs(dft_centered(img)))


def _distance(shape):
    """Distance D(u, v) of every frequency sample from the spectrum centre."""
    m, n = shape
    u = np.arange(m) - m // 2
    v = np.arange(n) - n // 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.hypot(uu, vv)


def ideal_lowpass(shape, d0):
    """Ideal low-pass: 1 where D <= d0, else 0. Output float64 (H, W)."""
    return (_distance(shape) <= d0).astype(np.float64)


def butterworth_lowpass(shape, d0, n=2):
    """Butterworth low-pass H = 1 / (1 + (D / d0) ** (2n)).

    Input: shape (H, W), cutoff d0 > 0 (pixels), order n.
    Output: float64 (H, W) in [0, 1]. H = 1 at D = 0, 0.5 at D = d0.
    """
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    return 1.0 / (1.0 + (_distance(shape) / d0) ** (2 * n))


def gaussian_lowpass(shape, d0):
    """Gaussian low-pass H = exp(-D^2 / (2 d0^2)). Output float64 in [0, 1]."""
    if d0 <= 0:
        raise ValueError("d0 must be positive")
    return np.exp(-(_distance(shape) ** 2) / (2.0 * d0 ** 2))


def notch_filter(shape, centers, d0):
    """Notch-reject filter: zero in a disc of radius d0 around each peak.

    centers: list of (row, col) positions in the centred spectrum. The
    symmetric partner (2*(M//2) - row, 2*(N//2) - col) is rejected too.
    Output: float64 (H, W) with values 0 or 1.
    """
    m, n = shape
    rows, cols = np.indices(shape)
    h = np.ones(shape, dtype=np.float64)
    for r, c in centers:
        for rr, cc in ((r, c), (2 * (m // 2) - r, 2 * (n // 2) - c)):
            h[np.hypot(rows - rr, cols - cc) <= d0] = 0.0
    return h


def apply_frequency_filter(img, h):
    """Filter img with transfer function h.

    Input: uint8 (H, W) 0-255 and h float (H, W). Multiplies the centred
    DFT by h, inverts, rounds and clips. Output: uint8 (H, W) 0-255.
    """
    if h.shape != img.shape:
        raise ValueError("h must have the same shape as img")
    out = inverse_dft(dft_centered(img) * h)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def find_spectrum_peaks(img, n_peaks=4, exclude=10, suppress=5):
    """Find the strongest peaks of the centred log spectrum.

    Input: uint8 (H, W). The disc of radius `exclude` around the centre
    (DC and low frequencies) is ignored. Only one half-plane is searched,
    because notch_filter adds the symmetric partner of every centre.
    Each chosen peak blanks a disc of radius `suppress` before the next.
    Output: list of (row, col) in the centred spectrum, strongest first.
    """
    s = log_spectrum(img)
    m, n = s.shape
    cy, cx = m // 2, n // 2
    rows, cols = np.indices(s.shape)
    s[np.hypot(rows - cy, cols - cx) <= exclude] = -np.inf
    s[(rows > cy) | ((rows == cy) & (cols >= cx))] = -np.inf
    peaks = []
    for _ in range(n_peaks):
        r, c = np.unravel_index(np.argmax(s), s.shape)
        if not np.isfinite(s[r, c]):
            break
        peaks.append((int(r), int(c)))
        s[np.hypot(rows - r, cols - c) <= suppress] = -np.inf
    return peaks


def busbar_notch(img, n_peaks=4, d0=3, exclude=10):
    """Remove periodic busbar / grid-finger peaks with a notch filter.

    Input: uint8 (H, W) 0-255. Finds the strongest spectrum peaks outside
    the low-frequency disc, rejects discs of radius d0 around them (and
    their symmetric partners), and inverts.
    Output: uint8 (H, W) 0-255.
    """
    peaks = find_spectrum_peaks(img, n_peaks, exclude)
    return apply_frequency_filter(img, notch_filter(img.shape, peaks, d0))
