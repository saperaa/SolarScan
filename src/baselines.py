"""Pure-NumPy helpers for the project's simple baselines."""

import numpy as np


def global_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Apply an inverse binary threshold to a grayscale uint8 image.

    Pixels less than or equal to ``threshold`` become 255; brighter pixels become
    0, matching ``cv2.THRESH_BINARY_INV``.

    Args:
        image: Grayscale uint8 NumPy array with shape (H, W), values in [0, 255].
        threshold: Integer threshold in [0, 255].

    Returns:
        A uint8 binary array with shape (H, W), values in {0, 255}.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 and values in [0, 255]")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, np.integer)):
        raise ValueError("threshold must be an integer in [0, 255]")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be an integer in [0, 255]")
    return np.where(image <= threshold, 255, 0).astype(np.uint8)
