"""Image and detection metrics used by SolarScan."""

import numpy as np

from src.pipeline import Box


def _validate_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Require two nonempty arrays with identical shapes.

    Args:
        a: First numeric or boolean NumPy array.
        b: Second numeric or boolean NumPy array with the same shape.

    Returns:
        None.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("inputs must be NumPy arrays")
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    if a.size == 0:
        raise ValueError("inputs must not be empty")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean squared error for same-shaped numeric arrays.

    Args:
        a: Numeric NumPy array; typically uint8 image values in [0, 255].
        b: Numeric NumPy array with the same shape and range as ``a``.

    Returns:
        Mean squared error as a Python float, calculated in float64.
    """
    _validate_same_shape(a, b)
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(
        b.dtype, np.number
    ):
        raise ValueError("inputs must have numeric dtypes")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff, dtype=np.float64))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Return peak signal-to-noise ratio for images in the uint8 range.

    Args:
        a: Numeric NumPy array; image intensities are expected in [0, 255].
        b: Numeric NumPy array with the same shape and range as ``a``.

    Returns:
        PSNR in decibels as a Python float, or positive infinity if identical.
    """
    error = mse(a, b)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / error))


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return intersection-over-union for same-shaped binary masks.

    Args:
        pred: Predicted mask, any array convertible to bool, shape (H, W).
        gt: Ground-truth mask, any array convertible to bool, shape (H, W).

    Returns:
        IoU in [0, 1]; returns 1.0 when both masks are empty.
    """
    _validate_same_shape(pred, gt)
    pred_mask = pred.astype(bool)
    gt_mask = gt.astype(bool)
    union = np.count_nonzero(pred_mask | gt_mask)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(pred_mask & gt_mask)
    return float(intersection / union)


def box_union_iou(
    pred: list[Box],
    gt: list[Box],
    shape: tuple[int, int],
) -> float:
    """Return mask IoU for unions of predicted and ground-truth boxes.

    Args:
        pred: Predicted inclusive boxes as (x_min, y_min, x_max, y_max).
        gt: Ground-truth inclusive boxes in the same format.
        shape: Output mask shape (H, W), with positive dimensions.

    Returns:
        IoU of the rasterized box unions, including the documented empty cases.
    """
    from src.pipeline import boxes_to_mask

    pred_mask = boxes_to_mask(pred, shape)
    gt_mask = boxes_to_mask(gt, shape)
    return mask_iou(pred_mask, gt_mask)


def _box_iou(a: Box, b: Box) -> float:
    """Compute IoU for two zero-based inclusive boxes.

    Args:
        a: First box (x_min, y_min, x_max, y_max).
        b: Second box in the same format.

    Returns:
        Scalar box IoU in [0, 1].
    """
    for box in (a, b):
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("box coordinates must be ordered min to max")
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    intersection = (
        max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
    )
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - intersection
    return float(intersection / union)


def detection_prf(
    pred: list[Box],
    gt: list[Box],
    iou_thr: float = 0.3,
) -> tuple[int, int, int]:
    """Greedily match boxes one-to-one by descending IoU.

    Args:
        pred: Predicted zero-based inclusive boxes.
        gt: Ground-truth zero-based inclusive boxes.
        iou_thr: Minimum box IoU in [0, 1] for a true positive.

    Returns:
        Counts (TP, FP, FN) after greedy one-to-one matching.
    """
    if not 0.0 <= iou_thr <= 1.0:
        raise ValueError("iou_thr must be in [0, 1]")
    pairs = [
        (_box_iou(pred_box, gt_box), pred_index, gt_index)
        for pred_index, pred_box in enumerate(pred)
        for gt_index, gt_box in enumerate(gt)
    ]
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for overlap, pred_index, gt_index in pairs:
        if overlap < iou_thr:
            break
        if pred_index not in matched_pred and gt_index not in matched_gt:
            matched_pred.add(pred_index)
            matched_gt.add(gt_index)
    true_positives = len(matched_pred)
    return true_positives, len(pred) - true_positives, len(gt) - true_positives


def count_error(n_pred: int, n_gt: int) -> int:
    """Return absolute difference between predicted and true region counts.

    Args:
        n_pred: Nonnegative predicted region count.
        n_gt: Nonnegative ground-truth region count.

    Returns:
        Absolute count error as an integer.
    """
    counts = (n_pred, n_gt)
    if any(
        isinstance(count, bool) or not isinstance(count, (int, np.integer))
        for count in counts
    ):
        raise ValueError("region counts must be integers")
    if n_pred < 0 or n_gt < 0:
        raise ValueError("region counts must be nonnegative")
    return abs(n_pred - n_gt)
