"""Binary morphology for defect-candidate masks."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import cv2
import numpy as np

StructuringShape = Literal["square", "disk"]


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a nonempty two-dimensional boolean mask."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional NumPy array")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if mask.dtype != np.bool_:
        raise ValueError("mask must have boolean dtype")
    return mask


def _make_structuring_element(shape: str, size: int) -> np.ndarray:
    """Create a centered square or disk footprint with an odd width."""
    if not isinstance(shape, str) or shape not in {"square", "disk"}:
        raise ValueError("shape must be 'square' or 'disk'")
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size < 1
        or size % 2 == 0
    ):
        raise ValueError("size must be a positive odd integer")

    radius = int(size) // 2
    if shape == "square":
        return np.ones((size, size), dtype=bool)

    rows, columns = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return rows * rows + columns * columns <= radius * radius


def _morph(mask: np.ndarray, shape: str, size: int, *, erode: bool) -> np.ndarray:
    """Apply a flat binary morphology operation using reflect padding."""
    mask = _validate_mask(mask)
    footprint = _make_structuring_element(shape, size)
    radius = footprint.shape[0] // 2
    padded = np.pad(mask, ((radius, radius), (radius, radius)), mode="reflect")

    result = np.ones_like(mask) if erode else np.zeros_like(mask)
    height, width = mask.shape
    for row, column in zip(*np.nonzero(footprint)):
        neighborhood = padded[row:row + height, column:column + width]
        if erode:
            np.logical_and(result, neighborhood, out=result)
        else:
            np.logical_or(result, neighborhood, out=result)
    return result


def erosion(
    mask: np.ndarray,
    shape: StructuringShape = "square",
    size: int = 3,
) -> np.ndarray:
    """Erode a boolean mask; pixels beyond the image reflect at its boundary.

    Reflect padding matches OpenCV's ``BORDER_REFLECT_101``. The input must be a
    nonempty two-dimensional boolean array and ``size`` must be a positive odd
    integer.
    """
    return _morph(mask, shape, size, erode=True)


def dilation(
    mask: np.ndarray,
    shape: StructuringShape = "square",
    size: int = 3,
) -> np.ndarray:
    """Dilate a boolean mask; pixels beyond the image reflect at its boundary.

    Reflect padding matches OpenCV's ``BORDER_REFLECT_101``. The input must be a
    nonempty two-dimensional boolean array and ``size`` must be a positive odd
    integer.
    """
    return _morph(mask, shape, size, erode=False)


def opening(
    mask: np.ndarray,
    shape: StructuringShape = "square",
    size: int = 3,
) -> np.ndarray:
    """Apply erosion followed by dilation using reflect boundary handling."""
    return dilation(erosion(mask, shape, size), shape, size)


def closing(
    mask: np.ndarray,
    shape: StructuringShape = "square",
    size: int = 3,
) -> np.ndarray:
    """Apply dilation followed by erosion using reflect boundary handling."""
    return erosion(dilation(mask, shape, size), shape, size)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill 4-connected background regions that do not touch the image border."""
    mask = _validate_mask(mask)
    background = np.pad(~mask, 1, mode="constant", constant_values=True).astype(
        np.uint8
    )
    cv2.floodFill(background, None, (0, 0), 2, flags=4)
    holes = background[1:-1, 1:-1] == 1
    return mask | holes


def apply_operations(
    mask: np.ndarray,
    operations: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Apply configured morphology operations in order to a binary mask.

    Supported operation names are ``erosion``, ``dilation``, ``opening``,
    ``closing``, and ``fill_holes``. The first four accept ``shape`` and ``size``
    options; those options default to ``square`` and 3. ``fill_holes`` takes no
    options.
    """
    result = _validate_mask(mask).copy()
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("operations must be a sequence of mappings")

    operation_functions = {
        "erosion": erosion,
        "dilation": dilation,
        "opening": opening,
        "closing": closing,
    }
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("each morphology operation must be a mapping")
        name = operation.get("op")
        if name == "fill_holes":
            if set(operation) != {"op"}:
                raise ValueError("fill_holes does not accept additional options")
            result = fill_holes(result)
            continue
        if not isinstance(name, str) or name not in operation_functions:
            raise ValueError(f"unsupported morphology operation: {name!r}")
        if set(operation) - {"op", "shape", "size"}:
            raise ValueError(f"unsupported options for morphology operation: {name}")
        result = operation_functions[name](
            result,
            shape=operation.get("shape", "square"),
            size=operation.get("size", 3),
        )
    return result
