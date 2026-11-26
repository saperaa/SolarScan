"""Image loading, annotation parsing, data splitting, and controlled noise."""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from src.pipeline import Box


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as grayscale uint8 with values in the range [0, 255].

    Args:
        path: Path to an encoded image file.

    Returns:
        A grayscale uint8 array with shape (H, W) and values in [0, 255].

    Raises:
        FileNotFoundError: If the file cannot be read as an image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)
    return image


def load_annotations(xml_path: str | Path) -> list[tuple[str, Box]]:
    """Parse Pascal VOC objects into class names and zero-based inclusive boxes.

    Args:
        xml_path: Path to a Pascal VOC XML annotation.

    Returns:
        A list of (class_name, box) pairs. Boxes are (x_min, y_min, x_max, y_max),
        zero-based and inclusive. Images without objects return an empty list.

    Raises:
        ValueError: If an object's name or bounding box is malformed.
    """
    root = ET.parse(xml_path).getroot()
    annotations: list[tuple[str, Box]] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")
        if (
            name_node is None
            or box_node is None
            or not name_node.text
            or not name_node.text.strip()
        ):
            raise ValueError(f"Malformed object annotation in {xml_path}")
        try:
            xmin = int(box_node.findtext("xmin", ""))
            ymin = int(box_node.findtext("ymin", ""))
            xmax = int(box_node.findtext("xmax", ""))
            ymax = int(box_node.findtext("ymax", ""))
        except ValueError as exc:
            raise ValueError(f"Malformed box coordinates in {xml_path}") from exc
        if xmax < xmin or ymax < ymin:
            raise ValueError(f"Invalid Pascal VOC box coordinates in {xml_path}")
        # VOC is 1-based, but converted datasets sometimes write 0; clamp to the image.
        box = (max(xmin - 1, 0), max(ymin - 1, 0), max(xmax - 1, 0), max(ymax - 1, 0))
        annotations.append((name_node.text.strip(), box))
    return annotations


def make_split(
    ids: list[str] | tuple[str, ...],
    dev_fraction: float = 0.7,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Shuffle unique image IDs deterministically into development and test sets.

    Args:
        ids: Unique image identifiers, typically strings.
        dev_fraction: Fraction assigned to development, strictly between 0 and 1.
        seed: Random seed used for the shuffle.

    Returns:
        Two shuffled lists, (dev_ids, test_ids), containing every input ID once.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")

    shuffled = np.asarray(ids, dtype=object)[
        np.random.default_rng(seed).permutation(len(ids))
    ].tolist()
    dev_count = int(round(len(shuffled) * dev_fraction))
    if len(shuffled) >= 2:
        dev_count = min(max(dev_count, 1), len(shuffled) - 1)
    return shuffled[:dev_count], shuffled[dev_count:]


def _validate_noise_input(image: np.ndarray, level: float) -> None:
    """Validate a grayscale uint8 image and a finite nonnegative noise level.

    Args:
        image: Expected uint8 array with shape (H, W) and range [0, 255].
        level: Finite nonnegative noise strength.

    Returns:
        None.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 and values in [0, 255]")
    if not np.isfinite(level) or level < 0:
        raise ValueError("level must be finite and nonnegative")


def add_noise(
    image: np.ndarray,
    kind: str,
    level: float,
    seed: int,
) -> np.ndarray:
    """Add seeded Gaussian, salt-pepper, speckle, or Poisson noise.

    Args:
        image: Grayscale uint8 array with shape (H, W) and values in [0, 255].
        kind: One of ``gaussian``, ``salt_pepper``, ``speckle``, or ``poisson``.
        level: Gaussian standard deviation in intensity units; total salt/pepper
            probability; speckle standard deviation relative to intensity; or
            positive peak photon count for Poisson noise.
        seed: Random seed, fixed for reproducible output.

    Returns:
        A grayscale uint8 array with shape (H, W), clipped to [0, 255].
    """
    _validate_noise_input(image, level)
    supported = {"gaussian", "salt_pepper", "speckle", "poisson"}
    if kind not in supported:
        raise ValueError(f"kind must be one of {sorted(supported)}")
    if kind == "salt_pepper" and level > 1:
        raise ValueError("salt_pepper level must be a probability in [0, 1]")
    if kind == "poisson" and level == 0:
        raise ValueError("poisson level must be greater than zero")

    rng = np.random.default_rng(seed)
    source = image.astype(np.float64)
    if kind == "gaussian":
        result = source + rng.normal(0.0, level, image.shape)
    elif kind == "salt_pepper":
        result = source.copy()
        draws = rng.random(image.shape)
        result[draws < level / 2.0] = 0.0
        result[(draws >= level / 2.0) & (draws < level)] = 255.0
    elif kind == "speckle":
        result = source + source * rng.normal(0.0, level, image.shape)
    else:
        normalized = source / 255.0
        result = rng.poisson(normalized * level) * (255.0 / level)
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a grayscale uint8 image using nearest-neighbour sampling.

    Args:
        image: Grayscale uint8 array with shape (H, W) and values in [0, 255].
        factor: Positive integer reduction factor.

    Returns:
        A grayscale uint8 array with shape
        (max(1, H // factor), max(1, W // factor)).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 and values in [0, 255]")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise ValueError("factor must be a positive integer")
    if factor < 1:
        raise ValueError("factor must be a positive integer")

    height, width = image.shape
    out_height = max(1, height // factor)
    out_width = max(1, width // factor)
    rows = (np.arange(out_height) * height // out_height).astype(int)
    cols = (np.arange(out_width) * width // out_width).astype(int)
    return image[np.ix_(rows, cols)]
