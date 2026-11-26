"""Run a configured SolarScan pipeline on one image."""

import argparse
import json
from pathlib import Path
import re
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from src.config import load_pipeline_config
from src.data import load_image
from src.pipeline import PipelineResult, run


def process_image(
    image: np.ndarray,
    config: Mapping[str, Any],
) -> PipelineResult:
    """Run the shared pipeline and verify its mask matches the input image."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty two-dimensional NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image must have uint8 dtype")
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    result = run(image, dict(config))
    if not isinstance(result, PipelineResult):
        raise TypeError("pipeline runner must return PipelineResult")
    if (
        not isinstance(result.mask, np.ndarray)
        or result.mask.dtype != np.bool_
        or result.mask.shape != image.shape
    ):
        raise ValueError("pipeline mask must be boolean and match the input image shape")
    return result


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert a numeric grayscale stage image into displayable uint8 values."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("stage images must be nonempty two-dimensional arrays")
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError("stage images must have numeric or boolean dtypes")
    values = image.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("stage images must contain only finite values")
    if values.min() >= 0.0 and values.max() <= 1.0:
        values = values * 255.0
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def display_image(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image or stage into RGB for display."""
    return cv2.cvtColor(to_uint8_image(image), cv2.COLOR_GRAY2RGB)


def overlay_boxes(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Return an RGB view of an image with inclusive predicted boxes."""
    overlay = display_image(image)
    height, width = overlay.shape[:2]
    for x_min, y_min, x_max, y_max in boxes:
        x_min = max(0, min(int(x_min), width - 1))
        y_min = max(0, min(int(y_min), height - 1))
        x_max = max(0, min(int(x_max), width - 1))
        y_max = max(0, min(int(y_max), height - 1))
        if x_min <= x_max and y_min <= y_max:
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return overlay


def write_result(
    output_dir: str | Path,
    image_path: str | Path,
    image: np.ndarray,
    result: PipelineResult,
) -> Path:
    """Save stage images, the final mask, and a box overlay for one input."""
    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    if (
        not isinstance(result.mask, np.ndarray)
        or result.mask.dtype != np.bool_
        or result.mask.shape != image.shape
    ):
        raise ValueError("result mask must be boolean and match the input image shape")
    destination = Path(output_dir) / Path(image_path).stem
    destination.mkdir(parents=True, exist_ok=True)

    stages = {"input": image, **result.stages, "final_mask": result.mask}
    for stage_name, stage_image in stages.items():
        if not isinstance(stage_name, str):
            raise ValueError("stage names must be strings")
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", stage_name).strip("._")
        if not filename:
            raise ValueError("stage name cannot be converted to a safe filename")
        output_path = destination / f"{filename}.png"
        if not cv2.imwrite(str(output_path), to_uint8_image(stage_image)):
            raise OSError(f"unable to write stage image: {output_path}")

    overlay_path = destination / "boxes.png"
    overlay = cv2.cvtColor(
        overlay_boxes(image, result.boxes),
        cv2.COLOR_RGB2BGR,
    )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"unable to write box overlay: {overlay_path}")
    return destination


def _result_summary(result: PipelineResult) -> dict[str, Any]:
    return {
        "decision": result.decision,
        "count": result.count,
        "area_pct": result.area_pct,
        "boxes": result.boxes,
        "timings_s": result.timings_s,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured pipeline from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to a grayscale EL image")
    parser.add_argument("--config", required=True, help="pipeline YAML configuration")
    parser.add_argument(
        "--output-dir",
        help="optional directory for stage images and the final box overlay",
    )
    args = parser.parse_args(argv)

    image = load_image(args.image)
    config = load_pipeline_config(args.config)
    result = process_image(image, config)
    if args.output_dir:
        output_path = write_result(args.output_dir, args.image, image, result)
        print(f"Saved outputs: {output_path}")
    print(json.dumps(_result_summary(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
