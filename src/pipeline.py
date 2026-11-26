"""Baselines and Pipelines A, B, C.

Owner: Tech Lead (role 1). Each pipeline is driven by one YAML file in ``configs/``.

Data flow between stages (to be completed with the block diagrams in Week 12):

    raw image  --load_image-->  uint8 (H, W) 0-255
      --> enhancement / restoration --> uint8 or float64 (H, W)
      --> segmentation              --> bool (H, W), True = defect candidate
      --> morphology                --> bool (H, W)
      --> measurement               --> list of boxes, count, area %
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src import morphology, segmentation
from src.baselines import global_threshold

Box = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), inclusive pixel coords


@dataclass
class PipelineResult:
    """Output of every pipeline: the final mask, boxes and measurements."""

    mask: np.ndarray
    boxes: list[Box] = field(default_factory=list)
    area_pct: float = 0.0
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of detected defect regions."""
        return len(self.boxes)

    @property
    def decision(self) -> str:
        """``"defective"`` if any region was found, else ``"ok"``."""
        return "defective" if self.boxes else "ok"


def boxes_to_mask(boxes: list[Box], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise boxes into one binary mask (pure NumPy).

    Args:
        boxes: Inclusive boxes (x_min, y_min, x_max, y_max) in pixel coordinates.
            Parts outside the image are clipped.
        shape: Output mask shape (H, W).

    Returns:
        ``bool`` (H, W) mask, True inside any box. All False for an empty list.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for x_min, y_min, x_max, y_max in boxes:
        x0, y0 = max(int(x_min), 0), max(int(y_min), 0)
        x1, y1 = min(int(x_max), width - 1), min(int(y_max), height - 1)
        if x0 <= x1 and y0 <= y1:
            mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) column pairs, inclusive, of the True runs in one mask row."""
    padded = np.concatenate(([False], row, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist()))


def connected_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Label 8-connected foreground regions of a binary mask (pure NumPy + Python).

    Works on runs of True pixels rather than single pixels, which keeps the Python loop
    short on large images. Pass 1 finds the runs of every row and joins (union-find) each
    run with the runs of the row above that touch it, including diagonally: two runs
    touch when ``start <= prev_end + 1`` and ``end >= prev_start - 1``. Pass 2 gives every
    union-find root a final label 1..n and paints the runs.

    Args:
        mask: ``bool`` (H, W) mask, True = foreground.

    Returns:
        ``(n, labels)``: n regions, and an ``int32`` (H, W) label image with 0 for
        background and 1..n for the regions, numbered in raster order of their first
        pixel. Same partition as ``cv2.connectedComponents(mask, connectivity=8)``.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    mask = mask.astype(bool)
    height, width = mask.shape

    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs: list[tuple[int, int, int]] = []          # (row, start, end)
    prev: list[tuple[int, int, int]] = []          # (start, end, run index) of row above
    for r in range(height):
        current = []
        j = 0
        for start, end in _row_runs(mask[r]):
            idx = len(runs)
            runs.append((r, start, end))
            parent.append(idx)
            while j < len(prev) and prev[j][1] < start - 1:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] <= end + 1:
                union(idx, prev[k][2])
                k += 1
            current.append((start, end, idx))
        prev = current

    labels = np.zeros((height, width), dtype=np.int32)
    final: dict[int, int] = {}
    for idx, (r, start, end) in enumerate(runs):
        root = find(idx)
        if root not in final:
            final[root] = len(final) + 1
        labels[r, start:end + 1] = final[root]
    return len(final), labels


def measure_regions(
    mask: np.ndarray, min_area: int = 0, max_aspect: float | None = None
) -> tuple[list[Box], np.ndarray]:
    """Turn a candidate mask into defect boxes.

    Labels 8-connected regions, then drops regions smaller than ``min_area`` pixels and,
    if ``max_aspect`` is given, regions whose box is longer than ``max_aspect`` times its
    width (long thin grid lines and busbars).

    Args:
        mask: ``bool`` (H, W) candidate mask.
        min_area: Minimum region size in pixels.
        max_aspect: Maximum box aspect ratio (long side / short side), or None.

    Returns:
        ``(boxes, kept_mask)``: inclusive boxes of the kept regions in raster order, and a
        ``bool`` (H, W) mask containing only the kept regions.
    """
    n, labels = connected_components(mask)
    kept = np.zeros(mask.shape, dtype=bool)
    if n == 0:
        return [], kept
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    area = np.bincount(lab, minlength=n + 1)
    x0 = np.full(n + 1, mask.shape[1])
    y0 = np.full(n + 1, mask.shape[0])
    x1 = np.full(n + 1, -1)
    y1 = np.full(n + 1, -1)
    np.minimum.at(x0, lab, xs)
    np.minimum.at(y0, lab, ys)
    np.maximum.at(x1, lab, xs)
    np.maximum.at(y1, lab, ys)

    boxes: list[Box] = []
    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if area[i] < min_area:
            continue
        w, h = x1[i] - x0[i] + 1, y1[i] - y0[i] + 1
        if max_aspect is not None and max(w, h) / min(w, h) > max_aspect:
            continue
        keep[i] = True
        boxes.append((int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i])))
    kept = keep[labels]
    return boxes, kept


def _segment(image: np.ndarray, cfg: dict) -> np.ndarray:
    """Run the configured segmentation method and return a ``bool`` candidate mask."""
    method = cfg.get("method")
    dark = cfg.get("dark_objects", True)
    if method == "global":
        mask = global_threshold(image, int(cfg.get("threshold", 127)))
        return mask if dark else ~mask
    if method == "otsu":
        return segmentation.threshold_mask(image, segmentation.otsu_threshold(image), dark)
    if method == "adaptive":
        return segmentation.adaptive_threshold(
            image, int(cfg["block_size"]), int(cfg["c"]), dark
        )
    raise ValueError(f"unsupported segmentation method: {method!r}")


BASELINE_TRIVIAL = {
    "name": "baseline_trivial",
    "segmentation": {"method": "global", "threshold": 127, "dark_objects": True},
    "measurement": {"min_area": 0},
}

BASELINE_SIMPLE = {
    "name": "baseline_simple",
    "segmentation": {"method": "otsu", "dark_objects": True},
    "morphology": [{"op": "opening", "shape": "square", "size": 3}],
    "measurement": {"min_area": 0},
}


def baseline_trivial(image: np.ndarray) -> PipelineResult:
    """Baseline 1: fixed global threshold of 127 on the raw grayscale image."""
    return run(image, BASELINE_TRIVIAL)


def baseline_simple(image: np.ndarray) -> PipelineResult:
    """Baseline 2: Otsu threshold followed by one 3x3 square opening."""
    return run(image, BASELINE_SIMPLE)


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Run the pipeline described by ``config`` (a parsed ``configs/*.yaml`` file).

    Supported now: ``segmentation`` (global / otsu / adaptive), ``morphology`` (a list of
    operations, see ``src.morphology.apply_operations``) and ``measurement``
    (``min_area``, ``max_aspect``). That covers both baselines. Enhancement,
    restoration and frequency stages are wired in Week 12 (Pipelines A, B, C).

    Args:
        image: ``uint8`` (H, W) grayscale image, 0-255.
        config: Parsed pipeline configuration.

    Returns:
        ``PipelineResult`` with the final ``bool`` mask (kept regions only), boxes,
        defect area as % of the image, every intermediate mask in ``stages`` and the time
        of each stage in seconds in ``timings_s``.
    """
    for stage in ("enhancement", "restoration", "frequency"):
        if config.get(stage):
            raise NotImplementedError(f"{stage} stages are wired in Week 12")
    if "segmentation" not in config:
        raise ValueError("config needs a 'segmentation' section")

    stages: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    t = time.perf_counter()
    mask = _segment(image, config["segmentation"])
    timings["segmentation"] = time.perf_counter() - t
    stages["segmentation"] = mask

    if config.get("morphology"):
        t = time.perf_counter()
        mask = morphology.apply_operations(mask, config["morphology"])
        timings["morphology"] = time.perf_counter() - t
        stages["morphology"] = mask

    measurement = config.get("measurement") or {}
    t = time.perf_counter()
    boxes, kept = measure_regions(
        mask, int(measurement.get("min_area", 0)), measurement.get("max_aspect")
    )
    timings["measurement"] = time.perf_counter() - t

    return PipelineResult(
        mask=kept,
        boxes=boxes,
        area_pct=100.0 * float(kept.mean()),
        stages=stages,
        timings_s=timings,
    )
