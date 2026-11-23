"""Baselines and Pipelines A, B, C.

Owner: Tech Lead (role 1). Each pipeline is driven by one YAML file in ``configs/``.

Data flow between stages (to be completed with the block diagrams in Week 12):

    raw image  --load_image-->  uint8 (H, W) 0-255
      --> enhancement / restoration --> uint8 or float64 (H, W)
      --> segmentation              --> bool (H, W), True = defect candidate
      --> morphology                --> bool (H, W)
      --> measurement               --> list of boxes, count, area %
"""

from dataclasses import dataclass, field

import numpy as np

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


def baseline_trivial(image: np.ndarray) -> PipelineResult:
    """Baseline 1: fixed global threshold of 127 on the raw grayscale image."""
    raise NotImplementedError("Week 10 - pipeline.baseline_trivial")


def baseline_simple(image: np.ndarray) -> PipelineResult:
    """Baseline 2: Otsu threshold followed by one 3x3 opening."""
    raise NotImplementedError("Week 10 - pipeline.baseline_simple")


def run(image: np.ndarray, config: dict) -> PipelineResult:
    """Run the pipeline described by ``config`` (a parsed ``configs/*.yaml`` file)."""
    raise NotImplementedError("Week 12 - pipeline.run")
