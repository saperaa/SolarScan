"""Smoke tests for the shared pipeline types (owner: Tech Lead)."""

import numpy as np

from src.pipeline import PipelineResult


def test_pipeline_result_decision():
    empty = PipelineResult(mask=np.zeros((4, 4), dtype=bool))
    assert empty.count == 0 and empty.decision == "ok"

    found = PipelineResult(mask=np.ones((4, 4), dtype=bool), boxes=[(0, 0, 3, 3)])
    assert found.count == 1 and found.decision == "defective"
