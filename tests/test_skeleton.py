"""Smoke tests for the shared pipeline types (owner: Tech Lead)."""

import numpy as np
import pytest

from src.pipeline import PipelineResult, boxes_to_mask


def test_pipeline_result_decision():
    empty = PipelineResult(mask=np.zeros((4, 4), dtype=bool))
    assert empty.count == 0 and empty.decision == "ok"

    found = PipelineResult(mask=np.ones((4, 4), dtype=bool), boxes=[(0, 0, 3, 3)])
    assert found.count == 1 and found.decision == "defective"


def test_boxes_to_mask_rasterizes_inclusive_boxes_and_clips_to_image():
    actual = boxes_to_mask([(-1, 1, 1, 3), (3, 3, 8, 8)], (4, 5))
    expected = np.zeros((4, 5), dtype=bool)
    expected[1:4, 0:2] = True
    expected[3, 3:5] = True

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "boxes,shape",
    [
        ([(0, 0, 1, 1)], (0, 2)),
        ([(0, 0, 1, 1)], (2,)),
        ([(0, 1, 1, 0)], (2, 2)),
        ([(0, 0, 1.5, 1)], (2, 2)),
        ([("x", 0, 1, 1)], (2, 2)),
    ],
)
def test_boxes_to_mask_rejects_invalid_inputs(boxes, shape):
    with pytest.raises(ValueError):
        boxes_to_mask(boxes, shape)
