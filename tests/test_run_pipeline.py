"""Tests for the single-image pipeline runner and output conversion."""

import cv2
import numpy as np
import pytest

import src.run_pipeline as runner
from src.pipeline import PipelineResult


def test_process_image_returns_shared_pipeline_result(monkeypatch):
    image = np.zeros((3, 4), dtype=np.uint8)
    expected = PipelineResult(mask=np.zeros(image.shape, dtype=bool))
    received = {}

    def fake_run(input_image, config):
        received["image"] = input_image
        received["config"] = config
        return expected

    monkeypatch.setattr(runner, "run", fake_run)

    result = runner.process_image(image, {"name": "test"})

    assert result is expected
    assert received["image"] is image
    assert received["config"] == {"name": "test"}


@pytest.mark.parametrize(
    "image",
    [None, np.zeros((0, 3)), np.zeros((3, 3, 1))],
)
def test_process_image_rejects_invalid_images(image):
    with pytest.raises(ValueError, match="image"):
        runner.process_image(image, {})


def test_process_image_requires_uint8_image():
    with pytest.raises(ValueError, match="uint8"):
        runner.process_image(np.zeros((3, 4), dtype=np.float32), {})


def test_process_image_rejects_pipeline_mask_shape_mismatch(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run",
        lambda image, config: PipelineResult(mask=np.zeros((1, 1), dtype=bool)),
    )

    with pytest.raises(ValueError, match="pipeline mask"):
        runner.process_image(np.zeros((3, 4), dtype=np.uint8), {})


def test_display_image_converts_boolean_and_normalized_float_stages():
    boolean_rgb = runner.display_image(np.array([[True, False]]))
    float_rgb = runner.display_image(np.array([[0.0, 1.0]]))

    np.testing.assert_array_equal(boolean_rgb[0, :, 0], [255, 0])
    np.testing.assert_array_equal(float_rgb[0, :, 0], [0, 255])
    assert boolean_rgb.shape == (1, 2, 3)


def test_write_result_saves_stages_mask_and_box_overlay(tmp_path):
    image = np.zeros((5, 6), dtype=np.uint8)
    mask = np.zeros(image.shape, dtype=bool)
    mask[1:3, 2:4] = True
    result = PipelineResult(
        mask=mask,
        boxes=[(2, 1, 3, 2)],
        stages={"segmentation": mask},
    )

    output = runner.write_result(tmp_path, "sample.jpg", image, result)

    assert sorted(path.name for path in output.iterdir()) == [
        "boxes.png",
        "final_mask.png",
        "input.png",
        "segmentation.png",
    ]
    saved_mask = cv2.imread(str(output / "final_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert saved_mask[1, 2] == 255
    overlay = cv2.imread(str(output / "boxes.png"), cv2.IMREAD_COLOR)
    assert overlay[1, 2].tolist() == [0, 0, 255]


def test_write_result_rejects_nonfinite_stage_values(tmp_path):
    image = np.zeros((2, 2), dtype=np.uint8)
    result = PipelineResult(
        mask=np.zeros(image.shape, dtype=bool),
        stages={"broken": np.array([[np.nan, 0], [0, 1]])},
    )

    with pytest.raises(ValueError, match="finite"):
        runner.write_result(tmp_path, "sample.png", image, result)


def test_write_result_rejects_mask_with_mismatched_shape(tmp_path):
    with pytest.raises(ValueError, match="match the input"):
        runner.write_result(
            tmp_path,
            "sample.png",
            np.zeros((2, 2), dtype=np.uint8),
            PipelineResult(mask=np.zeros((1, 1), dtype=bool)),
        )
