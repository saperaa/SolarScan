"""Smoke tests for the Gradio demo interface."""

import numpy as np

import app.app as demo_module
from src.pipeline import PipelineResult


class FakeComponent:
    def __init__(self, **kwargs):
        self.options = kwargs


class FakeInterface:
    def __init__(self, **kwargs):
        self.options = kwargs

    @property
    def fn(self):
        return self.options["fn"]


def _replace_gradio_components(monkeypatch):
    monkeypatch.setattr(demo_module.gr, "Image", FakeComponent)
    monkeypatch.setattr(demo_module.gr, "File", FakeComponent)
    monkeypatch.setattr(demo_module.gr, "JSON", FakeComponent)
    monkeypatch.setattr(demo_module.gr, "Gallery", FakeComponent)
    monkeypatch.setattr(demo_module.gr, "Interface", FakeInterface)


def test_create_app_builds_gradio_interface(monkeypatch, tmp_path):
    _replace_gradio_components(monkeypatch)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("name: synthetic\n", encoding="utf-8")

    demo = demo_module.create_app(config_path)

    assert isinstance(demo, FakeInterface)
    assert demo.options["title"] == "SolarScan"
    assert len(demo.options["inputs"]) == 2
    assert demo.options["inputs"][1].options["label"].startswith(
        "Optional Pascal VOC annotation"
    )


def test_demo_callback_displays_stages_and_predictions(monkeypatch, tmp_path):
    _replace_gradio_components(monkeypatch)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("name: synthetic\n", encoding="utf-8")
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros(image.shape, dtype=bool)
    mask[1:3, 1:3] = True
    result = PipelineResult(
        mask=mask,
        boxes=[(1, 1, 2, 2)],
        area_pct=25.0,
        stages={"segmentation": mask},
    )
    monkeypatch.setattr(demo_module, "load_image", lambda path: image)
    monkeypatch.setattr(
        demo_module,
        "load_annotations",
        lambda path: [("crack", (1, 1, 2, 2))],
    )
    monkeypatch.setattr(demo_module, "process_image", lambda pixels, config: result)

    demo = demo_module.create_app(config_path)
    summary, gallery, overlay, ground_truth_overlay = demo.fn(
        "uploaded.jpg", "annotation.xml"
    )
    assert summary["decision"] == "defective"
    assert summary["count"] == 1
    assert summary["box_union_iou"] == 1.0
    assert summary["count_error"] == 0
    assert [caption for _, caption in gallery] == [
        "input",
        "segmentation",
        "final_mask",
    ]
    assert overlay.shape == (4, 4, 3)
    np.testing.assert_array_equal(ground_truth_overlay, overlay)

    prediction_only, _, _, no_ground_truth_overlay = demo.fn(
        "uploaded.jpg", None
    )
    assert "box_union_iou" not in prediction_only
    assert no_ground_truth_overlay is None
