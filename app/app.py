"""Gradio demo for the configured SolarScan pipeline."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from src.config import load_pipeline_config
from src.data import load_annotations, load_image
from src.metrics import box_union_iou, count_error
from src.run_pipeline import display_image, overlay_boxes, process_image


def create_app(config_path: str | Path = "configs/pipeline_c.yaml") -> gr.Interface:
    """Create the demo UI with a preloaded YAML pipeline config."""
    config: dict[str, Any] = load_pipeline_config(config_path)

    def analyze(image_path: str | None, annotation_path: str | None = None):
        if not image_path:
            raise gr.Error("Upload an image before running the pipeline.")
        image = load_image(image_path)
        result = process_image(image, config)
        stages = [("input", image), *result.stages.items(), ("final_mask", result.mask)]
        gallery = [(display_image(stage), name) for name, stage in stages]
        summary = {
            "decision": result.decision,
            "count": result.count,
            "area_pct": result.area_pct,
            "boxes": result.boxes,
            "timings_s": result.timings_s,
        }
        ground_truth_overlay = None
        if annotation_path:
            ground_truth = load_annotations(annotation_path)
            ground_truth_boxes = [box for _, box in ground_truth]
            summary.update(
                {
                    "ground_truth_count": len(ground_truth_boxes),
                    "box_union_iou": box_union_iou(
                        result.boxes, ground_truth_boxes, image.shape
                    ),
                    "count_error": count_error(
                        len(result.boxes), len(ground_truth_boxes)
                    ),
                }
            )
            ground_truth_overlay = overlay_boxes(image, ground_truth_boxes)
        return (
            summary,
            gallery,
            overlay_boxes(image, result.boxes),
            ground_truth_overlay,
        )

    return gr.Interface(
        fn=analyze,
        inputs=[
            gr.Image(type="filepath", label="Electroluminescence image"),
            gr.File(
                type="filepath",
                file_types=[".xml"],
                label="Optional Pascal VOC annotation (for failure analysis)",
            ),
        ],
        outputs=[
            gr.JSON(label="Measurements"),
            gr.Gallery(label="Pipeline stages"),
            gr.Image(label="Predicted boxes"),
            gr.Image(label="Ground-truth boxes"),
        ],
        title="SolarScan",
        description="Upload a grayscale EL image to inspect each pipeline stage.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the local demo app."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/pipeline_c.yaml",
        help="pipeline YAML configuration",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args(argv)

    create_app(args.config).launch(
        server_name=args.server_name,
        server_port=args.server_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
