# Demo app

Launch the local Gradio app from the repository root:

```bash
python -m app.app --config configs/pipeline_c.yaml
```

Upload an unseen grayscale EL image to see the configured pipeline stages, final defect mask,
count, boxes, and area percentage. For failure analysis, optionally upload its Pascal VOC XML
annotation to compare predicted and ground-truth boxes, box-union IoU, and count error. The app
calls the shared `src.pipeline.run` backend, which is still a scaffold in this branch;
inference requires the dispatcher, baselines, and image-processing stages to be integrated.
Do not commit raw images or annotations.
