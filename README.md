# SolarScan

Training-free localization of defects in solar-cell electroluminescence (EL) images
using classical digital image processing only (no neural networks).

CSE281 Digital Image Processing — course project, Alamein International University.

## Problem statement

A PV manufacturer or solar-farm operator inspects electroluminescence (EL) images of solar
cells to find defective cells before they reduce a module's power output. EL images are taken
while the cell is electrically biased, so active silicon glows and defects (cracks, finger
interruptions, black cores) appear as dark regions. SolarScan takes one raw grayscale EL cell
image and outputs the number of defect regions, a bounding box for each, and the defect area
as a percentage of the cell area. A missed defect leaves a power-losing cell in service; a
false alarm sends a good cell to manual re-inspection. The pipeline must therefore be
reliable in both directions, which is why we report both recall-sensitive and
precision-sensitive metrics.

Full details: [`docs/problem_statement.md`](docs/problem_statement.md)

## Dataset

**PVEL-AD** (Su, Zhou, Chen, *IEEE Trans. Ind. Inform.*, 2022) — 36,543 near-infrared EL
images of polycrystalline solar cells with bounding-box annotations for 12 defect classes.
Access is granted for research and academic use through a signed request form.
See [`docs/dataset_card.md`](docs/dataset_card.md) and [`data/README.md`](data/README.md).

Raw data is **not** stored in git.

## Metrics

| Role | Metric |
|------|--------|
| Primary | Per-image IoU between the union of predicted boxes and the union of ground-truth boxes |
| Secondary | Detection precision / recall / F1 at IoU ≥ 0.3; absolute count error per image |
| Paper-compatible | mAP@0.5 (requires a confidence score per box) |

Definitions and edge cases: [`docs/metrics.md`](docs/metrics.md)

## Repository structure

Target layout. Each `src/` file is created by the member who owns it (see
[`docs/roles.md`](docs/roles.md)); files appear as their pull requests are merged.

```
SolarScan/
├── README.md
├── requirements.txt       pinned package versions
├── configs/               one file per pipeline configuration
├── data/                  download / preparation instructions (no raw data in git)
├── docs/                  problem statement, dataset card, metrics, roles, work plan
├── notebooks/             01_image_analysis.ipynb, 02_results.ipynb
├── src/
│   ├── data.py            loading, splitting, controlled degradation
│   ├── metrics.py         MSE, PSNR, IoU, count error
│   ├── baselines.py       NumPy global threshold for Baseline 1
│   ├── enhancement.py     point operations, histograms, spatial filters
│   ├── restoration.py     noise models and restoration filters
│   ├── frequency.py       DFT, spectrum, frequency-domain filters
│   ├── segmentation.py    edge detection, thresholding, Hough transform
│   ├── morphology.py      morphological operations, hole filling, measurement
│   ├── pipeline.py        baselines and Pipelines A, B, C
│   ├── evaluate.py        test-set metrics and failure analysis
│   └── run_pipeline.py    full pipeline on a raw image
├── tests/                 unit tests (hand-worked examples)
├── experiments/           experiment log (CSV) and saved figures
├── app/                   demo application
└── report/                report PDF and slides
```

## Setup

```bash
git clone https://github.com/saperaa/SolarScan.git
cd SolarScan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Usage

```bash
# Not implemented yet — available from Week 12
python -m src.run_pipeline --image path/to/cell.jpg --config configs/pipeline_c.yaml
```

## Results

_To be filled in Week 13 after the single test-set evaluation._

| Method | IoU (mean ± std) | F1 @ IoU 0.3 | Count error | Time / image |
|--------|------------------|--------------|-------------|--------------|
| Baseline 1 — threshold 127 | – | – | – | – |
| Baseline 2 — Otsu + 3×3 opening | – | – | – | – |
| Pipeline A | – | – | – | – |
| Pipeline B | – | – | – | – |
| Pipeline C | – | – | – | – |

## Team

See [`docs/roles.md`](docs/roles.md). Workflow rules: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

```
B. Su, Z. Zhou, and H. Chen, "PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic
Cell Anomaly Detection," IEEE Trans. Ind. Inform., 2022, doi: 10.1109/TII.2022.3162846.
```
