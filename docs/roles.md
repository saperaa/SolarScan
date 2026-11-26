# Team roles

Each member owns one work package and is examined on it. Every member also writes at least
**two NumPy implementations** (checked against OpenCV / scikit-image, with a hand-worked
3×3 or 5×5 example) and reports at least **one experiment**.

Fill in names and GitHub usernames in Week 9. Each member creates their own files; the
full per-member task list is in the team work guide shared by the Tech Lead.

| # | Role | Member | GitHub | Owns files | NumPy implementations (≥ 2) | Main experiment |
|---|------|--------|--------|------------|-----------------------------|-----------------|
| 1 | Tech Lead | Mohamed Alaa | @saperaa | `pipeline.py`, `configs/` | Connected-component labelling; `boxes_to_mask` | Final pipeline at 3 noise levels × 3 seeds |
| 2 | Data & Baseline | | | `data.py`, `metrics.py`, baseline configs | Global threshold; nearest-neighbour downsampling | Resolution & gray-level study; Pipeline A vs baselines |
| 3 | Enhancement | | | `enhancement.py` | 2-D convolution; histogram equalization | Enhancement comparison; kernel-size study |
| 4 | Restoration & Frequency | Omar Hany Elazab | @OmarElshaer1 | `restoration.py`, `frequency.py` | Median filter; Butterworth / notch filter | PSNR per noise type; spatial vs frequency ablation |
| 5 | Segmentation | | | `segmentation.py` | Otsu threshold; Sobel magnitude | Detector & threshold-method studies |
| 6 | Morphology & Evaluation | | | `morphology.py`, `evaluate.py`, `run_pipeline.py`, `app/` | Erosion & dilation; hole filling | Structuring-element study; failure analysis |

**Team of five:** merge roles 1 and 2 — the Tech Lead also owns data, metrics and baselines.

## Contribution table (for the report)

Updated at the end of each phase.

| Member | Work package | Files | Experiments run | PRs merged | PRs reviewed |
|--------|--------------|-------|-----------------|-----------:|-------------:|
| | | | | | |
