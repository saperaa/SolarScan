# Team roles

Each member owns one work package and is examined on it. Every member also writes at least
**two NumPy implementations** (checked against OpenCV / scikit-image, with a hand-worked
3×3 or 5×5 example) and reports at least **one experiment**.

Fill in names and GitHub usernames in Week 9. Each member creates their own files; the
full per-member task list is in the team work guide shared by the Tech Lead.

| # | Role | Member | GitHub | Owns files | NumPy implementations (≥ 2) | Main experiment |
|---|------|--------|--------|------------|-----------------------------|-----------------|
| 1 | Tech Lead | Mohamed Alaa | @saperaa | `pipeline.py`, `configs/` | Connected-component labelling; `boxes_to_mask` | Final pipeline at 3 noise levels × 3 seeds |
| 2 | Data & Baseline | Abdelrahman | @abdelrahman22wanas | `data.py`, `metrics.py`, `baselines.py`, baseline configs | Global threshold; nearest-neighbour downsampling | Resolution & gray-level study; Pipeline A vs baselines |
| 3 | Enhancement | Yousef Elhusseiny | @YousefElhusseiny | `enhancement.py` | 2-D convolution; histogram equalization | Enhancement comparison; kernel-size study |
| 4 | Restoration & Frequency | Omar Hany Elazab | @OmarElshaer1 | `restoration.py`, `frequency.py`, `psnr_study.py` | Median filter; Butterworth / notch filter | PSNR per noise type; spatial vs frequency ablation |
| 5 | Segmentation | Mohamed Alaa | @saperaa | `segmentation.py` | Otsu threshold; Sobel magnitude | Detector & threshold-method studies |
| 6 | Morphology & Evaluation | | @abdelrahman22wanas | `morphology.py`, `evaluate.py`, `run_pipeline.py`, `app/` | Erosion & dilation; hole filling | Structuring-element size/shape study; failure analysis |

**Current split:** the Tech Lead (Mohamed) also owns role 5, Segmentation.

## Contribution table (for the report)

Updated at the end of each phase.

| Member | Work package | Files | Experiments run | PRs merged | PRs reviewed |
|--------|--------------|-------|-----------------|-----------:|-------------:|
| @abdelrahman22wanas | Morphology & Evaluation (PR #16 open) | `src/morphology.py`, `src/evaluate.py`, `src/run_pipeline.py`, `app/`, tests, docs | Not run — shared pipeline stages are incomplete | 0 | 0 |
|Yousef Elhusseiny |Enhancement Engineer |src/enhancement.py, tests/test_enhancement.py |Image enhancement testing, convolution comparison with OpenCV, filtering and background flattening |#11 |—  |
