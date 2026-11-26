# Metrics

All metrics are implemented once in `src/metrics.py`, tested on hand-worked arrays in
`tests/test_metrics.py`, and used everywhere else.

## Primary metric — per-image box-union IoU

For one image, rasterise all ground-truth boxes into a binary mask `G` and all predicted boxes
into a binary mask `P` (same size as the image), then

```
IoU = |P ∩ G| / |P ∪ G|
```

**Why:** the course asks for IoU per image for detection tasks. Using the union of boxes
avoids having to match boxes one-to-one, rewards covering the defect area, and punishes
spurious boxes (they enlarge the union).

**Edge cases (fixed now, before any results):**

| Ground truth | Prediction | IoU |
|--------------|------------|-----|
| no boxes | no boxes | 1.0 (correctly called clean) |
| no boxes | ≥ 1 box | 0.0 (false alarm) |
| ≥ 1 box | no boxes | 0.0 (missed defect) |

Reported as mean ± std over the test set, plus best and worst image.

## Secondary metrics

### Detection precision / recall / F1 at IoU ≥ 0.3

Greedy one-to-one matching of predicted to ground-truth boxes by descending IoU; a pair is a
true positive if its box IoU ≥ 0.3.

**Why 0.3 and not 0.5:** ground-truth boxes for thin cracks are drawn loosely around a
1–3 px wide structure, so a correct, tight prediction can have low box IoU with a loose
annotation. 0.3 accepts "found the right defect" while rejecting boxes in the wrong place.
We also report F1 at 0.5 for comparison.

The evaluator aggregates TP, FP, and FN across evaluated images (micro average) before
calculating precision, recall, and F1. A zero denominator produces a reported value of 0.
Per-image counts and metrics are written to `experiments/runs/`; the aggregate summary also
lists the best- and worst-IoU images and the ten lowest-IoU cases.

### Absolute count error

`|n_pred − n_gt|` per image, mean over the test set.

### mAP@0.5 (paper-compatible, optional)

Needs a confidence score per predicted box (planned: mean darkness contrast between the
region and its surroundings). Lets us compare with published PVEL-AD results using the
dataset authors' evaluation script (`AP50-5-95.py` in the PVEL-AD repository).

## Restoration metrics

Used only for the controlled-degradation study (noise added to clean images):

```
MSE  = mean((I − K)²)
PSNR = 10 · log10(255² / MSE)      (dB; +inf when MSE = 0)
```
