# Work plan (Weeks 9–14)

Checkpoints are for our own planning; only the Week 14 discussion is graded.

| Week | Checkpoint | Done when |
|------|------------|-----------|
| 9 | Topic settled | Dataset checked against criteria, access form sent, metric fixed, roles assigned, repo skeleton merged, first commit from every member |
| 10 | Baselines ready | Data loaded, dev/test split locked (hash recorded), metrics implemented and tested, image analysis notebook, both baselines measured, controlled degradation working |
| 11 | Stages in progress | Enhancement, restoration, frequency, segmentation, morphology modules with ≥ 2 NumPy functions per member, each checked against the library with a hand-worked test |
| 12 | Stages working | Pipelines A, B, C end to end, block diagrams with dtypes, experiment log filling up |
| 13 | Results frozen | One-factor-at-a-time tuning, four ablations, final pipeline chosen on dev, test set evaluated once, failure analysis, report drafted |
| 14 | Final discussion | Tag `v1.0`, report PDF and slides in `report/`, live demo, individual defenses |

## Ablations (planned)

1. With vs without contrast enhancement / background flattening
2. Spatial vs frequency-domain smoothing (and with vs without busbar notch filter)
3. Mean vs median under salt-and-pepper noise
4. Opening-then-closing vs closing-then-opening

## Tuning (one factor at a time)

Smoothing kernel size (3 values), cutoff D0 (3), threshold method (3),
structuring-element size (3) and shape (2). Final run: 3 noise levels × 3 seeds, mean ± std.
