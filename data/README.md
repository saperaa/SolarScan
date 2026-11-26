# Data

Raw data is **not** committed. Expected layout after download:

```
data/
├── raw/PVEL-AD/          # extracted images (git-ignored)
│   └── PVELAD/EL2021/
│       ├── trainval/JPEGImages/, trainval/Annotations/
│       └── test/JPEGImages/
├── raw/PVEL-AD-test-annotations/test/  # XMLs distributed separately
└── splits/               # committed: dev.txt and test.txt (image IDs), created in Week 10
```

## How to get PVEL-AD

Access requires a hand-signed request form with institutional emails; see
[`docs/dataset_card.md`](../docs/dataset_card.md#access-procedure-action-for-the-team-lead).
After access is granted, download from the link the authors send and extract the images into
`data/raw/PVEL-AD/`. The current test XML download is separate from the test JPEG directory;
keep it under `data/raw/PVEL-AD-test-annotations/` or pass its location to the evaluator.
Check the folder names above against future archives and update this file if they differ.

## Splits

`data/splits/dev.txt` contains 3,150 IDs selected from the 4,500 trainval pairs with
`src.data.make_split(dev_fraction=0.7, seed=0)`. IDs are sorted in the manifest after
selection for a stable diff. It must not be regenerated.

Use `src.data.save_split(dev_ids, test_ids, out_dir)` to write one ID per line in sorted order
and return the SHA-256 of the exact `test.txt` bytes. It refuses to overwrite an existing
manifest. The held-out sample in `test.txt` comes from the separate external test set; fix its
size and seed before final evaluation. Never use test results for tuning.

| File | Source | SHA-256 | Count | Created |
|------|--------|---------|------:|---------|
| dev.txt | trainval, seed 0, dev fraction 0.7 | `02c8a05de5281d5deb55534bbb1cfb1265d4b8495b6c3c23f0d0fd9e58f0ca5e` | 3,150 | 2026-11-26 |
| test.txt | PVEL-AD test, sample not yet locked | | | |
