# Data

Raw data is **not** committed. Expected layout after download:

```
data/
├── raw/PVEL-AD/          # extracted dataset (git-ignored)
│   ├── trainval/JPEGImages/, trainval/Annotations/
│   └── test/JPEGImages/,     test/Annotations/
└── splits/               # committed: dev.txt and test.txt (image IDs), created in Week 10
```

## How to get PVEL-AD

Access requires a hand-signed request form with institutional emails; see
[`docs/dataset_card.md`](../docs/dataset_card.md#access-procedure-action-for-the-team-lead).
After access is granted, download from the link the authors send and extract into
`data/raw/PVEL-AD/`. Check the folder names above against the actual archive and update this
file if they differ.

## Splits

`data/splits/dev.txt` and `data/splits/test.txt` are created once by `src.data.make_split`
with a fixed seed and are never regenerated. Record the SHA-256 of `test.txt` here when it is
created so anyone can check it was not changed:

| File | SHA-256 | Created |
|------|---------|---------|
| test.txt | | |
