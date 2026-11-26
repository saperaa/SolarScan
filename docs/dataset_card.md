# Dataset card — PVEL-AD

Verified on 23 Nov 2026 against the official repository
(https://github.com/binyisu/PVEL-AD) and the IEEE DataPort entry
(DOI 10.21227/pz6t-3s77).

## Course criteria check

| Criterion (spec §1) | Requirement | PVEL-AD | Status |
|---------------------|-------------|---------|--------|
| Size | ≥ 100 real images | 36,543 EL images | ✅ |
| Ground truth | Reference answer per test image | Bounding boxes (Pascal VOC XML); test annotations now public | ✅ |
| Task difficulty | Output is a measurement; needs object/background separation | Count + location of dark defect regions | ✅ |
| Real-world mess | ≥ 2 of: low contrast, uneven illumination, noise, blur, gaps | Uneven illumination, grain texture/noise, low-contrast cracks, cracks broken after thresholding | ✅ |
| Not a toy | Real, not synthetic | Real industrial EL images | ✅ |
| Licence | Academic use permitted, cited | Academic/research use only, via a signed request form; repository code under Apache-2.0 | ⚠️ form must be signed and sent |
| Feasibility | Test set in < 10 min on free Colab CPU | Full test set too large → fixed subsample | ⚠️ subsample size set in Week 10 |

## Access procedure (action for the team lead)

1. Download `Industrial_Data_Access_Form.docx` from the PVEL-AD GitHub repository.
2. Fill in purpose ("CSE281 Digital Image Processing course project, Alamein International
   University — classical, training-free defect localization"), team lead and members.
3. **Institutional emails only** (Gmail etc. are rejected). Use AIU addresses.
4. **Hand-sign and date** the form.
5. Email it to the address given in the repository README, and include a Google account
   email if you want Google Drive access.
6. The authors say replies can take up to 2 weeks → send it **this week**.

Record the date sent and the date access was granted here:

| Event | Date | By |
|-------|------|----|
| Form sent | | |
| Access granted | | |
| Data downloaded | | |

## Classes (from the official README)

| Class | trainval boxes | test boxes | In scope? |
|-------|---------------:|-----------:|-----------|
| finger | 2,958 | 22,638 | yes |
| crack | 1,260 | 2,797 | yes |
| black_core | 1,028 | 3,877 | yes |
| thick_line | 981 | 1,585 | yes |
| horizontal_dislocation | 798 | 1,582 | yes (decided 26 Nov) |
| short_circuit | 492 | 1,215 | yes (decided 26 Nov) |
| vertical_dislocation | 137 | 271 | yes (decided 26 Nov) |
| star_crack | 135 | 83 | yes |
| printing_error | 32 | 48 | no (too rare) |
| corner | 9 | 12 | no |
| fragment | 7 | 5 | no |
| scratch | 5 | 3 | no |

## Known challenges

- **Polycrystalline grain boundaries** look dark and crack-like → main expected false-positive source.
- **Busbars and grid fingers** are periodic → visible as peaks in the Fourier spectrum.
- **Very large dataset** → the dev and test sets are fixed random subsamples
  (seeded), with the sampling recorded in `data/splits/`.
- **Two image sizes** (dev set): 2,794 cells are 1024 × 1024 and 356 are 512 × 512. Pixel
  parameters (`block_size`, `min_area`, structuring-element size) must scale with the
  image width, or the 512² cells get twice the effective kernel size.
- **Near-black cells**: the 10th percentile of per-image mean intensity is about 1, so some
  cells are almost entirely dark (black core / dead cells). Thresholding marks the whole
  cell; check how these affect IoU before tuning.

## Class decision (26 Nov)

The three classes left open in Week 9 are **in scope**. All three have a visible dark
signature in EL images and more than 50 training boxes, which is the rule in
`docs/problem_statement.md`:

| Class | trainval boxes | Reason |
|-------|---------------:|--------|
| horizontal_dislocation | 798 | dark band across the cell |
| short_circuit | 492 | whole cell or large region dark |
| vertical_dislocation | 137 | dark band, above the 50-box minimum |

Boxes of `corner`, `fragment`, `scratch` and `printing_error` are ignored by the evaluator,
so an image whose only boxes are out of scope counts as defect-free. To change the list,
edit `src.data.IN_SCOPE_CLASSES` (or set `evaluation: classes:` in a config) and say so
in the PR.

## Fallback

If access is not granted by the end of Week 10, switch to **Concrete Crack Segmentation**
(Mendeley Data; 458 images with binary crack masks) — same structure (thin dark cracks on a
textured, unevenly lit surface), so the pipeline design carries over.

## Citation

```
B. Su, Z. Zhou, and H. Chen, "PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic
Cell Anomaly Detection," IEEE Transactions on Industrial Informatics, 2022.
doi: 10.1109/TII.2022.3162846
```
