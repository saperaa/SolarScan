# Problem statement

## One-paragraph statement

A PV manufacturer or solar-farm operator inspects electroluminescence (EL) images of solar
cells to find defective cells before they reduce a module's power output. SolarScan takes one
raw grayscale EL cell image and outputs the number of defect regions, a bounding box for each,
and the defect area as a percentage of the cell area. A missed defect (false negative) leaves
a power-losing cell in service and can grow into a hot spot; a false alarm (false positive)
sends a good cell to manual re-inspection and costs technician time. Both errors have a real
cost, so the primary metric must penalise missed and spurious detections together.

## Input

| Property | Value |
|----------|-------|
| Modality | Near-infrared electroluminescence image of a single solar cell |
| Cell type | Polycrystalline silicon (PVEL-AD) |
| Colour | Grayscale (stored as 3-channel JPEG; converted to 1 channel on load) |
| Bit depth | 8-bit, `uint8` 0–255 |
| Size | To be measured in the Week 10 image analysis (expected ~1024 × 1024) |

## Output

For each image:

1. `count` — number of detected defect regions (int)
2. `boxes` — list of `(x_min, y_min, x_max, y_max)` in pixel coordinates
3. `area_pct` — total defect pixel area / cell area × 100 (reported, not evaluated: the
   ground truth has boxes, not masks)
4. `decision` — `"defective"` if `count > 0`, else `"ok"`
5. Output image with boxes drawn

## Scope

- **In scope:** defect classes with enough examples and a visible dark signature:
  `crack`, `finger`, `black_core`, `thick_line`, `star_crack`.
  Final class list to be confirmed in the Week 10 image analysis.
- **Out of scope:** rare classes with fewer than ~50 training examples
  (`corner`, `fragment`, `scratch`, `printing_error`) and classification of the defect type.
  SolarScan localises defects; it does not name them.
- **Method:** classical image processing only. No trained models (course rule).

## Why this is a good fit for the course

- Uneven brightness across a cell (illumination) → background flattening, gamma, CLAHE
- Grain texture and sensor noise → smoothing comparison, restoration filters
- Periodic busbars and grid fingers → frequency-domain notch filtering, Hough transform
- Thin cracks broken into fragments by thresholding → closing, hole filling
- Output is a count + locations → directly checkable against bounding boxes

## Why each error matters (metric justification)

| Error | Consequence | Captured by |
|-------|-------------|-------------|
| Missed defect | Faulty cell stays in service | Recall, IoU |
| False alarm | Unneeded re-inspection | Precision, IoU, count error |
| Box in roughly the right place but wrong size | Area estimate wrong | IoU |
| Defect split into many fragments | Count inflated | Count error |
