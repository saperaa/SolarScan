"""PSNR-per-noise-type study for the restoration filters (Member 4).

Run from the repo root:  python -m src.psnr_study --out experiments/psnr_synthetic.csv
Until PVEL-AD arrives it uses synthetic images, so the numbers are a dry run
and must not be reported as dev-set results.
"""
import argparse
import csv

import numpy as np
from skimage import data as skdata

from src import metrics, restoration
from src.data import add_noise

NOISES = {
    "gaussian": (10.0, 25.0),
    "salt_pepper": (0.05, 0.15),
    "speckle": (0.1, 0.3),
    "poisson": (30.0, 100.0),
}

FILTERS = {
    "none": lambda img: img,
    "arithmetic_mean": restoration.arithmetic_mean,
    "geometric_mean": restoration.geometric_mean,
    "harmonic_mean": restoration.harmonic_mean,
    "contraharmonic_q+1.5": lambda i: restoration.contraharmonic_mean(i, 3, 1.5),
    "contraharmonic_q-1.5": lambda i: restoration.contraharmonic_mean(i, 3, -1.5),
    "median": restoration.median_filter,
    "max": restoration.max_filter,
    "min": restoration.min_filter,
    "midpoint": restoration.midpoint_filter,
    "alpha_trimmed_d2": restoration.alpha_trimmed_mean,
}


def synthetic_images(size=128):
    """Return a dict of name -> uint8 (size, size) test images.

    'camera' is a crop of scikit-image's bundled photo. 'cell' imitates an
    EL cell: bright centre, darker edges and vertical busbar stripes.
    """
    camera = skdata.camera()[:size, :size].copy()
    y, x = np.mgrid[0:size, 0:size]
    radial = np.hypot(y - size / 2, x - size / 2) / (size / 2)
    cell = 200.0 - 80.0 * np.clip(radial, 0, 1) ** 2
    cell[:, size // 3::size // 3] -= 40.0
    return {"camera": camera, "cell": np.clip(cell, 0, 255).astype(np.uint8)}


def run_study(images, seed=0):
    """Return a list of result dicts (image, noise, level, filter, mse, psnr).

    images: dict name -> clean uint8 (H, W) image. Noise is added with the
    fixed seed, then every filter in FILTERS is applied and compared with
    the clean image using metrics.mse and metrics.psnr.
    """
    rows = []
    for name, clean in images.items():
        for kind, levels in NOISES.items():
            for level in levels:
                noisy = add_noise(clean, kind, level, seed)
                for fname, fn in FILTERS.items():
                    out = fn(noisy)
                    rows.append({
                        "image": name, "noise": kind, "level": level,
                        "filter": fname, "seed": seed,
                        "mse": round(metrics.mse(out, clean), 4),
                        "psnr": round(metrics.psnr(out, clean), 4),
                    })
    return rows


def best_filters(rows):
    """Return {(image, noise, level): (filter, psnr)} with the highest PSNR."""
    best = {}
    for row in rows:
        key = (row["image"], row["noise"], row["level"])
        if key not in best or row["psnr"] > best[key][1]:
            best[key] = (row["filter"], row["psnr"])
    return best


def main():
    """Run the study on synthetic images, write a CSV and print the winners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run_study(synthetic_images(), args.seed)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key, (fname, value) in best_filters(rows).items():
        print(key, "->", fname, value)


if __name__ == "__main__":
    main()
