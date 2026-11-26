import numpy as np

from src import psnr_study as p


def test_run_study_row_count_and_ranges():
    img = np.tile(np.arange(0, 64, dtype=np.uint8), (32, 1)) * 3
    rows = p.run_study({"ramp": img}, seed=0)
    n_levels = sum(len(v) for v in p.NOISES.values())
    assert len(rows) == n_levels * len(p.FILTERS)
    assert all(r["mse"] >= 0 and r["psnr"] > 0 for r in rows)


def test_run_study_is_reproducible():
    img = np.full((24, 24), 120, dtype=np.uint8)
    assert p.run_study({"flat": img}, 1) == p.run_study({"flat": img}, 1)


def test_median_beats_mean_on_salt_pepper():
    # Median removes impulses; the arithmetic mean only smears them.
    img = p.synthetic_images(64)["cell"]
    rows = p.run_study({"cell": img}, seed=0)
    best = p.best_filters(rows)[("cell", "salt_pepper", 0.15)]
    psnr = {r["filter"]: r["psnr"] for r in rows
            if r["noise"] == "salt_pepper" and r["level"] == 0.15}
    assert psnr["median"] > psnr["arithmetic_mean"]
    assert best[1] == max(psnr.values())
