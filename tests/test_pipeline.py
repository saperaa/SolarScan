"""Tests for pipeline helpers owned by the Tech Lead."""

import cv2
import numpy as np
import pytest

from src.pipeline import (
    baseline_simple,
    baseline_trivial,
    boxes_to_mask,
    connected_components,
    measure_regions,
    run,
)


def test_boxes_to_mask_hand_worked_6x6():
    # Box A covers x 0-2, y 0-1 (6 px); box B covers x 2-3, y 1-3 (6 px).
    # They share pixel (x=2, y=1), so the union has 6 + 6 - 1 = 11 pixels.
    mask = boxes_to_mask([(0, 0, 2, 1), (2, 1, 3, 3)], (6, 6))
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert mask.dtype == bool
    assert mask.sum() == 11
    np.testing.assert_array_equal(mask, expected)


def test_boxes_to_mask_matches_opencv_filled_rectangles():
    rng = np.random.default_rng(0)
    boxes = []
    for _ in range(5):
        x0, y0 = rng.integers(0, 30, 2)
        boxes.append((int(x0), int(y0), int(x0 + rng.integers(0, 10)),
                      int(y0 + rng.integers(0, 10))))
    drawn = np.zeros((32, 40), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(drawn, (x0, y0), (x1, y1), 255, thickness=-1)
    np.testing.assert_array_equal(boxes_to_mask(boxes, (32, 40)), drawn == 255)


def test_boxes_to_mask_empty_and_clipped():
    assert not boxes_to_mask([], (4, 4)).any()
    clipped = boxes_to_mask([(2, 2, 10, 10)], (4, 4))
    assert clipped.sum() == 4 and clipped[2:, 2:].all()


# --- connected_components (NumPy implementation, Tech Lead) ---------------------------

def _same_partition(labels_a, labels_b, mask):
    pairs = set(zip(labels_a[mask].tolist(), labels_b[mask].tolist()))
    return len(pairs) == len({a for a, _ in pairs}) == len({b for _, b in pairs})


def test_connected_components_hand_worked_5x5():
    # Two blobs. Blob A: (0,0),(0,1),(1,1),(2,2) - (1,1)->(2,2) is a DIAGONAL step,
    # so with 8-connectivity A is one region of 4 pixels.
    # Blob B: (3,4),(4,3),(4,4) - 3 pixels, not touching A.
    mask = np.array(
        [[1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 1]],
        dtype=bool,
    )
    n, labels = connected_components(mask)
    expected = np.array(
        [[1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 2],
         [0, 0, 0, 2, 2]],
    )
    assert n == 2
    np.testing.assert_array_equal(labels, expected)
    assert np.bincount(labels.ravel()).tolist() == [18, 4, 3]


def test_connected_components_u_shape_merges_later():
    # A "U": the two arms only meet on the bottom row, so the union step must merge
    # two labels that started separately.
    mask = np.array(
        [[1, 0, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=bool,
    )
    n, labels = connected_components(mask)
    assert n == 1 and set(labels[mask].tolist()) == {1}


@pytest.mark.parametrize("density", [0.2, 0.5, 0.7])
def test_connected_components_matches_opencv(density):
    mask = np.random.default_rng(int(density * 10)).random((60, 80)) < density
    n, labels = connected_components(mask)
    n_cv, labels_cv = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    assert n == n_cv - 1
    assert _same_partition(labels, labels_cv, mask)


def test_connected_components_empty():
    n, labels = connected_components(np.zeros((4, 4), dtype=bool))
    assert n == 0 and not labels.any()


# --- measurement ------------------------------------------------------------------------

def test_measure_regions_boxes_area_and_aspect_filters():
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:5, 2:5] = True        # 3x3 blob, area 9
    mask[10, 0:30] = True        # 1x30 line, aspect 30 (a grid line)
    mask[15, 20] = True          # single-pixel speck

    boxes, kept = measure_regions(mask)
    assert boxes == [(2, 2, 4, 4), (0, 10, 29, 10), (20, 15, 20, 15)]

    boxes, kept = measure_regions(mask, min_area=2, max_aspect=10)
    assert boxes == [(2, 2, 4, 4)]
    assert kept.sum() == 9 and kept.dtype == bool


# --- baselines and run ------------------------------------------------------------------

def _synthetic_cell():
    img = np.full((64, 64), 180, dtype=np.uint8)
    img[20:30, 10:40] = 40          # one dark defect
    img[50, 50] = 90                # one dark speck
    return img


def test_baseline_trivial_finds_dark_regions():
    result = baseline_trivial(_synthetic_cell())
    assert result.boxes == [(10, 20, 39, 29), (50, 50, 50, 50)]
    assert result.decision == "defective"
    assert result.mask.dtype == bool and result.mask.shape == (64, 64)
    assert result.area_pct == pytest.approx(100 * 301 / 4096)


def test_baseline_simple_opening_removes_speck():
    result = baseline_simple(_synthetic_cell())
    assert result.boxes == [(10, 20, 39, 29)]
    assert set(result.stages) == {"segmentation", "morphology"}
    assert set(result.timings_s) == {"segmentation", "morphology", "measurement"}


def test_clean_cell_gives_no_boxes():
    clean = np.full((32, 32), 200, dtype=np.uint8)
    assert baseline_trivial(clean).decision == "ok"


def test_run_reads_baseline_yaml_configs():
    import yaml

    img = _synthetic_cell()
    for name, fn in (("baseline_trivial", baseline_trivial),
                     ("baseline_simple", baseline_simple)):
        with open(f"configs/{name}.yaml", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        assert run(img, config).boxes == fn(img).boxes


def test_run_rejects_unwired_stages():
    with pytest.raises(NotImplementedError):
        run(_synthetic_cell(), {"enhancement": [{"op": "gamma"}],
                                "segmentation": {"method": "otsu"}})
