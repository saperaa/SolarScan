"""Tests for image and dataset helpers."""

import cv2
import numpy as np
import pytest

from src.data import add_noise, downsample, load_annotations, load_image, make_split


def test_load_image_converts_color_to_grayscale(tmp_path):
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    color[:, :, 0] = 50
    color[:, :, 1] = 100
    color[:, :, 2] = 200
    path = tmp_path / "cell.png"
    assert cv2.imwrite(str(path), color)

    image = load_image(path)

    expected = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert image.dtype == np.uint8
    assert image.shape == (2, 3)
    np.testing.assert_array_equal(image, expected)


def test_load_image_reports_unreadable_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_annotations_converts_voc_coordinates_to_zero_based(tmp_path):
    xml = """\
<annotation>
  <object>
    <name>crack</name>
    <bndbox>
      <xmin>1</xmin><ymin>2</ymin><xmax>4</xmax><ymax>5</ymax>
    </bndbox>
  </object>
</annotation>
"""
    path = tmp_path / "cell.xml"
    path.write_text(xml, encoding="utf-8")

    assert load_annotations(path) == [("crack", (0, 1, 3, 4))]


def test_load_annotations_clamps_zero_based_coordinates(tmp_path):
    xml = """\
<annotation>
  <object>
    <name>finger</name>
    <bndbox>
      <xmin>0</xmin><ymin>0</ymin><xmax>3</xmax><ymax>2</ymax>
    </bndbox>
  </object>
</annotation>
"""
    path = tmp_path / "zero.xml"
    path.write_text(xml, encoding="utf-8")

    assert load_annotations(path) == [("finger", (0, 0, 2, 1))]


def test_load_annotations_returns_empty_list_for_clean_image(tmp_path):
    path = tmp_path / "clean.xml"
    path.write_text("<annotation />", encoding="utf-8")

    assert load_annotations(path) == []


def test_make_split_is_seeded_and_partitions_all_ids():
    ids = [f"cell-{index}" for index in range(10)]

    dev, test = make_split(ids, dev_fraction=0.7, seed=3)
    repeated = make_split(ids, dev_fraction=0.7, seed=3)

    assert (dev, test) == repeated
    assert len(dev) == 7
    assert len(test) == 3
    assert set(dev).isdisjoint(test)
    assert set(dev + test) == set(ids)


@pytest.mark.parametrize(
    ("kind", "level"),
    [
        ("gaussian", 10),
        ("salt_pepper", 0.2),
        ("speckle", 0.1),
        ("poisson", 20),
    ],
)
def test_add_noise_is_seeded_uint8_and_clipped(kind, level):
    image = np.full((16, 16), 127, dtype=np.uint8)

    first = add_noise(image, kind, level, seed=4)
    repeated = add_noise(image, kind, level, seed=4)

    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255
    np.testing.assert_array_equal(first, repeated)


def test_add_noise_rejects_invalid_kind_and_poisson_level():
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="kind"):
        add_noise(image, "unknown", 1, seed=0)
    with pytest.raises(ValueError, match="greater than zero"):
        add_noise(image, "poisson", 0, seed=0)


def test_downsample_matches_opencv_and_hand_worked_example():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    result = downsample(image, factor=2)

    expected = cv2.resize(
        image, (2, 2), interpolation=cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, [[0, 2], [8, 10]])


def test_downsample_matches_opencv_for_nondivisible_dimensions():
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    expected = cv2.resize(
        image, (2, 1), interpolation=cv2.INTER_NEAREST
    )

    np.testing.assert_array_equal(downsample(image, factor=3), expected)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_downsample_rejects_invalid_factor(factor):
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="factor"):
        downsample(image, factor)
