"""Tests for the Phase 2 command-line tools: run_baselines and lock_test_split."""

import csv
import hashlib

import cv2
import numpy as np
import pytest

from src import lock_test_split, run_baselines


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_cell(images, annotations, image_id, box=None):
    img = np.full((40, 40), 180, dtype=np.uint8)
    xml = "<annotation>"
    if box:
        x0, y0, x1, y1 = box
        img[y0:y1 + 1, x0:x1 + 1] = 40
        xml += (f"<object><name>crack</name><bndbox><xmin>{x0 + 1}</xmin>"
                f"<ymin>{y0 + 1}</ymin><xmax>{x1 + 1}</xmax><ymax>{y1 + 1}</ymax>"
                "</bndbox></object>")
    cv2.imwrite(str(images / f"{image_id}.png"), img)
    (annotations / f"{image_id}.xml").write_text(xml + "</annotation>", encoding="utf-8")


@pytest.fixture
def dev_data(tmp_path):
    images, annotations, splits = (tmp_path / n for n in ("img", "ann", "splits"))
    for d in (images, annotations, splits):
        d.mkdir()
    _write_cell(images, annotations, "a", (5, 10, 20, 15))
    _write_cell(images, annotations, "b", (2, 2, 8, 8))
    (splits / "dev.txt").write_text("a\nb\n", encoding="utf-8")
    return tmp_path


def test_run_baselines_writes_outputs_and_log_rows(dev_data):
    log = dev_data / "log.csv"
    code = run_baselines.main([
        "--image-dir", str(dev_data / "img"), "--annotation-dir", str(dev_data / "ann"),
        "--ids-file", str(dev_data / "splits" / "dev.txt"), "--author", "tester",
        "--output-root", str(dev_data / "runs"), "--log", str(log),
    ])
    assert code == 0
    rows = _read_rows(log)
    assert {r["pipeline"] for r in rows} == {"baseline_trivial", "baseline_simple"}
    assert len(rows) == 2 * 8
    iou = {r["pipeline"]: float(r["metric_value_dev"]) for r in rows
           if r["metric_name"] == "box_union_iou_mean"}
    assert iou["baseline_trivial"] == pytest.approx(1.0)
    assert len(list((dev_data / "runs").iterdir())) == 2


def test_run_baselines_refuses_test_split(dev_data):
    with pytest.raises(SystemExit):
        run_baselines.main([
            "--image-dir", "x", "--annotation-dir", "y", "--author", "t",
            "--ids-file", str(dev_data / "splits" / "test.txt"),
        ])


def test_append_log_keeps_existing_rows(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text(",".join(run_baselines.LOG_COLUMNS) + "\n", encoding="utf-8")
    row = dict.fromkeys(run_baselines.LOG_COLUMNS, "x")
    run_baselines.append_log(log, [row])
    run_baselines.append_log(log, [row])
    assert len(_read_rows(log)) == 2


def test_sample_ids_is_seeded_sorted_and_sized():
    ids = [f"t{i:03d}" for i in range(50)]
    first = lock_test_split.sample_ids(ids, 10, seed=0)
    assert first == lock_test_split.sample_ids(ids, 10, seed=0)
    assert first == sorted(first) and len(set(first)) == 10
    with pytest.raises(ValueError):
        lock_test_split.sample_ids(ids, 51, seed=0)


def test_lock_test_split_writes_test_next_to_existing_dev(dev_data, capsys):
    test_img, test_ann = dev_data / "timg", dev_data / "tann"
    test_img.mkdir()
    test_ann.mkdir()
    for i in range(6):
        _write_cell(test_img, test_ann, f"t{i}")
    (test_img / "noxml.png").write_bytes(b"")          # image without XML is skipped

    lock_test_split.main([
        "--image-dir", str(test_img), "--annotation-dir", str(test_ann),
        "--size", "4", "--seed", "1", "--split-dir", str(dev_data / "splits"),
    ])

    test_txt = (dev_data / "splits" / "test.txt").read_bytes()
    assert len(test_txt.decode().split()) == 4
    assert (dev_data / "splits" / "dev.txt").read_text(encoding="utf-8") == "a\nb\n"
    assert hashlib.sha256(test_txt).hexdigest() in capsys.readouterr().out
    with pytest.raises(FileExistsError):               # locked: can't run twice
        lock_test_split.main([
            "--image-dir", str(test_img), "--annotation-dir", str(test_ann),
            "--size", "4", "--split-dir", str(dev_data / "splits"),
        ])
