"""Lock the held-out test sample once (Phase 2).

Draws a fixed-size, seeded random sample of image IDs from the external PVEL-AD test
set (only IDs that have both an image and an XML), writes ``data/splits/test.txt``
with ``src.data.save_split`` next to the already-locked ``dev.txt``, and prints the
SHA-256 to record in ``data/README.md``.

It only lists file names: it never opens a test image. Run it exactly once:

    python -m src.lock_test_split --image-dir <test JPEGImages> \\
        --annotation-dir <test XMLs> --size 1000 --seed 0

The size must keep the final evaluation under 10 minutes on free Colab CPU
(spec 1, feasibility). Baselines take milliseconds per image, but Pipeline C will be
slower; 1000 images leaves room for about 0.5 s per image.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.data import save_split
from src.evaluate import IMAGE_SUFFIXES, read_ids_file


def available_ids(image_dir: str | Path, annotation_dir: str | Path) -> list[str]:
    """Sorted IDs that have both an image file and an XML annotation."""
    images = {p.stem for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    xmls = {p.stem for p in Path(annotation_dir).iterdir() if p.suffix.lower() == ".xml"}
    return sorted(images & xmls)


def sample_ids(ids: Sequence[str], size: int, seed: int) -> list[str]:
    """Seeded sample of ``size`` IDs without replacement, returned sorted."""
    if not 0 < size <= len(ids):
        raise ValueError(f"size must be between 1 and {len(ids)}")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(ids), size=size, replace=False)
    return sorted(ids[i] for i in picked)


def main(argv: Sequence[str] | None = None) -> int:
    """Sample the test IDs, lock test.txt next to dev.txt and print its SHA-256."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, help="external test image folder")
    parser.add_argument("--annotation-dir", required=True, help="external test XML folder")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-dir", default="data/splits")
    args = parser.parse_args(argv)

    dev_ids = read_ids_file(Path(args.split_dir) / "dev.txt")
    test_ids = sample_ids(available_ids(args.image_dir, args.annotation_dir),
                          args.size, args.seed)
    digest = save_split(dev_ids, test_ids, args.split_dir)
    print(f"Locked {len(test_ids)} test IDs (seed {args.seed}).")
    print(f"test.txt SHA-256: {digest}  <- record this in data/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
