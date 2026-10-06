from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from scripts.main_spec import DATASETS

# The official archive lays every image out as <category>/<kind>/<defect>/<file>.
RE_TRAIN = re.compile(r"^(?:\./)?([^/]+)/train/good/([^/]+\.png)$")
RE_TEST = re.compile(r"^(?:\./)?([^/]+)/test/([^/]+)/([^/]+\.png)$")
RE_MASK = re.compile(r"^(?:\./)?([^/]+)/ground_truth/([^/]+)/([^/]+\.png)$")


def destination(name: str) -> str | None:
    """Map an official-layout path to its split-first path, or None if it is
    not one of the dataset's images (readme.txt / license.txt land here)."""
    m = RE_TRAIN.match(name)
    if m:
        return f"images/train/{m[1]}/good/{m[2]}"
    m = RE_TEST.match(name)
    if m:
        return f"images/test/{m[1]}/{m[2]}/{m[3]}"
    m = RE_MASK.match(name)
    if m:
        return f"masks/test/{m[1]}/{m[2]}/{m[3]}"
    return None


def from_tar(tar_path: Path, out: Path) -> tuple[int, int]:
    """Stream the archive, writing image members straight to their new home.

    Bytes are copied verbatim — no decode/re-encode — so every file is
    identical to the one inside the archive.
    """
    written = skipped = 0
    t0 = time.time()
    with tarfile.open(tar_path, "r:*") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            rel = destination(mem.name)
            if rel is None:
                skipped += 1
                continue
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(mem)
            with open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            written += 1
            if written % 1000 == 0:
                print(f"[prepare] {written} images "
                      f"({round(time.time()-t0,1)}s)", flush=True)
    print(f"[prepare] wrote {written} images in "
          f"{round(time.time()-t0,1)}s", flush=True)
    return written, skipped


def from_dir(src_root: Path, out: Path) -> tuple[int, int]:
    """Same mapping from an already-extracted root.

    Files are hard-linked when the destination is on the same filesystem —
    instant and free — and copied otherwise.
    """
    written = skipped = 0
    t0 = time.time()
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = destination(path.relative_to(src_root).as_posix())
        if rel is None:
            skipped += 1
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(path, dst)
        except OSError:
            shutil.copy2(path, dst)
        written += 1
    print(f"[prepare] linked {written} images in "
          f"{round(time.time()-t0,1)}s", flush=True)
    return written, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert the official MVTec AD release into the "
                    "split-first layout, dropping readme.txt / license.txt.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tar", help="path to mvtec_anomaly_detection.tar.xz")
    g.add_argument("--src", help="path to an already-extracted official root "
                                 "(the one holding bottle/, cable/, ...)")
    p.add_argument("--out", required=True,
                   help="output root in the split-first layout (e.g. data/mvtec)")
    args = p.parse_args()

    out = Path(args.out)
    if args.tar:
        tar_path = Path(args.tar)
        if not tar_path.exists():
            sys.exit(f"tar not found: {tar_path}")
        print(f"[prepare] reading {tar_path} -> {out}", flush=True)
        written, skipped = from_tar(tar_path, out)
    else:
        src_root = Path(args.src)
        if not src_root.is_dir():
            sys.exit(f"source root not found: {src_root}")
        print(f"[prepare] reading {src_root} -> {out}", flush=True)
        written, skipped = from_dir(src_root, out)

    print(f"[prepare] skipped {skipped} non-image files "
          f"(readme.txt / license.txt)", flush=True)

    got = sorted(p.name for p in (out / "images" / "train").iterdir()
                 if p.is_dir())
    if got != DATASETS["mvtec"]:
        sys.exit(f"category mismatch after conversion: {got}")
    if written != 6612:
        sys.exit(f"expected 6612 images, wrote {written}")
    print(f"[prepare] verified {len(got)} categories -> {out}", flush=True)


if __name__ == "__main__":
    main()
