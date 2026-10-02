from __future__ import annotations

import argparse
import csv
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from datasets.loader import VISA_CATEGORIES, detect_categories


def find_visa_root(base: Path) -> Path:
    """Locate the directory that contains split_csv/1cls.csv."""
    for cand in [base, *sorted(base.glob("**/split_csv/.."))]:
        if (cand / "split_csv" / "1cls.csv").exists():
            return cand
    hits = list(base.glob("**/1cls.csv"))
    if hits:
        return hits[0].parent.parent
    raise FileNotFoundError(f"split_csv/1cls.csv not found under {base}")


def extract_tar(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted_ok"
    if marker.exists():
        print(f"[prepare] tar already extracted at {dest}", flush=True)
        return find_visa_root(dest)
    print(f"[prepare] extracting {tar_path} -> {dest} ...", flush=True)
    t0 = time.time()
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    marker.write_text("ok\n")
    print(f"[prepare] extracted in {round(time.time()-t0,1)}s", flush=True)
    return find_visa_root(dest)


def save_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, format="PNG")


def save_mask_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as mk:
        arr = (np.array(mk.convert("L")) > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(dst, format="PNG")


def reorganize(visa_root: Path, out: Path) -> None:
    csv_path = visa_root / "split_csv" / "1cls.csv"
    rows = list(csv.DictReader(csv_path.open()))
    print(f"[prepare] 1cls.csv: {len(rows)} rows", flush=True)
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        cat = r["object"].strip()
        split = r["split"].strip()              # train | test
        label = r["label"].strip().lower()      # normal | anomaly
        img_rel = r["image"].strip()
        mask_rel = (r.get("mask") or "").strip()
        src_img = visa_root / img_rel
        stem = Path(img_rel).stem
        if split == "train":
            sub = "images/train/%s/good" % cat
        elif label == "normal":
            sub = "images/test/%s/good" % cat
        else:
            sub = "images/test/%s/bad" % cat
        save_png(src_img, out / sub / f"{stem}.png")
        if split == "test" and label != "normal" and mask_rel:
            save_mask_png(visa_root / mask_rel,
                          out / "masks" / "test" / cat / "bad"
                          / f"{stem}_mask.png")
        key = f"{cat}/{split}/{label}"
        counts[key] = counts.get(key, 0) + 1
        if (i + 1) % 1000 == 0:
            print(f"[prepare] {i+1}/{len(rows)} "
                  f"({round(time.time()-t0,1)}s)", flush=True)
    print(f"[prepare] done in {round(time.time()-t0,1)}s", flush=True)
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tar", required=True, help="path to VisA_20220922.tar")
    p.add_argument("--extract-dir", required=True,
                   help="scratch dir for the raw extracted tar")
    p.add_argument("--out", required=True,
                   help="output root in the split-first layout (e.g. data/visa)")
    args = p.parse_args()

    tar_path = Path(args.tar)
    if not tar_path.exists():
        sys.exit(f"tar not found: {tar_path}")
    visa_root = extract_tar(tar_path, Path(args.extract_dir))
    print(f"[prepare] VisA root: {visa_root}", flush=True)
    reorganize(visa_root, Path(args.out))
    got = detect_categories(args.out)
    if got != VISA_CATEGORIES:
        sys.exit(f"category mismatch after conversion: {got}")
    print(f"[prepare] verified {len(got)} categories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
