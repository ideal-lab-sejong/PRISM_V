from __future__ import annotations

import argparse
import csv
import fcntl
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from scripts.main_spec import DATASETS

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def download_tar(tar_path: Path) -> Path:
    """Download the official release when the local archive is missing."""
    if tar_path.is_file():
        return tar_path
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    partial = tar_path.with_name(tar_path.name + ".part")
    with tar_path.with_name("." + tar_path.name + ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if tar_path.is_file():
            return tar_path
        print(f"[prepare] downloading {VISA_URL} -> {tar_path}", flush=True)
        downloaded = 0
        next_report = 64 * 1024 * 1024
        with urllib.request.urlopen(VISA_URL, timeout=60) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    suffix = f" / {total / 1024**2:.0f} MiB" if total else " MiB"
                    print(f"[prepare] downloaded {downloaded / 1024**2:.0f}{suffix}", flush=True)
                    next_report = downloaded + 64 * 1024 * 1024
        if downloaded == 0 or (total and downloaded != total):
            raise RuntimeError(f"Incomplete VisA download ({downloaded}/{total} bytes); rerun to retry")
        with tarfile.open(partial) as archive:
            if not any(member.isfile() and member.name.endswith("split_csv/1cls.csv")
                       for member in archive):
                raise ValueError("Downloaded VisA archive does not contain split_csv/1cls.csv")
        partial.replace(tar_path)
        print(f"[prepare] downloaded {downloaded / 1024**2:.0f} MiB -> {tar_path}", flush=True)
    return tar_path


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
    Image.fromarray(arr).save(dst, format="PNG")


def reorganize(visa_root: Path, out: Path) -> None:
    csv_path = visa_root / "split_csv" / "1cls.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
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
    p.add_argument("--tar", default=str(THIS.parent.parent / "VisA_20220922.tar"),
                   help="path to VisA_20220922.tar (downloaded automatically if missing)")
    p.add_argument("--extract-dir", default=str(THIS.parent.parent / "data" / "visa_raw"),
                   help="scratch dir for the raw extracted tar")
    p.add_argument("--out", default=str(THIS.parent.parent / "data" / "visa"),
                   help="output root in the split-first layout (e.g. data/visa)")
    args = p.parse_args()

    tar_path = Path(args.tar)
    try:
        download_tar(tar_path)
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        sys.exit(f"VisA download failed: {exc}")
    visa_root = extract_tar(tar_path, Path(args.extract_dir))
    print(f"[prepare] VisA root: {visa_root}", flush=True)
    reorganize(visa_root, Path(args.out))
    got = sorted(p.name for p in (Path(args.out) / "images" / "train").iterdir()
                 if p.is_dir())
    if got != DATASETS["visa"]:
        sys.exit(f"category mismatch after conversion: {got}")
    print(f"[prepare] verified {len(got)} categories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
