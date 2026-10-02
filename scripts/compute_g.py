from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from evaluation.metrics import g_img, g_pix
from evaluation.score_io import load_scores


def _split_stem(stem: str) -> tuple[str, int | None]:
    if "_seed" in stem:
        cat, seed = stem.rsplit("_seed", 1)
        return cat, int(seed)
    return stem, None


def process_dir(scores_dir: Path, seeds: list[int] | None) -> dict:
    files = sorted(scores_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz files in {scores_dir}")

    per_file = []
    for fp in files:
        cat, seed = _split_stem(fp.stem)
        if seeds is not None and seed is not None and seed not in seeds:
            continue
        z = load_scores(fp)
        rec = {
            "category": cat,
            "seed": seed,
            "g_img": g_img(z["labels"], z["maps"]),
            "g_pix": g_pix(z["masks"], z["maps"]),
        }
        per_file.append(rec)
        print(f"  {fp.stem:28s} g_img={rec['g_img']:7.3f}  "
              f"g_pix={rec['g_pix']:7.3f}", flush=True)

    cats = sorted({r["category"] for r in per_file})
    per_category = {}
    for cat in cats:
        rs = [r for r in per_file if r["category"] == cat]
        per_category[cat] = {
            "g_img": float(np.mean([r["g_img"] for r in rs])),
            "g_pix": float(np.mean([r["g_pix"] for r in rs])),
            "n_seeds": len(rs),
        }
    dataset_mean = {
        "g_img": float(np.mean([v["g_img"] for v in per_category.values()])),
        "g_pix": float(np.mean([v["g_pix"] for v in per_category.values()])),
    }
    return {"per_file": per_file, "per_category": per_category,
            "dataset_mean": dataset_mean}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scores_dirs", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Restrict to these seeds (seed-agnostic files always "
                        "kept). Default: all found.")
    p.add_argument("--out-dir", default=None,
                   help="Directory for g_metrics.json (default: each "
                        "scores_dir itself).")
    args = p.parse_args()

    for d in args.scores_dirs:
        scores_dir = Path(d)
        print(f"== {scores_dir} ==", flush=True)
        res = process_dir(scores_dir, args.seeds)
        out = (Path(args.out_dir) if args.out_dir else scores_dir) \
            / "g_metrics.json"
        out.write_text(json.dumps(res, indent=2))
        dm = res["dataset_mean"]
        print(f"  DATASET MEAN  g_img={dm['g_img']:.3f}  "
              f"g_pix={dm['g_pix']:.3f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
