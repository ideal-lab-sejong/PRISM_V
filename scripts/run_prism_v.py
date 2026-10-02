from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from evaluation.metrics import (aupro, image_auprc_tvar, image_auroc_tvar,
                                pixel_auroc)
from evaluation.score_io import save_scores, score_path
from models.padim_head import PaDiM
from datasets.loader import CategoryDataset, collate, detect_categories
from models.prism_v import build_random_cnn_extractor
from models.raw_patches import build_raw_patch_extractor

SEEDS = [1, 2, 3, 4, 5]

PIPELINES = [
    "random_cnn_padim",
]


EXTRA_PIPELINES = ["raw_mahalanobis"]

# Pipelines without weights: the result cannot depend on a seed, so they run
# once and their scores are stored seed-agnostically (<category>.npz).
DETERMINISTIC_PIPELINES = {"raw_mahalanobis"}

METRIC_KEYS = ("pixel_auroc", "image_auroc_tvar", "aupro", "image_auprc_tvar")

# Fixed by the adopted protocol — no experiment in this repository varies them.
CNN_DEPTH = 4
CNN_KERNEL = (3, 3, 3, 1, 1)
TAP_LAYERS = ("stage2", "stage3", "stage4", "stage5")
COV_RIDGE = 0.01

_COV_POOL = {"none": 0, "r1": 1, "r2": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(THIS.parent.parent / "data" / "mvtec"))
    p.add_argument("--out",
                   default=str(THIS.parent.parent / "results" / "seeds"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Categories to run (default: auto-detected from "
                        "--root, e.g. the 15 MVTec AD or 12 VisA classes).")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--cnn-norm", default="instance",
                   choices=["none", "instance"])
    p.add_argument("--cnn-downsample", default="blur",
                   choices=["maxpool", "blur"],
                   help="blur = blur-only anti-aliased downsample WITHOUT a "
                        "max nonlinearity (main); maxpool = plain stride-2 "
                        "max-pool (building-block ablation arms).")
    p.add_argument("--cnn-blur-kernel", default="bin7",
                   choices=["tri3", "bin5", "bin7", "bin9"],
                   help="BlurPool low-pass width (with --cnn-downsample "
                        "blur): tri3 [1,2,1], bin5 [1,4,6,4,1], "
                        "bin7 [1,6,15,20,15,6,1] (main; widest tested that "
                        "monotonically improved both datasets), or bin9 "
                        "[1,8,28,56,70,56,28,8,1] = stronger anti-aliasing.")
    p.add_argument("--cnn-channels", type=int, nargs="+",
                   default=[128, 128, 128, 128, 128],
                   help="Per-stage channel widths (five values). Default "
                        "128/128/128/128/128 = the adopted fixed-128 isotropic "
                        "encoder.")
    p.add_argument("--cnn-activation", default="leaky_relu",
                   choices=["relu", "leaky_relu", "tanh", "sigmoid"],
                   help="Nonlinearity inside each conv block (adopted: leaky_relu).")
    p.add_argument("--cnn-negative-slope", type=float, default=0.01,
                   help="LeakyReLU negative slope (used when --cnn-activation "
                        "leaky_relu). 1.0 == identity, i.e. removes the "
                        "activation (component-analysis 'w/o Act.' arm).")
    p.add_argument("--padim-cov-pool", default="r1",
                   choices=["none", "r1", "r2"],
                   help="Spatially pool the PaDiM per-position covariance to "
                        "fight d>>N sample-starvation (the mean stays "
                        "per-position): none, r1 (3x3; adopted main), r2 "
                        "(5x5). random_cnn_padim only.")
    p.add_argument("--pipelines", nargs="+", default=PIPELINES,
                   choices=PIPELINES + EXTRA_PIPELINES)
    return p.parse_args()


def _loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate,
                      pin_memory=True)


def make_loaders(root, category, image_size, batch_size, num_workers,
                 resize=None):
    train_ds = CategoryDataset(root, category, "train", image_size=image_size,
                             resize=resize)
    test_ds = CategoryDataset(root, category, "test", image_size=image_size,
                            resize=resize)
    return (train_ds, _loader(train_ds, batch_size, num_workers),
            test_ds, _loader(test_ds, batch_size, num_workers))


def collect_scores(model, loader):
    s, m, l = [], [], []
    for batch in loader:
        s.append(model.score(batch["image"]).cpu().numpy())
        m.append(batch["mask"][:, 0].numpy())
        l.append(batch["label"].numpy())
    return (np.concatenate(s).astype(np.float32),
            np.concatenate(m).astype(np.uint8),
            np.concatenate(l))


def metrics_from_maps(maps, masks, labels) -> dict:
    return {
        "pixel_auroc": pixel_auroc(masks, maps),
        "image_auroc_tvar": image_auroc_tvar(labels, maps, rho=0.01),
        "aupro": aupro(masks, maps),
        "image_auprc_tvar": image_auprc_tvar(labels, maps, rho=0.01),
    }


def fit_one(pipeline, seed, args, train_loader, test_loader):
    """Fit + score one (pipeline, seed). Returns (maps, masks, labels)."""
    if pipeline == "random_cnn_padim":
        ext = build_random_cnn_extractor(
            seed=seed, depth=CNN_DEPTH,
            output_layers=TAP_LAYERS,
            norm=args.cnn_norm, downsample=args.cnn_downsample,
            blur_kernel=args.cnn_blur_kernel,
            kernel_size=CNN_KERNEL, channels=tuple(args.cnn_channels),
            activation=args.cnn_activation,
            negative_slope=args.cnn_negative_slope)
        head = PaDiM(ext, layer_names=list(TAP_LAYERS),
                     image_size=args.image_size, device=args.device,
                     cov_dtype="float32",
                     cov_pool_radius=_COV_POOL[args.padim_cov_pool],
                     cov_ridge=COV_RIDGE)
    elif pipeline == "raw_mahalanobis":
        ext = build_raw_patch_extractor(factor=2)
        head = PaDiM(ext, layer_names=["raw"],
                     image_size=args.image_size, device=args.device,
                     cov_dtype="float32",
                     cov_pool_radius=_COV_POOL[args.padim_cov_pool],
                     cov_ridge=COV_RIDGE)
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")

    head.fit(train_loader)
    out = collect_scores(head, test_loader)
    del head, ext
    torch.cuda.empty_cache()
    return out


def run_category(args, category) -> dict:
    train_ds, train_loader, test_ds, test_loader = make_loaders(
        args.root, category, args.image_size, args.batch_size,
        args.num_workers, resize=args.resize_size)

    rec = {
        "category": category,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "seeds": [],
    }

    scores_tables_dir = Path(args.out) / "tables"
    used_seeds: set[int] = set()

    for pipe in args.pipelines:
        t0 = time.time()
        per_seed_metrics = []
        pipe_seeds = ([None] if pipe in DETERMINISTIC_PIPELINES
                      else list(args.seeds))

        for seed in pipe_seeds:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                used_seeds.add(seed)
            try:
                maps, masks, labels = fit_one(
                    pipe, seed, args, train_loader, test_loader)
            except Exception as exc:
                per_seed_metrics.append({"seed": seed, "error": repr(exc)})
                torch.cuda.empty_cache()
                continue

            save_scores(score_path(scores_tables_dir, pipe, category, seed),
                        labels=labels, maps=maps, masks=masks)
            per_seed_metrics.append(
                {"seed": seed, **metrics_from_maps(maps, masks, labels)})
            del maps

        ok = [m for m in per_seed_metrics if "error" not in m]
        if not ok:
            rec[pipe] = {"per_seed": per_seed_metrics,
                         "elapsed_sec": round(time.time() - t0, 1)}
            continue

        per_seed_mean = {k: float(np.mean([m[k] for m in ok]))
                         for k in METRIC_KEYS}
        per_seed_std = {k: float(np.std([m[k] for m in ok]))
                        for k in METRIC_KEYS}

        rec[pipe] = {
            "per_seed": per_seed_metrics,
            "per_seed_mean": per_seed_mean,
            "per_seed_std": per_seed_std,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"[{category}][{pipe}] n={len(ok)}  "
            f"per-seed pix={per_seed_mean['pixel_auroc']:.4f}"
            f"+/-{per_seed_std['pixel_auroc']:.4f}  "
            f"aupro={per_seed_mean['aupro']:.4f}  "
            f"img-tvar={per_seed_mean['image_auroc_tvar']:.4f}",
            flush=True,
        )

    rec["seeds"] = sorted(used_seeds)
    return rec


def main() -> None:
    args = parse_args()
    if not args.categories:
        args.categories = detect_categories(args.root)
        print(f"categories (auto-detected): {args.categories}", flush=True)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    log_path = out / "logs" / "run_prism_v.log"
    json_path = out / "tables" / "seeds.json"

    results = []
    with open(log_path, "w") as logf:
        logf.write(f"Args: {vars(args)}\n")
        logf.flush()
        for cat in args.categories:
            t0 = time.time()
            try:
                rec = run_category(args, cat)
            except Exception as exc:
                msg = f"[{cat}] FAILED: {exc!r}"
                print(msg, flush=True)
                logf.write(msg + "\n")
                logf.flush()
                continue
            results.append(rec)
            json_path.write_text(json.dumps(results, indent=2))
            logf.write(f"[{cat}] done in {time.time() - t0:.1f}s\n")
            logf.flush()

    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
