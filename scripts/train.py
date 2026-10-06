from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import fcntl
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE = Path(__file__).resolve().parents[1]

from evaluation.evaluate import METRIC_KEYS, metrics_from_maps
from evaluation.score_io import save_scores, score_path
from models.prism_head import PrismHead
from datasets.loader import CategoryDataset, collate
from models.prism_v import build_random_cnn_extractor
from scripts.experiment_state import (FIT_VERSION, execute_categories)
from scripts.main_spec import DATASETS, MAIN_CONFIG, PIPELINE, SEEDS

def make_args(dataset, *, root=None, results_root=None, batch_size=16,
              num_workers=4, device=None):
    args = Namespace(root=str(Path(root or PACKAGE / "data" / dataset).resolve()),
                     out=str(Path(results_root or PACKAGE / "results").resolve() / dataset),
                     batch_size=batch_size, num_workers=num_workers,
                     device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
                     categories=list(DATASETS[dataset]))
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers must be nonnegative")
    for name, value in deepcopy(MAIN_CONFIG).items():
        setattr(args, name, value)
    args.seeds = list(SEEDS)
    args.pipelines = [PIPELINE]
    return args


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


def fit_one(pipeline, seed, args, train_loader, test_loader):
    """Fit + score one (pipeline, seed). Returns (maps, masks, labels)."""
    if pipeline == PIPELINE:
        ext = build_random_cnn_extractor(
            seed=seed,
            output_layers=tuple(args.cnn_layers),
            channels=tuple(args.cnn_channels))
        head = PrismHead(ext, layer_names=list(args.cnn_layers),
                     image_size=args.image_size, device=args.device,
                     cov_dtype="float32")
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
        pipe_seeds = list(args.seeds)

        for seed in pipe_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            used_seeds.add(seed)
            try:
                maps, masks, labels = fit_one(
                    pipe, seed, args, train_loader, test_loader)
                maps = maps.astype(np.float16).astype(np.float32)
                if not np.isfinite(maps).all():
                    raise ValueError("Nonfinite anomaly maps after float16 persistence")
                metrics = metrics_from_maps(maps, masks, labels)
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise ValueError("Nonfinite evaluation metrics")
                save_scores(score_path(scores_tables_dir, pipe, category, seed),
                            labels=labels, maps=maps, masks=masks,
                            fit_version=FIT_VERSION,
                            test_paths=[str(p) for p, _, _, _ in test_ds.samples])
                per_seed_metrics.append({"seed": seed, **metrics})
                del maps
            except Exception as exc:
                per_seed_metrics.append({"seed": seed, "error": repr(exc)})
                torch.cuda.empty_cache()
                continue

        ok = [m for m in per_seed_metrics if "error" not in m]
        if len(ok) != len(pipe_seeds):
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


def run_dataset(args):
    device_id = args.device.replace(":", "_").replace("/", "_")
    lock_path = Path("/tmp") / f"prism_v_device_{os.getuid()}_{device_id}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_categories(args, CategoryDataset, run_category, "run.log")
