from __future__ import annotations

from pathlib import Path

import numpy as np


def scores_root(tables_dir) -> Path:
    return Path(tables_dir).parent / "scores"


def score_path(tables_dir, pipeline: str, category: str,
               seed: int | None = None) -> Path:
    d = scores_root(tables_dir) / pipeline
    d.mkdir(parents=True, exist_ok=True)
    name = f"{category}.npz" if seed is None else f"{category}_seed{seed}.npz"
    return d / name


def save_scores(path, *, labels, maps=None, masks=None, image_scores=None, **meta) -> None:
    arrs: dict[str, np.ndarray] = {
        "labels": np.asarray(labels).astype(np.int64).reshape(-1),
    }
    if maps is not None:
        arrs["maps"] = np.asarray(maps).astype(np.float16)
    if masks is not None:
        arrs["masks"] = np.asarray(masks).astype(np.uint8)
    if image_scores is not None:
        arrs["image_scores"] = np.asarray(image_scores).astype(np.float32).reshape(-1)
    for k, v in meta.items():
        arrs[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(str(path), **arrs)


def load_scores(path) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
