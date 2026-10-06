from __future__ import annotations

from pathlib import Path

import numpy as np


def scores_root(tables_dir) -> Path:
    return Path(tables_dir).parent / "scores"


def score_path(tables_dir, pipeline: str, category: str,
               seed: int) -> Path:
    d = scores_root(tables_dir) / pipeline
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{category}_seed{seed}.npz"


def save_scores(path, *, labels, maps, masks, **meta) -> None:
    arrs: dict[str, np.ndarray] = {
        "labels": np.asarray(labels).astype(np.int64).reshape(-1),
    }
    arrs["maps"] = np.asarray(maps).astype(np.float16)
    arrs["masks"] = np.asarray(masks).astype(np.uint8)
    for k, v in meta.items():
        arrs[f"meta_{k}"] = np.asarray(v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrs)
    tmp.replace(path)
