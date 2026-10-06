"""Resume and validate PRISM-V main results using saved tables and scores."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts.main_spec import MAIN_CONFIG, PIPELINE, SEEDS

FIT_VERSION = "single-pass-v2"
REQUIRED_METRICS = ("pixel_auroc", "aupro", "image_auroc_tvar", "image_auprc_tvar")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def prepare_run(args, dataset_cls):
    """Collect expected dataset counts and read previous category results."""
    for key in ("categories", "seeds", "pipelines"):
        values = getattr(args, key)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique")
    inventory = {}
    for category in args.categories:
        counts = {}
        for split in ("train", "test"):
            ds = dataset_cls(args.root, category, split, image_size=args.image_size,
                             resize=args.resize_size)
            counts["n_" + split] = len(ds)
            if split == "test":
                counts["test_paths"] = [str(p) for p, _, _, _ in ds.samples]
                counts["test_labels"] = [label for _, _, label, _ in ds.samples]
        inventory[category] = counts
    out = Path(args.out)
    records_fp = out / "tables/seeds.json"
    records = json.loads(records_fp.read_text()) if records_fp.exists() else []
    if len({r["category"] for r in records}) != len(records):
        raise ValueError("Duplicate checkpoint categories")
    return inventory, {r["category"]: r for r in records}


def load_verified_scores(out, pipeline, category, seed, *, n_test, inventory=None):
    fp = Path(out) / "scores" / pipeline / f"{category}_seed{seed}.npz"
    with np.load(fp, allow_pickle=False) as saved:
        shape = (n_test, MAIN_CONFIG["image_size"], MAIN_CONFIG["image_size"])
        maps, masks, labels = saved["maps"], saved["masks"], saved["labels"]
        if maps.shape != shape or masks.shape != shape or labels.shape != (shape[0],):
            raise ValueError(f"{fp}: wrong array shapes")
        if maps.dtype != np.float16 or not np.isfinite(maps).all():
            raise ValueError(f"{fp}: invalid saved maps")
        if not np.isin(masks, (0, 1)).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{fp}: invalid masks or labels")
        if str(saved["meta_fit_version"].item()) != FIT_VERSION:
            raise ValueError(f"{fp}: incompatible fitting version")
        paths = saved["meta_test_paths"].tolist()
        if len(paths) != n_test:
            raise ValueError(f"{fp}: wrong test path count")
        if inventory is not None and (labels.tolist() != inventory["test_labels"] or
                                      paths != inventory["test_paths"]):
            raise ValueError(f"{fp}: wrong test ordering")
    return maps, masks, labels


def validate_record(record, out, *, inventory=None, check_scores=True, check_metrics=True):
    cat = record["category"]
    for key in ("n_train", "n_test"):
        count = record.get(key)
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{cat}: invalid {key}")
        if inventory is not None and count != inventory[key]:
            raise ValueError(f"{cat}: wrong {key}")
    for pipeline in (PIPELINE,):
        values = record[pipeline]["per_seed"]
        seeds = SEEDS
        if len(values) != len(seeds) or {v["seed"] for v in values} != set(seeds):
            raise ValueError(f"{cat}/{pipeline}: missing or duplicate seeds")
        for value in values:
            if "error" in value:
                raise ValueError(f"{cat}/{pipeline}: {value['error']}")
            if check_metrics:
                for metric in REQUIRED_METRICS:
                    if not math.isfinite(value[metric]) or not 0 <= value[metric] <= 1 + 1e-9:
                        raise ValueError(f"{cat}/{pipeline}: invalid {metric}")
            if check_scores:
                load_verified_scores(out, pipeline, cat, value["seed"],
                                     n_test=record["n_test"], inventory=inventory)
        if not check_metrics:
            continue
        for metric in REQUIRED_METRICS:
            v = [r[metric] for r in values]
            for key, expected in (("per_seed_mean", np.mean(v)), ("per_seed_std", np.std(v))):
                if not math.isclose(record[pipeline][key][metric], expected, abs_tol=1e-10):
                    raise ValueError(f"{cat}/{pipeline}: inconsistent {key}/{metric}")


def validate_run(out, *, categories=None, check_scores=True, check_metrics=True):
    out = Path(out)
    records = json.loads((out / "tables/seeds.json").read_text())
    cats = [r["category"] for r in records]
    if (not cats or len(set(cats)) != len(cats) or
            (categories is not None and set(cats) != set(categories))):
        raise ValueError(f"{out}: missing or duplicate categories")
    for record in records:
        validate_record(record, out, check_scores=check_scores, check_metrics=check_metrics)
    return records


def execute_categories(args, dataset_cls, run_category, log_name):
    """Checkpoint atomically; retry incomplete categories; never average failures."""
    import fcntl
    import time

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory, previous = prepare_run(args, dataset_cls)
        records = []
        errors = []
        (out / "logs").mkdir(exist_ok=True)
        with (out / "logs" / log_name).open("a") as log:
            log.write(f"Args: {vars(args)}\nfit={FIT_VERSION}\n")
            for category in args.categories:
                started = time.time()
                rec = previous.get(category)
                try:
                    if rec is None:
                        raise ValueError("No checkpoint")
                    validate_record(rec, out, inventory=inventory[category])
                    print(f"[{category}] SKIP verified complete", flush=True)
                except Exception:
                    try:
                        rec = run_category(args, category)
                        validate_record(rec, out, inventory=inventory[category])
                    except Exception as exc:
                        errors.append(f"{category}: {exc!r}")
                        print(f"[{category}] FAILED: {exc!r}", flush=True)
                        if rec is None:
                            rec = {"category": category, "error": repr(exc)}
                records.append(rec)
                processed = {r["category"] for r in records}
                remaining = [previous[c] for c in args.categories
                             if c not in processed and c in previous]
                atomic_json(out / "tables/seeds.json", records + remaining)
                log.write(f"[{category}] {time.time() - started:.1f}s\n")
                log.flush()
            try:
                if errors:
                    raise ValueError("; ".join(errors))
                validate_run(out, categories=args.categories, check_scores=False)
            except Exception as exc:
                raise RuntimeError(f"Incomplete run at {out}: {exc}") from exc
    print(f"COMPLETE: {out}", flush=True)
    return records
