"""Recompute the four main metrics from verified saved scores and write reports."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import fcntl
from pathlib import Path

import numpy as np

from evaluation.metrics import aupro, image_auprc_tvar, image_auroc_tvar, pixel_auroc
from scripts.experiment_state import atomic_json, load_verified_scores, validate_record, validate_run
from scripts.main_spec import DATASETS, PIPELINE, SEEDS

METRIC_KEYS = ("pixel_auroc", "image_auroc_tvar", "aupro", "image_auprc_tvar")
METRICS = [("image_auroc_tvar", "img-AUROC"), ("image_auprc_tvar", "img-AUPRC"),
           ("pixel_auroc", "pix-AUROC"), ("aupro", "AUPRO")]


def metrics_from_maps(maps, masks, labels):
    return {"pixel_auroc": pixel_auroc(masks, maps),
            "image_auroc_tvar": image_auroc_tvar(labels, maps, rho=0.01),
            "aupro": aupro(masks, maps),
            "image_auprc_tvar": image_auprc_tvar(labels, maps, rho=0.01)}


def load_inputs(root, datasets, allow_subset=False, *, check_scores=True, check_metrics=True):
    data = {}
    for ds in datasets:
        path = root / ds
        records = validate_run(path, check_scores=check_scores, check_metrics=check_metrics)
        categories = {record["category"] for record in records}
        if (not categories <= set(DATASETS[ds]) or
                (not allow_subset and categories != set(DATASETS[ds]))):
            raise ValueError(f"{path}: expected the full {ds} category set")
        data[ds] = records
    return data


def cells(records):
    """Mean and population std of seed-level category macro means."""
    per_seed = [{value["seed"]: value for value in record[PIPELINE]["per_seed"]}
                for record in records]
    values = np.asarray([[[record[seed][key] for key, _ in METRICS]
                          for record in per_seed] for seed in SEEDS])
    macro = values.mean(axis=1)
    return [(float(mean), float(std))
            for mean, std in zip(macro.mean(axis=0), macro.std(axis=0, ddof=0))]


def row(label, records):
    formatted = [f"{100 * mean:.1f}±{100 * std:.1f}" for mean, std in cells(records)]
    return f"| {label} | " + " | ".join(formatted) + " |"


def render_report(data, datasets, allow_subset=False):
    lines = []
    if allow_subset:
        lines += ["**Subset check: this report does not represent the full benchmark.**", ""]
    header = "| Dataset / category | " + " | ".join(label for _, label in METRICS) + " |"
    separator = "|---|" + "---:|" * len(METRICS)
    lines += [header, separator]
    for ds in datasets:
        records = data[ds]
        lines.append(row(f"{ds} ({len(records)} categories)", records))
    for ds in datasets:
        lines += ["", f"## {ds} — per category", "", header, separator]
        for record in sorted(data[ds], key=lambda r: r["category"]):
            lines.append(row(record["category"], [record]))
    return "\n".join(lines) + "\n"


def evaluate_results(root, *, allow_subset=False):
    root = Path(root).resolve()
    datasets = [dataset for dataset in DATASETS if (root / dataset).is_dir()]
    if not datasets:
        raise ValueError(f"No MVTec AD or VisA results at {root}; run an experiment first")
    with ExitStack() as locks:
        for dataset in datasets:
            lock = locks.enter_context((root / dataset / ".run.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = load_inputs(root, datasets, allow_subset,
                           check_scores=False, check_metrics=False)
        evaluated = deepcopy(data)
        for dataset in datasets:
            out = root / dataset
            for record in evaluated[dataset]:
                values = []
                for seed in SEEDS:
                    maps, masks, labels = load_verified_scores(
                        out, PIPELINE, record["category"], seed, n_test=record["n_test"])
                    metrics = metrics_from_maps(maps.astype(np.float32), masks, labels)
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError(f"{dataset}/{record['category']}/seed{seed}: nonfinite evaluation metrics")
                    values.append({"seed": seed, **metrics})
                record[PIPELINE]["per_seed"] = values
                record[PIPELINE]["per_seed_mean"] = {
                    key: float(np.mean([value[key] for value in values])) for key in METRIC_KEYS}
                record[PIPELINE]["per_seed_std"] = {
                    key: float(np.std([value[key] for value in values])) for key in METRIC_KEYS}
                validate_record(record, out, check_scores=False)
                print(f"[{dataset}/{record['category']}] evaluated {len(SEEDS)} seeds", flush=True)
        report = render_report(evaluated, datasets, allow_subset)
        for dataset in datasets:
            atomic_json(root / dataset / "tables/seeds.json", evaluated[dataset])
        _write_report(root, report)
    return evaluated


def _write_report(root, report):
    destination = root / "comparison.md"
    tmp = destination.with_suffix(".md.tmp")
    tmp.write_text(report)
    tmp.replace(destination)
    print(f"wrote {destination}")
