"""The three commands for the fixed PRISM-V main experiment."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))


def _fit_options(parser):
    parser.add_argument("--root", help="Dataset directory (default: data/<dataset> in this repo)")
    parser.add_argument("--results-root", default=os.environ.get("PRISM_RESULTS_ROOT", str(PACKAGE / "results")))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(prog="run", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mvtec = commands.add_parser("mvtec", help="Run the MVTec AD main experiment")
    mvtec_commands = mvtec.add_subparsers(dest="mvtec_command", required=True)
    _fit_options(mvtec_commands.add_parser("ad", help="Fit and score all 15 categories"))
    _fit_options(commands.add_parser("visa", help="Run the VisA main experiment"))
    evaluation = commands.add_parser("evaluation", help="Recompute main metrics from saved scores")
    evaluation.add_argument("--results-root", default=os.environ.get("PRISM_RESULTS_ROOT", str(PACKAGE / "results")))
    args = parser.parse_args(arguments)
    if args.command != "evaluation" and (args.batch_size < 1 or args.num_workers < 0):
        parser.error("--batch-size must be positive and --num-workers must be nonnegative")
    return args


def _prepare_dataset(dataset, root=None):
    from scripts.main_spec import DATASETS

    root = Path(root).resolve() if root else PACKAGE / "data" / dataset
    archive_name = {"mvtec": "mvtec_anomaly_detection.tar.xz",
                    "visa": "VisA_20220922.tar"}[dataset]
    marker = root / ".preparing"

    def ready():
        train = root / "images" / "train"
        test = root / "images" / "test"
        masks = root / "masks" / "test"
        expected = set(DATASETS[dataset])
        return (not marker.exists() and train.is_dir()
                and {p.name for p in train.iterdir() if p.is_dir()} == expected
                and all(any((train / cat / "good").glob("*.png"))
                        and any((test / cat / "good").glob("*.png"))
                        and (masks / cat).is_dir() for cat in expected))

    if ready():
        return root
    candidates = [PACKAGE / archive_name, PACKAGE / "data" / archive_name]
    archive = next((p for p in candidates if p.is_file()), None)
    if archive is None and dataset == "visa":
        archive = PACKAGE / archive_name
    if archive is None:
        raise FileNotFoundError(
            f"Dataset is not prepared at {root}. "
            f"Place {archive_name} in {PACKAGE} and rerun this command.")
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready():
            return root
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(archive) + "\n")
        command = [sys.executable, str(PACKAGE / "datasets" / f"dataset_prepare_{dataset}.py"),
                   "--tar", str(archive), "--out", str(root)]
        if dataset == "visa":
            command += ["--extract-dir", str(root.parent / "visa_raw")]
        print(f"[prepare] {dataset}: {archive} -> {root}", flush=True)
        subprocess.run(command, check=True)
        marker.unlink()
        if not ready():
            raise ValueError(f"Dataset preparation did not produce the expected layout at {root}")
    return root


def main():
    args = parse_args()
    try:
        if args.command == "evaluation":
            from evaluation.evaluate import evaluate_results
            evaluate_results(Path(args.results_root).resolve())
        else:
            from scripts.train import make_args, run_dataset
            dataset = "mvtec" if args.command == "mvtec" else "visa"
            root = _prepare_dataset(dataset, args.root)
            run_dataset(make_args(dataset, root=root, results_root=args.results_root,
                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                  device=args.device))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
    except (ValueError, FileNotFoundError, RuntimeError, BlockingIOError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
