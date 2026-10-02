#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-python}
cd "$ROOT"

bash scripts/run_main.sh mvtec
bash scripts/run_main.sh visa

bash scripts/run_ablation.sh all mvtec
bash scripts/run_ablation.sh all visa

$PY scripts/aggregate_ablation.py

echo "== ALL DONE — tables in results/ablation/<name>/comparison.md"
