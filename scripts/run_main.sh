#!/usr/bin/env bash
set -u
DS=${1:?usage: run_main.sh <mvtec|visa>}
PY=${PYTHON:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN=$ROOT/scripts/run_prism_v.py
OUT=$ROOT/results/main
LOG=$OUT/master_${DS}.log
SEEDS="1 2 3 4 5"
mkdir -p "$OUT"

if [ "$DS" = "visa" ]; then
  DS_ARGS="--root $ROOT/data/visa"
else
  DS_ARGS=""
fi

ts() { date +%H:%M:%S; }
echo "=== MAIN ($DS) START $(ts) ===" > "$LOG"

if [ -f "$OUT/$DS/tables/seeds.json" ]; then
  echo "=== SKIP main/$DS (seeds.json exists) $(ts) ===" | tee -a "$LOG"
else
  echo "=== START main/$DS  $(ts) ===" | tee -a "$LOG"
  $PY "$RUN" $DS_ARGS --out "$OUT/$DS" --seeds $SEEDS \
      --pipelines random_cnn_padim >>"$LOG" 2>&1
  echo "=== DONE  main/$DS rc=$? $(ts) ===" | tee -a "$LOG"
fi

echo "=== G SIDECARS main/$DS  $(ts) ===" | tee -a "$LOG"
$PY "$ROOT/scripts/compute_g.py"     "$OUT/$DS/scores/random_cnn_padim" --seeds $SEEDS >>"$LOG" 2>&1

echo "=== MAIN ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
