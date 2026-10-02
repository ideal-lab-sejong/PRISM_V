#!/usr/bin/env bash
set -u
NAME=${1:?usage: run_ablation.sh <component|building_block|blur_kernel|cov_pool|hidden_dim|activation_fn|all> [mvtec|visa]}
DS=${2:-}
PY=${PYTHON:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN=$ROOT/scripts/run_prism_v.py
ABL=$ROOT/results/ablation
SEEDS="1 2 3 4 5"

ts() { date +%H:%M:%S; }

need_ds() {
  [ -n "$DS" ] || { echo "error: '$NAME' needs a dataset arg (mvtec|visa)"; exit 1; }
  if [ "$DS" = "visa" ]; then
    DS_ARGS="--root $ROOT/data/visa"
  else
    DS_ARGS=""
  fi
}

run_one() {
  local out=$1 arm=$2 arm_seeds=$3; shift 3
  if [ -f "$out/$arm/$DS/tables/seeds.json" ]; then
    echo "=== SKIP ${arm}/${DS} (seeds.json exists) $(ts) ===" | tee -a "$LOG"
    return
  fi
  echo "=== START ${arm}/${DS}  $(ts) ===" | tee -a "$LOG"
  local seed_args=""
  [ -n "$arm_seeds" ] && seed_args="--seeds $arm_seeds"
  $PY "$RUN" $DS_ARGS --out "$out/$arm/$DS" $seed_args "$@" \
      >>"$LOG" 2>&1
  echo "=== DONE  ${arm}/${DS} rc=$? $(ts) ===" | tee -a "$LOG"
}

ab_component() {
  need_ds
  local OUT=$ABL/component; LOG=$OUT/master_${DS}.log; mkdir -p "$OUT"
  echo "=== COMPONENT ($DS) START $(ts) ===" > "$LOG"
  run_one "$OUT" prism_v_no_act "$SEEDS" --pipelines random_cnn_padim \
      --cnn-negative-slope 1.0
  run_one "$OUT" raw_mahalanobis "" --pipelines raw_mahalanobis
  local arm pipe
  for arm_pipe in "prism_v_no_act:random_cnn_padim:--seeds $SEEDS" \
                  "raw_mahalanobis:raw_mahalanobis:"; do
    arm=${arm_pipe%%:*}; rest=${arm_pipe#*:}; pipe=${rest%%:*}; flags=${rest#*:}
    echo "=== G SIDECARS ${arm}/${DS}  $(ts) ===" | tee -a "$LOG"
    $PY "$ROOT/scripts/compute_g.py"     "$OUT/$arm/$DS/scores/$pipe" $flags >>"$LOG" 2>&1
  done
  echo "=== COMPONENT ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
}

ab_building_block() {
  need_ds
  local OUT=$ABL/building_block; LOG=$OUT/master_${DS}.log; mkdir -p "$OUT"
  echo "=== BUILDING-BLOCK ($DS) START $(ts) ===" > "$LOG"
  run_one "$OUT" no_blur "$SEEDS" --pipelines random_cnn_padim \
      --cnn-downsample maxpool
  run_one "$OUT" no_in "$SEEDS" --pipelines random_cnn_padim \
      --cnn-norm none
  run_one "$OUT" base "$SEEDS" --pipelines random_cnn_padim \
      --cnn-norm none --cnn-downsample maxpool
  echo "=== BUILDING-BLOCK ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
}

ab_blur_kernel() {
  need_ds
  local OUT=$ABL/blur_kernel; LOG=$OUT/master_${DS}.log; mkdir -p "$OUT"
  echo "=== BLUR-KERNEL ($DS) START $(ts) ===" > "$LOG"
  for k in tri3 bin5 bin9; do
    run_one "$OUT" "$k" "$SEEDS" --pipelines random_cnn_padim \
        --cnn-downsample blur --cnn-blur-kernel "$k"
  done
  echo "=== BLUR-KERNEL ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
}

ab_cov_pool() {
  need_ds
  local OUT=$ABL/cov_pool; LOG=$OUT/master_${DS}.log; mkdir -p "$OUT"
  echo "=== COV-POOL ($DS) START $(ts) ===" > "$LOG"
  run_one "$OUT" covnone "$SEEDS" --pipelines random_cnn_padim \
      --cnn-downsample blur --cnn-blur-kernel bin7 --padim-cov-pool none
  run_one "$OUT" covr2 "$SEEDS" --pipelines random_cnn_padim \
      --cnn-downsample blur --cnn-blur-kernel bin7 --padim-cov-pool r2
  echo "=== COV-POOL ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
}

ab_hidden_dim() {
  need_ds
  local OUT=$ABL/hidden_dim; LOG=$OUT/master_${DS}.log; mkdir -p "$OUT"
  echo "=== HIDDEN-DIM ($DS) START $(ts) ===" > "$LOG"
  for d in 32 64 192; do
    run_one "$OUT" "d${d}" "$SEEDS" --pipelines random_cnn_padim \
        --cnn-channels $d $d $d $d $d
  done
  echo "=== HIDDEN-DIM ($DS) COMPLETE $(ts) ===" | tee -a "$LOG"
}

ab_activation_fn() {
  local OUT=$ABL/activation_fn; LOG=$OUT/master.log; mkdir -p "$OUT"
  echo "=== ACTIVATION-FN START $(ts) ===" > "$LOG"
  for act in relu tanh sigmoid; do
    for d in mvtec visa; do
      if [ -f "$OUT/${act}/${d}/tables/seeds.json" ]; then
        echo "=== SKIP ${act}/${d} (seeds.json exists) $(ts) ===" | tee -a "$LOG"
        continue
      fi
      if [ "$d" = "visa" ]; then
        A="--root $ROOT/data/visa"
      else
        A=""
      fi
      echo "=== START ${act}/${d}  $(ts) ===" | tee -a "$LOG"
      $PY "$RUN" $A --out "$OUT/${act}/${d}" --seeds $SEEDS \
          --pipelines random_cnn_padim --cnn-activation "$act" >>"$LOG" 2>&1
      echo "=== DONE  ${act}/${d} rc=$? $(ts) ===" | tee -a "$LOG"
    done
  done
  echo "=== ACTIVATION-FN COMPLETE $(ts) ===" | tee -a "$LOG"
}

case "$NAME" in
  component)      ab_component ;;
  building_block) ab_building_block ;;
  blur_kernel)    ab_blur_kernel ;;
  cov_pool)       ab_cov_pool ;;
  hidden_dim)     ab_hidden_dim ;;
  activation_fn)  ab_activation_fn ;;
  all)
    need_ds
    ab_component
    ab_building_block
    ab_blur_kernel
    ab_cov_pool
    ab_hidden_dim
    if [ "$DS" = "mvtec" ]; then
      ab_activation_fn
    fi
    ;;
  *)
    echo "error: unknown ablation '$NAME'"; exit 1 ;;
esac
