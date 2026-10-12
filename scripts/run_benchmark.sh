#!/bin/bash
# Run strategies x seeds for one dataset, one job per GPU at a time.
#
#   DATASET=siim DATA_ROOT=/path/to/siim GPUS="0 1 2 3" SEEDS="0 1 2 3 4" bash scripts/run_benchmark.sh
#
# DATASET: siim | ddr | vindr | chestxdet10. MODES overrides the strategy list.
set -u
cd "$(dirname "$0")/.."

DATASET=${DATASET:?set DATASET}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-results}
IFS=' ' read -r -a GPUS <<< "${GPUS:-0}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"

case "$DATASET" in
  siim)
    ARGS="--siim_root $DATA_ROOT --siim_positive_only --grid 7 --num_samples 15 --round_num 25 --batch_size 8"
    DEFAULT_MODES="sparcl random area_random uncertainty uncertainty_area coreset diversity taudis usimc loss_prediction paal" ;;
  ddr)
    ARGS="--ddr_root $DATA_ROOT --ddr_lesion MA --grid 7 --num_samples 15 --round_num 24 --batch_size 8"
    DEFAULT_MODES="sparcl random area_random uncertainty uncertainty_area coreset diversity taudis usimc loss_prediction paal" ;;
  vindr)
    ARGS="--vindr_root $DATA_ROOT --target_lesion dispersed --grid 5 --num_samples 50 --round_num 25 --batch_size 16"
    DEFAULT_MODES="sparcl random area_random uncertainty uncertainty_area coreset diversity taudis usimc" ;;
  chestxdet10)
    ARGS="--chestxdet10_root $DATA_ROOT --target_lesion all --grid 5 --num_samples 50 --round_num 50 --batch_size 16"
    DEFAULT_MODES="sparcl random area_random uncertainty uncertainty_area coreset diversity taudis usimc" ;;
  *) echo "unknown DATASET: $DATASET"; exit 1 ;;
esac
IFS=' ' read -r -a MODES <<< "${MODES:-$DEFAULT_MODES}"

JOBS=()
for seed in "${SEEDS[@]}"; do for mode in "${MODES[@]}"; do JOBS+=("$seed:$mode"); done; done
mkdir -p "$OUTPUT_DIR/$DATASET"

for gi in "${!GPUS[@]}"; do
  gpu=${GPUS[$gi]}
  (
    for ji in "${!JOBS[@]}"; do
      (( ji % ${#GPUS[@]} == gi )) || continue
      seed=${JOBS[$ji]%:*}; mode=${JOBS[$ji]#*:}
      echo "[gpu$gpu] start $DATASET $mode seed$seed $(date '+%F %T')"
      CUDA_VISIBLE_DEVICES=$gpu python run_al.py --dataset "$DATASET" --mode "$mode" --seed "$seed" \
        $ARGS --output_dir "$OUTPUT_DIR" > "$OUTPUT_DIR/$DATASET/${mode}_seed${seed}.log" 2>&1
      echo "[gpu$gpu] done  $DATASET $mode seed$seed $(date '+%F %T')"
    done
  ) &
done
wait
