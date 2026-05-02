#!/usr/bin/env bash
set -euo pipefail

cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM

TASK="task_hzey_stage1_aonly_bm"
DATA_ROOT="/data/xuewz/WSI_PRE/CLAM_0423/data/features"
SPLIT_DIR="splits/task_hzey_stage1_aonly_bm_100"

export_one_method () {
  local METHOD_NAME="$1"
  local MODEL_TYPE="$2"
  local CKPT_DIR="$3"
  local SAVE_DIR="$4"

  echo ""
  echo "================ export ${METHOD_NAME} ================"
  mkdir -p "${SAVE_DIR}"

  for f in 0 1 2 3 4; do
    local CKPT="${CKPT_DIR}/s_${f}_checkpoint.pt"
    local CSV="${SAVE_DIR}/energy_scores_fold_${f}.csv"

    if [ ! -f "${CKPT}" ]; then
      echo "[ERROR] missing checkpoint: ${CKPT}"
      exit 1
    fi

    if [ -f "${CSV}" ]; then
      echo "[SKIP] ${CSV} exists"
      continue
    fi

    python tools/export_energy_scores.py \
      --task "${TASK}" \
      --data_root_dir "${DATA_ROOT}" \
      --split_dir "${SPLIT_DIR}" \
      --ckpt_path "${CKPT}" \
      --model_type "${MODEL_TYPE}" \
      --temperature 1.0 \
      --save_dir "${SAVE_DIR}" \
      --fold "${f}" \
      --embed_dim 1024 \
      --drop_out 0.25
  done
}

export_one_method \
  "CLAM-SB baseline" \
  "clam_sb" \
  "/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM/results/hzey_stage1_aonly_bm_clam_sb_r50_s1" \
  "eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy"

export_one_method \
  "ECLAM-SB constant" \
  "eclam_sb" \
  "results/hzey_stage1_aonly_bm_eclam_sb_constant_check_s1" \
  "eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_constant_check_energy"

export_one_method \
  "ECLAM-SB linear" \
  "eclam_sb" \
  "results/hzey_stage1_aonly_bm_eclam_sb_linear_warmup_s1" \
  "eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy"

export_one_method \
  "ECLAM-SB sigmoid" \
  "eclam_sb" \
  "results/hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_s1" \
  "eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup_energy"

echo ""
echo "===== energy CSV count ====="
find eval_results -path "*energy/energy_scores_fold_*.csv" -print | sort
