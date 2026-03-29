#!/usr/bin/env bash
# =============================================================================
# run_from_p1_weights.sh — Full ensemble pipeline for Phase 2 position decoding
# =============================================================================
#
# Implements teammate's recommendation:
#   "use pretrained weights from your Phase 1 model AND ensemble weights
#    (m1/m2/m5) to fine-tune for Phase 2 classification."
#
# Steps:
#   1. Fine-tune Phase 2 decoder from each of 4 Phase 1 checkpoints
#      (user's own, teammate's m1, m2, m5)  with different seeds
#   2. Ensemble inference: average predictions from all 4 models
#
# Expected improvement over single model (0.474 public):
#   - Ensemble of diverse Phase 1 inits → ~0.55–0.65 range
#   - Balanced val subset during training fixes biased early stopping
#
# Usage:
#   bash run_from_p1_weights.sh                         # default paths
#   DATA_DIR=/your/data EPOCHS=40 bash run_from_p1_weights.sh
#
# Override any variable:
#   USER_CKPT=/path/to/phase1/best.pt \
#   M1_CKPT=/path/to/m1.pt \
#   EPOCHS=50 \
#   bash run_from_p1_weights.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Paths ------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-/scratch/ml8347/neuroinformatics/project2/phase2/kaggle_data}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}}"

USER_CKPT="${USER_CKPT:-/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/best_model.pt}"
M1_CKPT="${M1_CKPT:-${SCRIPT_DIR}/../teamate/Project2/cpt/m1.pt}"
M2_CKPT="${M2_CKPT:-${SCRIPT_DIR}/../teamate/Project2/cpt/m2.pt}"
M5_CKPT="${M5_CKPT:-${SCRIPT_DIR}/../teamate/Project2/cpt/m5.pt}"

# ---- Hyperparameters --------------------------------------------------------
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-5e-4}"
WARMUP="${WARMUP:-5}"
DROPOUT="${DROPOUT:-0.1}"

mkdir -p "${OUT_ROOT}/models"

echo "========================================"
echo " Phase 2 Ensemble Decoder Pipeline"
echo "========================================"
echo "DATA_DIR   = ${DATA_DIR}"
echo "OUT_ROOT   = ${OUT_ROOT}"
echo "EPOCHS     = ${EPOCHS}"
echo "USER_CKPT  = ${USER_CKPT}"
echo "M1_CKPT    = ${M1_CKPT}"
echo "M2_CKPT    = ${M2_CKPT}"
echo "M5_CKPT    = ${M5_CKPT}"
echo ""

# ---- Helper -----------------------------------------------------------------
train_one() {
    local tag="$1"
    local ckpt="$2"
    local seed="$3"
    local outdir="${OUT_ROOT}/models/${tag}"

    echo "=== [${tag}] Fine-tuning from: ${ckpt} (seed=${seed}) ==="
    python "${SCRIPT_DIR}/train.py" \
        --data-dir "${DATA_DIR}" \
        --pretrained-checkpoint "${ckpt}" \
        --seed "${seed}" \
        --outdir "${outdir}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --encoder-lr "${ENCODER_LR}" \
        --head-lr "${HEAD_LR}" \
        --warmup-epochs "${WARMUP}" \
        --dropout "${DROPOUT}" \
        --n-val-sessions 8

    echo "[${tag}] Done. Model: ${outdir}/best_model.pt"
    echo ""
}

# ---- 1. Train 4 ensemble members -------------------------------------------
train_one "user_ft" "${USER_CKPT}" 0
train_one "m1_ft"   "${M1_CKPT}"   1
train_one "m2_ft"   "${M2_CKPT}"   2
train_one "m5_ft"   "${M5_CKPT}"   3

# ---- 2. Collect trained checkpoints ----------------------------------------
USER_FT="${OUT_ROOT}/models/user_ft/best_model.pt"
M1_FT="${OUT_ROOT}/models/m1_ft/best_model.pt"
M2_FT="${OUT_ROOT}/models/m2_ft/best_model.pt"
M5_FT="${OUT_ROOT}/models/m5_ft/best_model.pt"

for f in "${USER_FT}" "${M1_FT}" "${M2_FT}" "${M5_FT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Missing checkpoint: ${f}" >&2
        exit 1
    fi
done

# ---- 3. Plain ensemble inference (recommended) -----------------------------
echo "=== Ensemble inference (plain, no TTA) ==="
python "${SCRIPT_DIR}/submit.py" \
    --data-dir "${DATA_DIR}" \
    --checkpoints "${USER_FT}" "${M1_FT}" "${M2_FT}" "${M5_FT}" \
    --tta-epochs 0 \
    --output "${OUT_ROOT}/submission_ensemble.csv"

echo ""
echo "========================================"
echo " Submission: ${OUT_ROOT}/submission_ensemble.csv"
echo "========================================"

# ---- 4. Optional: 3-model sub-ensembles for ablation -----------------------
echo ""
echo "=== Ablation: 3-model ensembles (optional, for analysis) ==="

python "${SCRIPT_DIR}/submit.py" \
    --data-dir "${DATA_DIR}" \
    --checkpoints "${M1_FT}" "${M2_FT}" "${M5_FT}" \
    --tta-epochs 0 \
    --output "${OUT_ROOT}/submission_m1m2m5_ens.csv" && \
    echo "  m1+m2+m5 ensemble saved." || true

python "${SCRIPT_DIR}/submit.py" \
    --data-dir "${DATA_DIR}" \
    --checkpoints "${USER_FT}" "${M1_FT}" "${M2_FT}" "${M5_FT}" \
    --tta-epochs 2 --tta-lr 7e-5 \
    --output "${OUT_ROOT}/submission_ensemble_tta2.csv" && \
    echo "  4-model + TTA2 ensemble saved." || true

echo ""
echo "Done."
