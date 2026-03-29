#!/usr/bin/env bash
set -euo pipefail

# Train 3 models from scratch with the best-known task8 settings, then
# reproduce the best submission via:
#   - TTA (epoch=14, lr=7e-5) with seeds 0..4 per model
#   - mean ensemble across seeds per model
#   - mean ensemble across models (m5 + m2 + m1)
#
# Usage:
#   bash run_best_pipeline_from_scratch.sh
#
# Optional overrides:
#   DATA_DIR=/scratch/.../kaggle_data OUT_ROOT=/scratch/.../my_run \
#   bash run_best_pipeline_from_scratch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/th3482/neuroinformatics/project2/kaggle_data}"
OUT_ROOT="${OUT_ROOT:-/scratch/th3482/neuroinformatics/project2/masked_autoencoder/repro_best_from_scratch_$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
DROPOUT="${DROPOUT:-0.05}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
LOSS_MODE="${LOSS_MODE:-mse}"
N_MASKED_CHANNELS="${N_MASKED_CHANNELS:-30}"
EMA_DECAY="${EMA_DECAY:-0.999}"

TTA_EPOCHS="${TTA_EPOCHS:-14}"
TTA_LR="${TTA_LR:-7e-5}"
TTA_BS="${TTA_BS:-32}"
SEEDS="${SEEDS:-0 1 2 3 4}"

mkdir -p "${OUT_ROOT}"/{models,seed_submissions,ensemble}

echo "== Train Config =="
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "epochs=${EPOCHS}, bs=${BATCH_SIZE}, lr=${LR}, warmup=${WARMUP}, dropout=${DROPOUT}"
echo "recon_weight=${RECON_WEIGHT}, loss_mode=${LOSS_MODE}, n_masked_channels=${N_MASKED_CHANNELS}"
echo "tta: epochs=${TTA_EPOCHS}, lr=${TTA_LR}, bs=${TTA_BS}, seeds=${SEEDS}"

train_one() {
  local tag="$1"
  local seed="$2"
  local use_ema="$3"
  local model_dir="${OUT_ROOT}/models/${tag}"
  mkdir -p "${model_dir}"
  pushd "${model_dir}" >/dev/null
  echo "[TRAIN] ${tag} seed=${seed} use_ema=${use_ema}"
  CMD=(
    python "${SCRIPT_DIR}/train.py"
    --data-dir "${DATA_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --warmup-epochs "${WARMUP}"
    --dropout "${DROPOUT}"
    --recon-weight "${RECON_WEIGHT}"
    --loss-mode "${LOSS_MODE}"
    --n-masked-channels "${N_MASKED_CHANNELS}"
    --include-test-unmasked-trials
    --seed "${seed}"
    --ema-decay "${EMA_DECAY}"
  )
  if [[ "${use_ema}" == "1" ]]; then
    CMD+=(--use-ema)
  fi
  "${CMD[@]}"
  popd >/dev/null
}

# Match the known best 3-model set:
# m1: seed=1, EMA off
# m2: seed=2, EMA off
# m5: seed=2, EMA on
train_one "m1" 1 0
train_one "m2" 2 0
train_one "m5" 2 1

run_tta_for_model() {
  local tag="$1"
  local ckpt="${OUT_ROOT}/models/${tag}/best_model.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint after training: ${ckpt}" >&2
    exit 1
  fi
  for s in ${SEEDS}; do
    local out_csv="${OUT_ROOT}/seed_submissions/${tag}__e${TTA_EPOCHS}_lr${TTA_LR//./p}__s${s}.csv"
    echo "[TTA] ${tag} seed=${s} -> ${out_csv}"
    python "${SCRIPT_DIR}/submit.py" \
      --data-dir "${DATA_DIR}" \
      --checkpoint "${ckpt}" \
      --tta-epochs "${TTA_EPOCHS}" \
      --tta-lr "${TTA_LR}" \
      --tta-bs "${TTA_BS}" \
      --tta-seed "${s}" \
      --output "${out_csv}"
  done
}

run_tta_for_model "m1"
run_tta_for_model "m2"
run_tta_for_model "m5"

python - <<'PY' "${OUT_ROOT}"
import json
import os
import sys
import pandas as pd

out_root = sys.argv[1]
seed_dir = os.path.join(out_root, "seed_submissions")
ens_dir = os.path.join(out_root, "ensemble")
os.makedirs(ens_dir, exist_ok=True)

def mean_csv(in_paths, out_path):
    base = None
    key_cols = None
    vals = []
    for p in in_paths:
        df = pd.read_csv(p)
        if base is None:
            base = df.copy()
            key_cols = [c for c in df.columns if c != "predicted_sbp"]
        else:
            if not base[key_cols].equals(df[key_cols]):
                raise RuntimeError(f"Index columns mismatch: {p}")
        vals.append(df["predicted_sbp"].values.astype("float64"))
    base["predicted_sbp"] = sum(vals) / len(vals)
    base.to_csv(out_path, index=False)
    return out_path

model_ens = {}
for tag in ["m1", "m2", "m5"]:
    in_paths = sorted(
        os.path.join(seed_dir, f) for f in os.listdir(seed_dir)
        if f.startswith(f"{tag}__") and f.endswith(".csv")
    )
    if len(in_paths) == 0:
        raise RuntimeError(f"No seed submissions found for {tag}")
    out_path = os.path.join(ens_dir, f"{tag}__ens_s0_s1_s2_s3_s4.csv")
    mean_csv(in_paths, out_path)
    model_ens[tag] = out_path

final_path = os.path.join(ens_dir, "m5plusm2plusm1.csv")
mean_csv([model_ens["m5"], model_ens["m2"], model_ens["m1"]], final_path)

summary = {
    "models_dir": os.path.join(out_root, "models"),
    "seed_submission_dir": seed_dir,
    "model_ensemble": model_ens,
    "final_submission": final_path,
}
with open(os.path.join(out_root, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"FINAL_SUBMISSION={final_path}")
print(f"SUMMARY_JSON={os.path.join(out_root, 'summary.json')}")
PY

echo "Done."
