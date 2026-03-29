#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best submission by:
# 1) running TTA for 3 checkpoints with seeds 0..4
# 2) mean-ensembling seeds per model
# 3) mean-ensembling the 3 model-level submissions
#
# Default checkpoints match the known best combo:
#   m5 + m2 + m1 with TTA (epoch=14, lr=7e-5, seeds=0..4)
#
# Usage:
#   bash run_best_pipeline_from_weights.sh
#
# Optional overrides:
#   DATA_DIR=/scratch/.../kaggle_data OUT_ROOT=/scratch/.../my_run \
#   CKPT_M1=/path/to/m1.pt CKPT_M2=/path/to/m2.pt CKPT_M5=/path/to/m5.pt \
#   bash run_best_pipeline_from_weights.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-/scratch/th3482/neuroinformatics/project2/kaggle_data}"
OUT_ROOT="${OUT_ROOT:-/scratch/th3482/neuroinformatics/project2/masked_autoencoder/repro_best_from_weights_$(date +%Y%m%d_%H%M%S)}"

CKPT_M1="${CKPT_M1:-/scratch/th3482/neuroinformatics/project2/masked_autoencoder/task8_seed_ema_4408948_1/model/best_model.pt}"
CKPT_M2="${CKPT_M2:-/scratch/th3482/neuroinformatics/project2/masked_autoencoder/task8_seed_ema_4408948_2/model/best_model.pt}"
CKPT_M5="${CKPT_M5:-/scratch/th3482/neuroinformatics/project2/masked_autoencoder/task8_seed_ema_4408948_5/model/best_model.pt}"

TTA_EPOCHS="${TTA_EPOCHS:-14}"
TTA_LR="${TTA_LR:-7e-5}"
TTA_BS="${TTA_BS:-32}"
SEEDS="${SEEDS:-0 1 2 3 4}"

mkdir -p "${OUT_ROOT}"/{seed_submissions,ensemble}

echo "== Config =="
echo "REPO_DIR=${REPO_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "CKPT_M1=${CKPT_M1}"
echo "CKPT_M2=${CKPT_M2}"
echo "CKPT_M5=${CKPT_M5}"
echo "TTA: epochs=${TTA_EPOCHS}, lr=${TTA_LR}, bs=${TTA_BS}, seeds=${SEEDS}"

for ck in "${CKPT_M1}" "${CKPT_M2}" "${CKPT_M5}"; do
  if [[ ! -f "${ck}" ]]; then
    echo "Missing checkpoint: ${ck}" >&2
    exit 1
  fi
done

run_tta_for_model() {
  local tag="$1"
  local ckpt="$2"
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

run_tta_for_model "m1" "${CKPT_M1}"
run_tta_for_model "m2" "${CKPT_M2}"
run_tta_for_model "m5" "${CKPT_M5}"

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
    cols = None
    vals = []
    for p in in_paths:
        df = pd.read_csv(p)
        if base is None:
            base = df.copy()
            cols = [c for c in df.columns if c != "predicted_sbp"]
        else:
            if not base[cols].equals(df[cols]):
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
