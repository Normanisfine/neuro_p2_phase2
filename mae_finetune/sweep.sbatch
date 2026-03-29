#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name=mae_ft_sweep
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/sweep_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/sweep_%A_%a.err
#SBATCH --array=0-7

# ─── Hyperparameter sweep: encoder_lr, head_lr, dropout, epochs ──────────
# Usage:
#   export PHASE1_CKPT=/path/to/best_model.pt
#   sbatch sweep.sbatch
# After sweep completes: bash pick_best.sh

PHASE1_CKPT="${PHASE1_CKPT:-/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/best_model.pt}"

#               0      1      2      3      4      5      6      7
ENC_LR=(     1e-5   1e-5   5e-5   5e-5   1e-5   1e-5   5e-5   5e-5)
HEAD_LR=(    5e-4   1e-4   5e-4   1e-4   5e-4   5e-4   1e-3   5e-4)
DROPOUT=(    0.10   0.10   0.10   0.10   0.15   0.05   0.10   0.10)
EPOCHS=(       40     40     40     40     50     40     30     50)
WARMUP=(        5      5      5      5      5      3      3      5)

i=$SLURM_ARRAY_TASK_ID
RUN_DIR="/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/sweep_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$RUN_DIR"

TAG="elr${ENC_LR[$i]}_hlr${HEAD_LR[$i]}_do${DROPOUT[$i]}_ep${EPOCHS[$i]}"
echo "Config ${i}: ${TAG}"
START_TIME=$(date +%s)

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd \"$RUN_DIR\"

    python /scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/train.py \
        --epochs ${EPOCHS[$i]} \
        --batch-size 64 \
        --encoder-lr ${ENC_LR[$i]} \
        --head-lr ${HEAD_LR[$i]} \
        --dropout ${DROPOUT[$i]} \
        --warmup-epochs ${WARMUP[$i]} \
        --pretrained-checkpoint \"${PHASE1_CKPT}\" \
        --wandb \
        --wandb-name mae_ft_${TAG}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Config ${i} done: $((DURATION / 60))m $((DURATION % 60))s"
