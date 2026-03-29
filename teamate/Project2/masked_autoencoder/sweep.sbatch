#!/bin/bash
#SBATCH --account=sth
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --job-name=mae_sweep
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_%A_%a.err
#SBATCH --array=0-7

# ─── Training sweep ──────────────────────────────────────────────────
# Same architecture (4L d256), sweep: lr, dropout, epochs, recon_weight
# After sweep: pick best → run submit.py with TTA + ridge combos

#              0     1     2     3     4     5     6     7
LR=(        5e-4  3e-4  1e-3  5e-4  5e-4  3e-4  5e-4  3e-4)
DROPOUT=(   0.10  0.10  0.10  0.15  0.05  0.05  0.10  0.10)
EPOCHS=(      40    40    40    40    40    50    30    50)
WARMUP=(       5     5     3     5     5     5     3     5)
RECON_W=(   0.00  0.00  0.00  0.00  0.00  0.00  0.05  0.05)

i=$SLURM_ARRAY_TASK_ID
RUN_DIR="/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$RUN_DIR"

TAG="lr${LR[$i]}_do${DROPOUT[$i]}_ep${EPOCHS[$i]}_wu${WARMUP[$i]}_rw${RECON_W[$i]}"
echo "Config ${i}: ${TAG}"
START_TIME=$(date +%s)

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd \"$RUN_DIR\"

    python /scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/train.py \
        --epochs ${EPOCHS[$i]} \
        --batch-size 64 \
        --lr ${LR[$i]} \
        --dropout ${DROPOUT[$i]} \
        --warmup-epochs ${WARMUP[$i]} \
        --recon-weight ${RECON_W[$i]} \
        --wandb \
        --wandb-name mae_${TAG}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Config ${i} done: $((DURATION / 60))m $((DURATION % 60))s"
