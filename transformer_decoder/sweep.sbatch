#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name=txf_sweep
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/sweep_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/sweep_%A_%a.err
#SBATCH --array=0-7

# ─── Hyperparameter sweep: lr, dropout, num_layers, epochs ───────────────

#               0      1      2      3      4      5      6      7
LR=(         5e-4   3e-4   5e-4   1e-3   5e-4   3e-4   5e-4   5e-4)
DROPOUT=(    0.10   0.10   0.15   0.10   0.05   0.10   0.10   0.10)
N_LAYERS=(      4      4      4      4      4      6      6      4)
EPOCHS=(       50     50     50     50     50     50     40     40)
WARMUP=(        5      5      5      3      5      5      5      3)

i=$SLURM_ARRAY_TASK_ID
RUN_DIR="/scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/sweep_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$RUN_DIR"

TAG="lr${LR[$i]}_do${DROPOUT[$i]}_L${N_LAYERS[$i]}_ep${EPOCHS[$i]}"
echo "Config ${i}: ${TAG}"
START_TIME=$(date +%s)

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd \"$RUN_DIR\"

    python /scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/train.py \
        --epochs ${EPOCHS[$i]} \
        --batch-size 64 \
        --lr ${LR[$i]} \
        --dropout ${DROPOUT[$i]} \
        --num-layers ${N_LAYERS[$i]} \
        --warmup-epochs ${WARMUP[$i]} \
        --wandb \
        --wandb-name txf_${TAG}
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Config ${i} done: $((DURATION / 60))m $((DURATION % 60))s"
