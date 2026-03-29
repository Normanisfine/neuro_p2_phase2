#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --job-name=mae_ft_sub_sweep
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/sub_sweep_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/sub_sweep_%A_%a.err
#SBATCH --array=0-7

# ─── TTA submission sweep ────────────────────────────────────────────────
# Set MAE_FT_CKPT to the best training checkpoint before running:
#   export MAE_FT_CKPT=/path/to/sweep_xxx/best_model.pt
# Then: sbatch submit_sweep.sbatch

CKPT="${MAE_FT_CKPT:-best_model.pt}"
WORKDIR="/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune"

#              0     1     2     3     4     5     6     7
TTA_EP=(       0     3     5    10     5     5    10     5)
TTA_LR=(    1e-5  1e-5  1e-5  1e-5  5e-6  2e-5  1e-5  1e-5)
TTA_BS=(      32    32    32    32    32    32    32    64)

i=$SLURM_ARRAY_TASK_ID
TAG="tta${TTA_EP[$i]}_lr${TTA_LR[$i]}"
OUTFILE="${WORKDIR}/submission_${TAG}.csv"

echo "Config ${i}: ${TAG}"
echo "Checkpoint: ${CKPT}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd \"$WORKDIR\"

    python submit.py \
        --checkpoint \"${CKPT}\" \
        --tta-epochs ${TTA_EP[$i]} \
        --tta-lr ${TTA_LR[$i]} \
        --tta-bs ${TTA_BS[$i]} \
        --output \"${OUTFILE}\"
"

echo "Done: ${OUTFILE}"
