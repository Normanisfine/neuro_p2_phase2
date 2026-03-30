#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name=mae_mt_p2
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_decoder/mae_multitask_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_decoder/mae_multitask_%j.err

START_TIME=$(date +%s)
echo "MAE multitask decoder started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

# Override before launching if you want a different Phase 1 checkpoint:
#   export PHASE1_CKPT=/path/to/best_model.pt
PHASE1_CKPT="${PHASE1_CKPT:-/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_3325164_5/best_model.pt}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics

    cd /scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_decoder

    echo '=== Training ==='
    python train.py \
        --epochs 50 \
        --batch-size 32 \
        --encoder-lr 1e-5 \
        --head-lr 5e-4 \
        --warmup-epochs 5 \
        --dropout 0.1 \
        --window-size 250 \
        --window-stride 125 \
        --position-blend 0.75 \
        --vel-weight 0.5 \
        --consistency-weight 0.25 \
        --extra-dropout-max 8 \
        --pretrained-checkpoint \"${PHASE1_CKPT}\" \
        --wandb

    echo ''
    echo '=== Submission ==='
    python submit.py \
        --checkpoint best_model.pt \
        --output submission_multitask.csv
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
