#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=mae_ft_p2
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/mae_finetune_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune/mae_finetune_%j.err

START_TIME=$(date +%s)
echo "MAE Finetune started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

# Set PHASE1_CKPT to the best Phase 1 checkpoint before running.
# Run: bash /scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/pick_best.sh
# to identify the best sweep checkpoint, then:
#   export PHASE1_CKPT=/path/to/sweep_JOBID_N/best_model.pt
PHASE1_CKPT="${PHASE1_CKPT:-/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/best_model.pt}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics

    cd /scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune

    echo '=== Training ==='
    python train.py \
        --epochs 40 \
        --batch-size 64 \
        --encoder-lr 1e-5 \
        --head-lr 5e-4 \
        --warmup-epochs 5 \
        --dropout 0.1 \
        --pretrained-checkpoint \"${PHASE1_CKPT}\" \
        --wandb

    echo ''
    echo '=== Plain submission ==='
    python submit.py --output submission_plain.csv

    echo ''
    echo '=== TTA submissions ==='
    python submit.py --tta-epochs 3  --tta-lr 1e-5 --output submission_tta3.csv
    python submit.py --tta-epochs 5  --tta-lr 1e-5 --output submission_tta5.csv
    python submit.py --tta-epochs 10 --tta-lr 1e-5 --output submission_tta10.csv
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
