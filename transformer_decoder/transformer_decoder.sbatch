#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --job-name=txf_p2
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/transformer_decoder_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder/transformer_decoder_%j.err

START_TIME=$(date +%s)
echo "Transformer Decoder started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c '
    source /ext3/env.sh
    conda activate neuroinformatics

    cd /scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder

    echo "=== Training ==="
    python train.py --epochs 50 --batch-size 64 --lr 5e-4 --wandb

    echo ""
    echo "=== Generating submission ==="
    python submit.py
'

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
