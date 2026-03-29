#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name=txf_mt_scr
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_multitask_scratch/transformer_multitask_scratch_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/transformer_multitask_scratch/transformer_multitask_scratch_%j.err

START_TIME=$(date +%s)
echo "Transformer multitask scratch started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics

    cd /scratch/ml8347/neuroinformatics/project2/phase2/transformer_multitask_scratch

    echo '=== Training ==='
    python train.py \
        --epochs 60 \
        --batch-size 48 \
        --lr 3e-4 \
        --warmup-epochs 5 \
        --num-layers 6 \
        --dropout 0.1 \
        --max-seq-len 192 \
        --inference-stride 96 \
        --position-blend 0.7 \
        --vel-weight 0.5 \
        --consistency-weight 0.25 \
        --extra-dropout-max 8 \
        --wandb

    echo ''
    echo '=== Submission ==='
    python submit.py \
        --checkpoint best_model.pt \
        --output submission_scratch_multitask.csv
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
