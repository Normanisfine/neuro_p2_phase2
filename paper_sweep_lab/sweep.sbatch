#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --job-name=p2_paper_sw
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/paper_sweep_lab/paper_sweep_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/paper_sweep_lab/paper_sweep_%A_%a.err
#SBATCH --array=0-7

START_TIME=$(date +%s)
echo "Paper sweep started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

CFG_DIR="/scratch/ml8347/neuroinformatics/project2/phase2/paper_sweep_lab/configs"
CFG_FILE=$(ls "${CFG_DIR}"/*.json | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
CFG_NAME=$(basename "${CFG_FILE}")
echo "Config: ${CFG_NAME}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd /scratch/ml8347/neuroinformatics/project2/phase2/paper_sweep_lab
    python sweep.py --run \"${CFG_NAME%.json}\" --wandb
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
