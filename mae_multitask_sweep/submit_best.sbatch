#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=0:30:00
#SBATCH --job-name=mae_submit
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_sweep/submit_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_sweep/submit_%j.err

# Usage:
#   sbatch submit_best.sbatch           # best run
#   sbatch submit_best.sbatch 2s_75ov   # specific run

RUN="${1:-best}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd /scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_sweep
    python results.py --submit ${RUN}
"
