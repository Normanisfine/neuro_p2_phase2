#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=ens_p2
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/ensemble_decoder/ensemble_%j.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/ensemble_decoder/ensemble_%j.err

START_TIME=$(date +%s)
echo "Ensemble Decoder started at: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

export CUDA_VISIBLE_DEVICES=0

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics

    cd /scratch/ml8347/neuroinformatics/project2/phase2/ensemble_decoder

    echo '=== Starting ensemble pipeline ==='
    bash run_from_p1_weights.sh

    echo ''
    echo '=== Done ==='
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 3600))h $(( (DURATION % 3600) / 60))m $((DURATION % 60))s"
