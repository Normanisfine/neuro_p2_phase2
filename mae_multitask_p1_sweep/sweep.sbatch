#!/bin/bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --job-name=mae_mt_p1_sweep
#SBATCH --output=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_p1_sweep/mae_mt_p1_%A_%a.out
#SBATCH --error=/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_p1_sweep/mae_mt_p1_%A_%a.err
#SBATCH --array=0-15

START_TIME=$(date +%s)
echo "MAE multitask P1 sweep started at: $(date)"

export CUDA_VISIBLE_DEVICES=0

CFG_DIR="/scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_p1_sweep/configs"
CFG_FILE=$(ls "${CFG_DIR}"/*.json | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
CFG_NAME=$(basename "${CFG_FILE%.json}")
echo "Array task: ${SLURM_ARRAY_TASK_ID} — config: ${CFG_NAME}"

singularity exec --nv \
    --overlay /scratch/ml8347/neuroinformatics/neuro_overlay.ext3:ro \
    /scratch/ml8347/container_volune/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh
    conda activate neuroinformatics
    cd /scratch/ml8347/neuroinformatics/project2/phase2/mae_multitask_p1_sweep
    python sweep.py --run \"${CFG_NAME}\" --wandb
"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Finished at: $(date)"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s"
