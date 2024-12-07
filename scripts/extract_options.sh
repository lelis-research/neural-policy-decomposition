#!/bin/bash
#SBATCH --cpus-per-task=20   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-9

module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

OMP_NUM_THREADS=1 python -m pipelines.extract_subpolicy_ppo \
--exp_name=extract_hc_all_segments_logits_levin --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID