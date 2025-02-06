#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-4:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-9

module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

wandb offline
OMP_NUM_THREADS=1 python -m pipelines.train_ppo --cpus=$SLURM_CPUS_PER_TASK --seed=$SLURM_ARRAY_TASK_ID

# wandb sync --include-offline --sync-all
