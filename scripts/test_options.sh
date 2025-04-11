#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-6:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-9

module load flexiblas
export FLEXIBLAS=blis2

source envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

OMP_NUM_THREADS=1 python -m pipelines.test_by_training --seed=$SLURM_ARRAY_TASK_ID
