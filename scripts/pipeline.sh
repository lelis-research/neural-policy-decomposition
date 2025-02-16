#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=0       # memory per node
#SBATCH --time=8-00:00      # time (DD-HH:MM)
#SBATCH --output=outputs/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-4

source /home/iprnb/venvs/neural-policy-decomposition/bin/activate
module load flexiblas
export FLEXIBLAS=imkl

OMP_NUM_THREADS=1 python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/driver.py  --seed $SLURM_ARRAY_TASK_ID
