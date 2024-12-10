#!/bin/bash
#SBATCH --cpus-per-task=8   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --time=4-00:00      # time (DD-HH:MM)
#SBATCH --output=outputs/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-3


source /home/iprnb/venvs/neural-policy-decomposition/bin/activate
module load python/3.11 scipy-stack
module load flexiblas
export FLEXIBLAS=imkl

OMP_NUM_THREADS=1 python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/agent_recurrent.py --problem "test" --seed $SLURM_ARRAY_TASK_ID
