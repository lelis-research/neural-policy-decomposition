#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=outputs/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-2


source /home/iprnb/venvs/neural-policy-decomposition/bin/activate
module load flexiblas
export FLEXIBLAS=imkl

python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/agent_recurrent.py --problem "BL-TR" --seed $SLURM_ARRAY_TASK_ID --episode_length 12 --num_steps 30 --total_timesteps 4000000
