#!/bin/bash
#SBATCH --cpus-per-task=32   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G        # memory per node
#SBATCH --time=2-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL


source /home/iprnb/venvs/neural-policy-decomposition/bin/activate
module load python/3.11 scipy-stack

python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/agent_recurrent.py --rnn_type "gru" --hidden_size 32 --l1_lambda 0.0001 --total_timesteps 20000000 --problem "TR-BL"
