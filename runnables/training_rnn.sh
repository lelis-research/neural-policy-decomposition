#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-3:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-19

module load python/3.11
source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

models=("gru" "lstm")
hidden_size=(6,16,32,64,128)

for model in "${models[@]}"; do
    for h_size in "${hidden_size[@]}"; do
        python3.11  /home/iprnb/projects/def-lelis/iprnb/neural-policy-decomposition/agent_recurrent.py --rnn_type "$model" --hidden_size $h_size
    done
done