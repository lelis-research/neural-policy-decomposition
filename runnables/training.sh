#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-3:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-19

module load python/3.10

# source /home/rezaabdz/personal-projects/neurosymbolic-programming/dec_options_baseline_v1_env/bin/activate

python3.10 /home/rezaabdz/personal-projects/neurosymbolic-programming/Dec-Options/Archive/combo4_training.py --seed $SLURM_ARRAY_TASK_ID --size 3 --num_iterations 170000 --reg_coef 0.0000 --clip_range 0.1 --ent_coef 0.1 --learning_rate 0.001 --network_size 24 --log_path "$dir"
