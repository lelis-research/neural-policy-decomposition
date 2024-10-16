#!/bin/bash
#SBATCH --cpus-per-task=20   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=6G        # memory per node
#SBATCH --time=0-8:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis
#SBATCH --array=0-2

source envs/venv/bin/activate # Assuming we have all our environments in  `../envs/`

seed = $SLURM_ARRAY_TASK_ID

python -m pipelines.extract_subpolicy_ppo.py --log_path=outputs/logs/logfile \
--exp_name=option_extraction_hc_logits_1000 --cpus=$SLURM_CPUS_PER_TASK