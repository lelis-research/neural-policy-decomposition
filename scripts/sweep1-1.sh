#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01-00:00
#SBATCH --output=outputs/sweep3_noFE_S0A/%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=behdin@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-999


source /home/iprnb/venvs/neural-policy-decomposition/bin/activate

module load flexiblas
export FLEXIBLAS=imkl

seeds=(1 2 3)
learning_rates=(0.00001 0.00075 0.005)
max_len=(50 60)
ent_coefs=(0.01 0.015 0.02 0.1 0.5)
clip_coefs=(0.01 0.025 0.1)
update_epochs=(1 5 10)
clip_vloss=(0)
visit_bonus=(1)
actor_size=(32 64 128)
critic_size=(32 64 128)

num_seed=${#seeds[@]}              # 1   
num_lr=${#learning_rates[@]}       # 2
num_max_len=${#max_len[@]}             # 2
num_ent=${#ent_coefs[@]}            # 2
num_clip=${#clip_coefs[@]}          # 2
num_epoch=${#update_epochs[@]}               # 2
num_vloss=${#clip_vloss[@]} 
num_visit=${#visit_bonus[@]}
num_actor=${#actor_size[@]} 
num_critic=${#critic_size[@]}  


# total combinations = #3 * 2^6 = 216
idx=$SLURM_ARRAY_TASK_ID

# Get index for learning rate
lr_index=$(( idx % num_lr ))
idx=$(( idx / num_lr ))

# Get index for num_max_len
max_len_index=$(( idx % num_max_len ))            # remainder
idx=$(( idx / num_max_len ))                 # integer division

# Get index for ent_coef
ent_index=$(( idx % num_ent ))
idx=$(( idx / num_ent ))

# Get index for clip_coef
clip_index=$(( idx % num_clip ))
idx=$(( idx / num_clip ))

# Get index for max_grad_norm
epoch_index=$(( idx % num_epoch ))
idx=$(( idx / num_epoch ))

# Get index for vf_coef
vloss_index=$(( idx % num_vloss ))
idx=$(( idx / num_vloss ))

# Get index for vf_coef
visit_index=$(( idx % num_visit ))
idx=$(( idx / num_visit ))

actor_index=$(( idx % num_actor ))
idx=$(( idx / num_actor ))

critic_index=$(( idx % num_critic ))
idx=$(( idx / num_critic ))


# Get index for seed
sd_index=$(( idx % num_seed ))


# Extract the actual values
SD="${seeds[${sd_index}]}"
LR="${learning_rates[${lr_index}]}"
CLIP="${clip_coefs[${clip_index}]}"
ENT="${ent_coefs[${ent_index}]}"
STEP="${max_len[${max_len_index}]}"
EPOCH="${update_epochs[${epoch_index}]}"
VL="${clip_vloss[${vloss_index}]}"
VB="${visit_bonus[${visit_index}]}"
AS="${actor_size[${actor_index}]}"
CS="${critic_size[${critic_index}]}"


# Run the training script
OMP_NUM_THREADS=1 python3.11 ~/scratch/neural-policy-decomposition/agents/agent_recurrent_positive.py \
  --seed "${SD}" \
  --num_steps "${STEP}" \
  --episode_length "${STEP}" \
  --learning_rate "${LR}" \
  --value_learning_rate "${LR}" \
  --ent_coef "${ENT}" \
  --clip_coef "${CLIP}" \
  --update_epochs "${EPOCH}" \
  --clip_vloss "${VL}" \
  --visitation_bonus "${VB}"\
  --actor_layer_size "${AS}"\
  --critic_layer_size "${CS}"
  