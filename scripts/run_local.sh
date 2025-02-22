#!/bin/bash

source envs/venv/bin/activate


# for seed in {0..4}
# do
#   # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_decOptionWhole_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_decOptionWhole_randomInit"
#   # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_learnOptions_randomInit_pitisFunction_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_learnOptions_randomInit_pitisFunction"
#   # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_learnOptions_randomInit_discreteMasks_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_learnOptions_randomInit_discreteMasks"
#   # python -m pipelines.train_ppo --seed=$seed --exp_name="test_noOptions"

#   python -m pipelines.train_ppo --seed=$seed --exp_name="train_ppoAgent_randomInit"
# done

for seed in {0..0}
do
  # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_decOptionWhole_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_decOptionWhole_randomInit"
  # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_learnOptions_randomInit_pitisFunction_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_learnOptions_randomInit_pitisFunction"
  # python -m pipelines.test_by_training --seed=$seed --exp_id="extract_learnOptions_randomInit_discreteMasks_MiniGrid-SimpleCrossingS9N1-v0_gw5_h64_l10_r400_envsd0,1,2" --test_exp_name="test_learnOptions_randomInit_discreteMasks"
  # python -m pipelines.train_ppo --seed=$seed --exp_name="test_noOptions"

  # python -m pipelines.train_ppo --seed=$seed --exp_name="train_ppoAgent_randomInit"

# python -m pipelines.extract_subpolicy_ppo --seed=$seed --exp_name="extract_ppoDecOption_randomInit"   
python -m pipelines.test_by_training --seed=$seed --exp_id="extract_ppoDecOption_randomInit_MiniGrid-SimpleCrossingS9N1-v0_gw5_h6_l10_r400_envsd0,1,2" --test_exp_name="test_ppoDecOption_randomInit"
done