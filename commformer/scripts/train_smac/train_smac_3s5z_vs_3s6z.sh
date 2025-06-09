#!/bin/sh
env="StarCraft2"
algo="commformer"
exp="single"
seed_max=3
name="CommFormer"
map="3s5z_vs_3s6z"
ppo_epochs=5
ppo_clip=0.05
steps=20000000
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
# do
#   CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#     --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 \
#     --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#     --use_value_active_masks --prefix_name ${name}   --use_bilevel --post_stable --self_loop_add --post_ratio 0.6 \
#     --comm_mode com --use_wandb
# done

for seed in `seq ${seed_max}`;
do
  CUDA_VISIBLE_DEVICES=1 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 4 --n_rollout_threads 8 --num_mini_batch 1 \
    --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
    --use_value_active_masks --prefix_name ${name}   --use_bilevel --post_stable --self_loop_add --post_ratio 0.6 \
    --comm_mode lf --use_wandb
done