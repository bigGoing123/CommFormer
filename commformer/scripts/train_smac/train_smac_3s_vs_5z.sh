#!/bin/sh
env="StarCraft2"
map="3s_vs_5z"
algo="commformer"
exp="check"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 4 --n_rollout_threads 16 --num_mini_batch 4 --episode_length 400 \
    --num_env_steps 5000000 --ppo_epoch 15 --clip_param 0.05 --use_value_active_masks  --eval_episodes 32 --stacked_frames 4 --use_stacked_frames\
    --lr 1e-4 --critic_lr 1e-4 --entropy_coef 0.01 --communication_weight 0.5 --attention_heads  --use_bilevel --comm_mode lf --use_wandb \

    CUDA_VISIBLE_DEVICES=3 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 4 --n_rollout_threads 16 --num_mini_batch 4 --episode_length 400 \
    --num_env_steps 5000000 --ppo_epoch 15 --clip_param 0.05 --use_value_active_masks  --eval_episodes 32 --stacked_frames 4 --use_stacked_frames\
    --lr 1e-4 --critic_lr 1e-4 --entropy_coef 0.01 --communication_weight 0.5 --attention_heads  --use_bilevel --comm_mode com --use_wandb \
done
