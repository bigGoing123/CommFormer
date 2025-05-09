#!/bin/sh
env="StarCraft2"
map="3s5z_vs_3s6z"
algo="mat"
exp="check"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 20000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32 \
    --use_wandb
done
