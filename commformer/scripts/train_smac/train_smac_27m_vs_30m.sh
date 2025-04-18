#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
algo="happo"
exp="check"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 10000000 --ppo_epoch 15 --clip_param 0.2 --use_value_active_masks --use_eval --eval_episodes 32 \
    --use_wandb
done
