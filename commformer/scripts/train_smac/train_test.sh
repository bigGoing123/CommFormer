#!/bin/sh
env="StarCraft2"
map="2c_vs_64zg"
algo="mat"
exp="check"
seed=2

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, running seed ${seed}"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 4 --n_rollout_threads 16 --num_mini_batch 4 --episode_length 400 \
    --num_env_steps 5000000 --ppo_epoch 15  --use_value_active_masks --use_eval --eval_episodes 64 --use_wandb \
    --lr 1e-4 --critic_lr 1e-4 --entropy_coef 0.05 --communication_weight 0.5 --attention_heads 4
