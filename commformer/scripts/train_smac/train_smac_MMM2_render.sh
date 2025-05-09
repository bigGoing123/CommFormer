#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="mappo"
exp="check"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python ../train/train_smac_render.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 2 --episode_length 100 \
    --num_env_steps 10000000 --ppo_epoch 5 --gain 1 --use_value_active_masks --use_eval --eval_episodes 32
done
