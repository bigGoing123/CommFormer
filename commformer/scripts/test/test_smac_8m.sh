#!/bin/sh
env="StarCraft2"
map="8m"
algo="mat"
exp="test"
model_dir="transformer_0.pt"  # 替换为实际模型路径

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"
CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --use_eval --model_dir ${model_dir} --eval_episodes 32 --n_rollout_threads 1 \
--n_training_threads 1