#!/bin/sh
env="MPE"
scenario="simple_tag" 
num_good_agents=1
num_adversaries=3
num_landmarks=2
algo="mat" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 20 --num_env_steps 10000000 --mode mixed \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 5e-4 --critic_lr 5e-4  --clip_param 0.05 --save_interval 100 \
    --use_wandb
done
