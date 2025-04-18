import os
from pathlib import Path

def get_path(run_dir):

    wandb_dir = run_dir / "wandb" 

    #获取该文件夹下以run开头的文件夹并取最新
    run_dirs = [f for f in os.listdir(wandb_dir) if f.startswith('run')]
    latest_run_dir = max(run_dirs, key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)))
    wandb_dir = wandb_dir / latest_run_dir / "files"
    # 获取wandb_dir下的所有以.pt结尾的文件
    files = [f for f in os.listdir(wandb_dir) if f.endswith('.pt')]
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)))
    model_path = os.path.join(wandb_dir, latest_file)
    return model_path