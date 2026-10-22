#!/bin/bash

# 1. Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rlpd

# 2. MuJoCo paths
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# 3. Force headless EGL rendering (Hardware acceleration via Windows GPU)
export MUJOCO_GL=egl

# 4. Bypass hardcoded mujoco-py NVIDIA check AND point to WSL GPU drivers
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib:/usr/lib/nvidia

# Navigate to your local WSL directory
cd ~/rlpd

# Optional: WSL has internet, so you can change this to "online" to see live graphs
export WANDB_MODE=offline
export WANDB_NAME=relocate_dgn_train1   

# Execute the training job (paths updated to local ~/)
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --env_name=relocate-binary-v0 \
    --utd_ratio=20 \
    --start_training 0 \
    --batch_size=128 \
    --max_steps 500000 \
    --config=configs/rlpd_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(256, 256, 256)" \
    --config.actor_lr=1e-4 --config.critic_lr=1e-4 --config.temp_lr=1e-4 \
    --project_name=rlpd_adroit \
    --checkpoint_model=True \
    --log_dir=~/rlpd/logs \
    --seed=43 \
