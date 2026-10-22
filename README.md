
# Installation

```bash
conda create -n rlpd python=3.9 # If you use conda.
conda activate rlpd
conda install patchelf  # If you use conda.
```

Requirements should be installed from freeze.txt, MuJoCo 2.1 and mj_envs should be installed separately.
Instructions coming soon.


## Adroit Binary

First, download and unzip `.npy` files into `~/.datasets/awac-data/` from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view).
Updated instructions for below coming soon.
Make sure you have `mjrl` installed:
```bash
git clone https://github.com/aravindr93/mjrl
cd mjrl
pip install -e .
```

Then, recursively clone `mj_envs` from this fork:
```bash
git clone --recursive https://github.com/philipjball/mj_envs.git
```

Then sync the submodules (add the `--init` flag if you didn't recursively clone):
```bash
$ cd mj_envs  
$ git submodule update --remote
```

Finally:
```bash
$ pip install -e .
```

Now you can run the following in this directory
```bash
#!/bin/bash


eval "$(conda shell.bash hook)"
conda activate rlpd


export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin


export MUJOCO_GL=egl


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib:/usr/lib/nvidia


cd ~/rlpd


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
    --dgn=True
```


