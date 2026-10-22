#! /usr/bin/env python
import os
import pickle

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.dgn import DGN, actor_mean, noise_scale  # DGN
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)

# DGN: Data-Guided Noise flags. Defaults follow the paper's Adroit settings (Table 2).
flags.DEFINE_boolean("dgn", False, "Use Data-Guided Noise for exploration rollouts.")
flags.DEFINE_integer(
    "dgn_update_interval", 2000, "Env steps between covariance refits (paper: 2000 Adroit)."
)
flags.DEFINE_integer("dgn_epochs", 10, "Epochs per refit (paper: 10 Adroit).")
flags.DEFINE_integer("dgn_hidden_dim", 256, "Covariance MLP width (paper: 256 Adroit).")
flags.DEFINE_integer("dgn_num_layers", 2, "Covariance MLP hidden layers (paper: 2).")
flags.DEFINE_integer("dgn_batch_size", 128, "Covariance fit batch size (paper: 128).")
flags.DEFINE_float("dgn_dropout", 0.5, "Covariance MLP dropout (paper: 0.5).")
flags.DEFINE_float("dgn_weight_decay", 3e-2, "AdamW weight decay (paper: 3e-2).")
flags.DEFINE_float("dgn_lr", 3e-4, "Covariance LR (NOT specified in the paper).") # Not specified
flags.DEFINE_float(
    "dgn_anneal_tau", 30000.0, "Noise scale exp(-t/tau); <=0 disables (paper Adroit: 30000)."
)

config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    # not ideal, but works for now:
    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    # DGN: the prior data (human demos + BC rollouts, exactly what RLPD's offline buffer
    # holds) is also the dataset the covariance net is fit on.
    dgn, demo_obs, demo_act = None, None, None
    if FLAGS.dgn:
        demo_obs = np.asarray(ds.dataset_dict["observations"], dtype=np.float32)
        demo_act = np.asarray(ds.dataset_dict["actions"], dtype=np.float32)
        dgn = DGN(
            obs_dim=demo_obs.shape[-1],
            action_dim=demo_act.shape[-1],
            seed=FLAGS.seed,
            hidden_dim=FLAGS.dgn_hidden_dim,
            num_layers=FLAGS.dgn_num_layers,
            dropout_rate=FLAGS.dgn_dropout,
            lr=FLAGS.dgn_lr,
            weight_decay=FLAGS.dgn_weight_decay,
            batch_size=FLAGS.dgn_batch_size,
            epochs=FLAGS.dgn_epochs,
            grad_clip=1.0,     # Enable global norm gradient clipping
            #min_diag=1e-2,     # Increase minimum diagonal for numerical stability
        )
        print(f"DGN: {len(demo_obs)} prior (s, a) pairs, action dim {demo_act.shape[-1]}")

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        # DGN: (re)fit the covariance net with the latest actor mean. The first fit happens
        # at i == start_training, right before the first guided action.
        if (
            FLAGS.dgn
            and i >= FLAGS.start_training
            and (i - FLAGS.start_training) % FLAGS.dgn_update_interval == 0
        ):
            dgn_info = dgn.refit(agent, demo_obs, demo_act)
            for k, v in dgn_info.items():
                wandb.log({f"dgn/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i < FLAGS.start_training:
            action = env.action_space.sample()
        elif FLAGS.dgn:  # DGN: a = clip(mu_theta(s) + exp(-t/tau) * A_phi(s) z)
            mu = actor_mean(agent, observation)
            eps = noise_scale(i, FLAGS.dgn_anneal_tau) * dgn.noise(observation)
            action = np.clip(mu + eps, env.action_space.low, env.action_space.high)
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
            )

            batch = combine(offline_batch, online_batch)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)
                if FLAGS.dgn:  # DGN
                    wandb.log(
                        {"dgn/noise_scale": noise_scale(i, FLAGS.dgn_anneal_tau)},
                        step=i + FLAGS.pretrain_steps,
                    )

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
