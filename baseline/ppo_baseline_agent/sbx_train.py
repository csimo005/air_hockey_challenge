import os
from pathlib import Path

import jax
import numpy as np
from brax.envs.wrappers import training
from gymnasium import spaces
from sbx import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)
from utils.callbacks import SaveVecNormalizeCallback

from baseline.ppo_baseline_agent.modified_envs.env_defend import EnvDefend
from baseline.ppo_baseline_agent.modified_envs.env_hit import EnvHit
from baseline.ppo_baseline_agent.modified_envs.env_prepare import EnvPrepare
from baseline.ppo_baseline_agent.utils.brax_wrapper import (
    AutoResetWrapper,
    BraxSB3Wrapper,
)

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True " "--xla_gpu_enable_latency_hiding_scheduler=true "
)
# We encounter NaN values even with 'high' precision, so we set it to 'highest'
jax.config.update("jax_default_matmul_precision", "highest")

import wandb

ENVS = {"defend": EnvDefend, "hit": EnvHit, "prepare": EnvPrepare}


def make_environments(
    env_name: str, num_envs: int, num_eval_envs: int, gamma: float, horizon: int
):
    train_env = ENVS[env_name]()

    min_action = np.array(train_env.act_low)
    max_action = np.array(train_env.act_high)
    action_space = spaces.Box(min_action, max_action)

    train_env = training.EpisodeWrapper(train_env, horizon, 1)
    train_env = training.VmapWrapper(train_env, num_envs)
    train_env = AutoResetWrapper(train_env)

    train_env = BraxSB3Wrapper(train_env, seed=0, keep_infos=False)
    train_env.action_space = action_space
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, gamma=gamma)

    eval_env = ENVS[env_name]()
    eval_env = training.EpisodeWrapper(eval_env, horizon, 1)
    eval_env = training.VmapWrapper(eval_env, num_eval_envs)
    eval_env = AutoResetWrapper(eval_env)

    eval_env = BraxSB3Wrapper(eval_env, seed=1, keep_infos=False)
    eval_env.action_space = action_space
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, gamma=gamma, training=False, norm_reward=False)

    return train_env, eval_env


def setup_wandb(train_dir: Path, env_name: str):
    wandb.init(
        entity="atalaydonat",
        project="air_hockey_challenge",
        group=env_name,
        mode="disabled",
        dir=train_dir,
        name=train_dir.name,
        sync_tensorboard=True,
    )


if __name__ == "__main__":
    # Set the environment name and parameters
    env_name = "hit"
    num_envs = 4096
    num_eval_envs = 30
    gamma = 0.99
    horizon = 500
    run_name = "test"
    # Define the directory to save the model and normalization stats
    train_dir = (
        Path(
            "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints"
        )
        / env_name
        / run_name
    )

    if train_dir.exists():
        raise FileExistsError(
            f"Directory {train_dir} already exists. Please choose a different run name."
        )

    setup_wandb(train_dir, env_name)

    # Create training and evaluation environments
    train_env, eval_env = make_environments(
        env_name, num_envs, num_eval_envs, gamma, horizon
    )

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=10,
        batch_size=512,
        learning_rate=5e-5,
        gamma=gamma,
        n_epochs=10,
        seed=42,
        verbose=1,
        tensorboard_log=train_dir,
    )

    sync_envs_normalization(train_env, eval_env)

    save_vecnormalize_callback = SaveVecNormalizeCallback(save_path=train_dir)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=train_dir,
        log_path=train_dir,
        eval_freq=int(1e7 / num_envs),
        deterministic=True,
        render=False,
        callback_on_new_best=save_vecnormalize_callback,
        n_eval_episodes=num_eval_envs,
    )

    # Start training the model
    model.learn(
        total_timesteps=int(5e8),
        progress_bar=True,
        tb_log_name="run",
        callback=eval_callback,
    )
