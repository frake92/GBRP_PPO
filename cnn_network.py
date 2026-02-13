# cnn_network.py
import os
import time
from pathlib import Path

import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecTransposeImage,
    SubprocVecEnv,
)

# --- GBRP (optional) ---
# Ha nem kell GBRP, csak állítsd USE_GBRP = False-ra
from gbrp import GBRPCallback, GBRPConfig, GBRPVecEnvWrapper


def build_env(env_id: str, n_envs: int, seed: int):
    # Regisztráljuk az ALE env-eket (különben "Namespace ALE not found")
    gym.register_envs(ale_py)

    # Atari env párhuzamosan (gyorsabb, mint DummyVecEnv)
    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    # 4 frame stack (klasszikus Atari)
    env = VecFrameStack(env, n_stack=4)

    # CNN-hez channel-first (C,H,W)
    env = VecTransposeImage(env)

    return env


def main():
    # ------------ CONFIG ------------
    env_id = "ALE/Seaquest-v5"
    seed = 0

    # Atarihoz gyakran a 4 env jobb, mint 8 (CPU emulator bottleneck miatt)
    n_envs = 4

    total_timesteps = 2_000_000  # <- amit kértél
    use_gbrp = True              # <- True: PPO+GBRP, False: sima PPO

    # PPO hyperparam baseline (Atari-hoz oké)
    learning_rate = 2.5e-4
    n_steps = 256
    batch_size = 256
    n_epochs = 4

    # log dir
    run_dir = Path("runs") / f"seaquest_{'gbrp' if use_gbrp else 'ppo'}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------ ENV ------------
    env = build_env(env_id, n_envs=n_envs, seed=seed)

    # GBRP wrapper (csak ha kell)
    callback = None
    if use_gbrp:
        env = GBRPVecEnvWrapper(env)
        callback = GBRPCallback(GBRPConfig())

    # ------------ MODEL ------------
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        device="cuda",  # ROCm-nál is így
        tensorboard_log=str(run_dir / "tb"),
    )

    # Logger
    model.set_logger(configure(folder=str(run_dir), format_strings=["stdout", "csv"]))

    print("SB3 model device:", model.device)
    print("Logging to:", run_dir)

    # ------------ TRAIN ------------
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callback,
    )

    # ------------ SAVE ------------
    model.save(str(run_dir / "final_model"))
    print("Saved:", run_dir / "final_model")


if __name__ == "__main__":
    # (opcionális) CPU thread limit, néha segít Subproc mellett
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
