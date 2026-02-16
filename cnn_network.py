"""CNN-based PPO training script for Atari Seaquest with optional GBRP.

Now supports CLI arguments so it can be used in automated sweeps.
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Seaquest training.

    Designed to mirror the most important knobs from network.py
    so we can orchestrate PPO vs GBRP runs in a consistent way.
    """

    p = argparse.ArgumentParser(description="Train PPO/CNN on Seaquest with optional GBRP.")
    p.add_argument("--env", default="ALE/Seaquest-v5")
    p.add_argument("--seed", type=int, default=0)

    # Training time
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=4)

    # GBRP on/off and core params
    p.add_argument("--gbrp", action="store_true", help="Enable GBRP reward-protection layer")
    p.add_argument("--gbrp-group-size", type=int, default=GBRPConfig.group_size)
    p.add_argument("--gbrp-baseline-window", type=int, default=GBRPConfig.baseline_window)
    p.add_argument("--gbrp-n1", type=int, default=GBRPConfig.n1)
    p.add_argument("--gbrp-n2", type=int, default=GBRPConfig.n2)
    p.add_argument("--gbrp-tau-h", type=float, default=GBRPConfig.tau_H, help="Entropy oscillation threshold")
    p.add_argument(
        "--gbrp-disable-e-len",
        action="store_true",
        help="Disable GBRP E_len (episode length / reward coupling) signal.",
    )

    # Logging layout similar to network.py
    p.add_argument("--log-root", default="runs")
    p.add_argument("--run-name", default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    env_id = args.env
    seed = int(args.seed)
    n_envs = int(args.n_envs)
    total_timesteps = int(args.total_timesteps)

    use_gbrp = bool(args.gbrp)

    # PPO hyperparam baseline (Atari-hoz oké)
    learning_rate = 2.5e-4
    n_steps = 256
    batch_size = 256
    n_epochs = 4

    # log dir: mirror network.py style (log_root / run_name)
    run_name = args.run_name.strip()
    if not run_name:
        prefix = "seaquest_gbrp" if use_gbrp else "seaquest_ppo"
        run_name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_dir = Path(args.log_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------ ENV ------------
    env = build_env(env_id, n_envs=n_envs, seed=seed)

    # GBRP wrapper (csak ha kell)
    callback = None
    if use_gbrp:
        env = GBRPVecEnvWrapper(env)
        gbrp_cfg = GBRPConfig(
            group_size=int(args.gbrp_group_size),
            baseline_window=int(args.gbrp_baseline_window),
            n1=int(args.gbrp_n1),
            n2=int(args.gbrp_n2),
            tau_H=float(args.gbrp_tau_h),
            enable_E_len=not bool(getattr(args, "gbrp_disable_e_len", False)),
        )
        callback = GBRPCallback(gbrp_cfg)

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
