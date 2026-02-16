from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from gbrp import GBRPCallback, GBRPConfig, GBRPVecEnvWrapper


@dataclass(frozen=True)
class TrainingConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0

    n_envs: int = 1
    n_steps: int = 500

    policy_updates: int = 500

    total_timesteps_override: int | None = None

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    net_arch: tuple[int, ...] = (64, 64)

    gbrp_enabled: bool = False
    gbrp_cfg: GBRPConfig = field(default_factory=GBRPConfig)

    @property
    def timesteps_per_update(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def total_timesteps(self) -> int:
        if self.total_timesteps_override is not None:
            return self.total_timesteps_override
        return self.timesteps_per_update * self.policy_updates

    @property
    def expected_policy_updates(self) -> int:

        return (self.total_timesteps + self.timesteps_per_update - 1) // self.timesteps_per_update


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def run_dir(self) -> Path:
        return self.root

    @property
    def sb3_log_dir(self) -> Path:
        return self.run_dir

    @property
    def progress_csv(self) -> Path:
        return self.run_dir / "progress.csv"


def _timestamp_run_name(prefix: str = "ppo_mlp") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def build_env(cfg: TrainingConfig) -> Any:

    try:
        import register_envs  # noqa: F401
    except Exception:
        pass
    env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)
    if cfg.gbrp_enabled:
        env = GBRPVecEnvWrapper(env)
    return env


def build_model(cfg: TrainingConfig, env: Any, log_dir: Path) -> PPO:
    policy_kwargs = {"net_arch": list(cfg.net_arch)}

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg.seed,
        device="auto",
        tensorboard_log=str(log_dir / "tb"),
    )

    new_logger = configure(folder=str(log_dir), format_strings=["stdout", "csv"])
    model.set_logger(new_logger)
    return model


def train(cfg: TrainingConfig, outputs: OutputPaths) -> None:
    outputs.run_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(cfg)
    model = build_model(cfg, env, outputs.sb3_log_dir)

    print(
        f"Training for total_timesteps={cfg.total_timesteps} (~{cfg.expected_policy_updates} policy updates; "
        f"timesteps_per_update={cfg.timesteps_per_update})."
    )
    callback = GBRPCallback(cfg.gbrp_cfg) if cfg.gbrp_enabled else None
    model.learn(total_timesteps=cfg.total_timesteps, log_interval=1, progress_bar=True, callback=callback)

    model.save(str(outputs.run_dir / "final_model"))

    print("\nDone.")
    print(f"Run dir: {outputs.run_dir}")
    print(f"Per-update metrics CSV: {outputs.progress_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single-MLP PPO agent and write per-update metrics to CSV.")
    p.add_argument("--env", default=TrainingConfig.env_id)
    p.add_argument("--seed", type=int, default=TrainingConfig.seed)
    p.add_argument("--policy-updates", type=int, default=TrainingConfig.policy_updates)
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        dest="total_timesteps",
        help="If set, overrides --policy-updates and trains for this many env steps.",
    )
    p.add_argument(
        "--max-steps",
        "--max_steps",
        type=int,
        default=None,
        dest="total_timesteps",
        help="Alias for --total-timesteps (Unity ML-Agents style: total training steps).",
    )
    p.add_argument("--n-steps", type=int, default=TrainingConfig.n_steps)
    p.add_argument("--n-envs", type=int, default=TrainingConfig.n_envs)

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

    p.add_argument("--log-root", default="runs")
    p.add_argument("--run-name", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gbrp_cfg = GBRPConfig(
        group_size=int(args.gbrp_group_size),
        baseline_window=int(args.gbrp_baseline_window),
        n1=int(args.gbrp_n1),
        n2=int(args.gbrp_n2),
        tau_H=float(args.gbrp_tau_h),
        enable_E_len=not bool(getattr(args, "gbrp_disable_e_len", False)),
    )

    cfg = TrainingConfig(
        env_id=args.env,
        seed=args.seed,
        policy_updates=args.policy_updates,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        total_timesteps_override=args.total_timesteps,
        gbrp_enabled=bool(args.gbrp),
        gbrp_cfg=gbrp_cfg,
    )

    run_name = args.run_name.strip() or _timestamp_run_name("ppo_mlp")
    outputs = OutputPaths(root=Path(args.log_root) / run_name)

    if cfg.total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")

    train(cfg, outputs)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
