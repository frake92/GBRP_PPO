#!/usr/bin/env python3
"""Overlay plots across seeds for PPO/GBRP progress CSVs.

You provide the progress CSV relative paths (plus env/algo/seed labels), and the script
creates, for each environment separately, per-algorithm plots where all provided seeds
appear on the same figure.

Required plots (each saved as a separate PNG):
- reward + episode length (one figure: 2 subplots)
- entropy
- value loss
- (GBRP only) lambda
- (GBRP only) level1_active
- (GBRP only) level2_active

Simple usage (no CLI args):

1) Edit the RUNS list near the top of this file.
2) Run:

    python3 plot_seed_overlays.py

CLI usage (optional):

  python3 plot_seed_overlays.py --outdir seed_overlay_plots \
    --run BipedalWalker ppo 0 runs/Seed0/BipedalWalker-Gymnasium/PPO/ppo_mlp_*/ppobip0progress.csv \
    --run BipedalWalker ppo 42 runs/Seed42/BipedalWalker-Gymnasium/BipedalWalkerPPO42/ppoProgress.csv \
    --run BipedalWalker gbrp 0 runs/Seed0/BipedalWalker-Gymnasium/GBRP_gs5_bw3_ElenOff/ppo_mlp_*/gbrpbip0progress.csv

You can also pass --config with a JSON file:

  {
    "runs": [
      {"env": "Seaquest", "algo": "gbrp", "seed": 0, "path": "runs/Seed0/Seaquest-ALEAtari/seaquest_gbrp/gbrpprogress.csv"}
    ]
  }

"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    env: str
    algo: str
    seed: int
    path: str


# -------------------------
# EDIT THIS SECTION
# -------------------------
# Add your runs here. Paths are workspace-relative by default.
# You can use globs like "runs/Seed0/.../ppo_mlp_*/progress.csv".
#
# Example:
# RUNS = [
#   RunSpec(env="Seaquest", algo="ppo", seed=0, path="runs/Seed0/Seaquest-ALEAtari/seaquest_ppo/ppoprogress.csv"),
#   RunSpec(env="Seaquest", algo="gbrp", seed=0, path="runs/Seed0/Seaquest-ALEAtari/seaquest_gbrp/gbrpprogress.csv"),
# ]
RUNS: list[RunSpec] = [

    RunSpec(
        env="Seaquest",
        algo="ppo",
        seed=0,
        path="runs/Seed0/Seaquest-ALEAtari/seaquest_ppo/ppoprogress.csv",
    ),
    RunSpec(
        env="Seaquest",
        algo="ppo",
        seed=42,
        path="runs/Seed42/Seaquest-ALEAtari/PPO/seaquest_ppo_20260216_142725/progress.csv",
    ),
    RunSpec(
        env="Seaquest",
        algo="ppo",
        seed=12345,
        path="runs/Seed12345/BipedalWalker-Gymnasium/PPO/ppo_mlp_20260216_022647/ppobipedalprogress.csv",
    ),
    RunSpec(
        env="Seaquest",
        algo="ppo",
        seed=101010,
        path="runs/Seed101010/Seaquest-ALEAtari/PPO/seaquest_ppo_20260216_115231/ppoAtari101progress.csv",
    ),
    
]

# Rolling-mean smoothing window for value loss curves.
# Set to 1 to disable smoothing.
VALUE_LOSS_SMOOTH_WINDOW: int = 10

# Rolling-mean smoothing window for GBRP lambda.
# Set to 1 to disable smoothing.
LAMBDA_SMOOTH_WINDOW: int = 10

# Rolling-mean smoothing window for GBRP level1_active.
# Set to 1 to disable smoothing (plots as step function).
LEVEL1_ACTIVE_SMOOTH_WINDOW: int = 10


def _sanitize_slug(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9._-]", "_", text)
    return text


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    # Most of these progress files are comma-separated with a header row.
    # We keep default dtype inference; missing values are ok.
    return pd.read_csv(path)


def _resolve_path_maybe_glob(root: Path, spec_path: str) -> Path:
    p = (root / spec_path).expanduser()

    # If it has glob characters, expand it.
    if any(ch in spec_path for ch in ("*", "?", "[")):
        matches = [Path(m) for m in glob.glob(str(p))]
        if not matches:
            raise FileNotFoundError(f"Glob matched nothing: {spec_path}")
        if len(matches) == 1:
            return matches[0]

        # Pick most recently modified match (common when there are multiple timestamped runs).
        matches.sort(key=lambda mp: mp.stat().st_mtime, reverse=True)
        print(
            f"[plot_seed_overlays] WARNING: glob '{spec_path}' matched {len(matches)} files; "
            f"using newest: {matches[0]}"
        )
        return matches[0]

    return p


def _choose_x(df: pd.DataFrame) -> tuple[str, pd.Series]:
    for col in ("time/total_timesteps", "total_timesteps", "timesteps", "step"):
        if col in df.columns:
            return col, df[col]
    # Fallback: integer index
    return "index", pd.Series(range(len(df)))


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _smooth_series(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    # Keep alignment with original index; tolerate NaNs.
    return y.rolling(window=window, min_periods=1).mean()


def _plot_metric_overlay(
    runs: list[tuple[RunSpec, pd.DataFrame]],
    metric_col: str,
    title: str,
    ylabel: str,
    outpath: Path,
    *,
    as_step: bool = False,
    smooth_window: int = 1,
    y_multiplier: float = 1.0,
) -> None:
    plt.figure(figsize=(10, 6))

    any_plotted = False
    for spec, df in runs:
        x_name, x = _choose_x(df)
        if metric_col not in df.columns:
            continue
        y = _smooth_series(df[metric_col], smooth_window) * y_multiplier

        # Drop NaNs to avoid broken lines.
        valid = y.notna() & x.notna()
        if valid.sum() == 0:
            continue

        if as_step:
            plt.step(x[valid], y[valid], where="post", label=f"seed {spec.seed}")
        else:
            plt.plot(x[valid], y[valid], label=f"seed {spec.seed}")
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


def _plot_reward_and_len(
    runs: list[tuple[RunSpec, pd.DataFrame]],
    reward_col: str,
    ep_len_col: str,
    title: str,
    outpath: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    any_reward = False
    any_len = False

    for spec, df in runs:
        _, x = _choose_x(df)

        if reward_col in df.columns:
            y = df[reward_col]
            valid = y.notna() & x.notna()
            if valid.sum() > 0:
                ax1.plot(x[valid], y[valid], label=f"seed {spec.seed}")
                any_reward = True

        if ep_len_col in df.columns:
            y = df[ep_len_col]
            valid = y.notna() & x.notna()
            if valid.sum() > 0:
                ax2.plot(x[valid], y[valid], label=f"seed {spec.seed}")
                any_len = True

    if not (any_reward or any_len):
        plt.close(fig)
        return

    fig.suptitle(title)

    ax1.set_ylabel("reward")
    ax1.grid(True, alpha=0.3)
    if any_reward:
        ax1.legend()

    ax2.set_xlabel("timesteps")
    ax2.set_ylabel("episode length")
    ax2.grid(True, alpha=0.3)
    if any_len:
        ax2.legend()

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_two_metrics_overlay(
    runs: list[tuple[RunSpec, pd.DataFrame]],
    *,
    top_col: str,
    top_ylabel: str,
    top_as_step: bool = False,
    top_smooth_window: int = 1,
    top_y_multiplier: float = 1.0,
    bottom_col: str,
    bottom_ylabel: str,
    bottom_as_step: bool = False,
    bottom_smooth_window: int = 1,
    bottom_y_multiplier: float = 1.0,
    title: str,
    outpath: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    any_top = False
    any_bottom = False

    for spec, df in runs:
        _, x = _choose_x(df)

        if top_col in df.columns:
            y = _smooth_series(df[top_col], top_smooth_window) * top_y_multiplier
            valid = y.notna() & x.notna()
            if valid.sum() > 0:
                if top_as_step:
                    ax1.step(x[valid], y[valid], where="post", label=f"seed {spec.seed}")
                else:
                    ax1.plot(x[valid], y[valid], label=f"seed {spec.seed}")
                any_top = True

        if bottom_col in df.columns:
            y = _smooth_series(df[bottom_col], bottom_smooth_window) * bottom_y_multiplier
            valid = y.notna() & x.notna()
            if valid.sum() > 0:
                if bottom_as_step:
                    ax2.step(x[valid], y[valid], where="post", label=f"seed {spec.seed}")
                else:
                    ax2.plot(x[valid], y[valid], label=f"seed {spec.seed}")
                any_bottom = True

    if not (any_top or any_bottom):
        plt.close(fig)
        return

    fig.suptitle(title)

    ax1.set_ylabel(top_ylabel)
    ax1.grid(True, alpha=0.3)
    if any_top:
        ax1.legend()

    ax2.set_xlabel("timesteps")
    ax2.set_ylabel(bottom_ylabel)
    ax2.grid(True, alpha=0.3)
    if any_bottom:
        ax2.legend()

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _load_runs(run_specs: list[RunSpec], root: Path) -> list[tuple[RunSpec, pd.DataFrame]]:
    loaded: list[tuple[RunSpec, pd.DataFrame]] = []
    for spec in run_specs:
        p = _resolve_path_maybe_glob(root, spec.path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = _read_csv(p)
        loaded.append((spec, df))
    return loaded


def _group_by_env_algo(runs: list[RunSpec]) -> dict[tuple[str, str], list[RunSpec]]:
    grouped: dict[tuple[str, str], list[RunSpec]] = {}
    for r in runs:
        key = (r.env, r.algo.lower())
        grouped.setdefault(key, []).append(r)

    # Stable seed ordering
    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda s: s.seed)
    return grouped


def _parse_config_json(path: Path) -> list[RunSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("runs")
    if not isinstance(items, list):
        raise ValueError("Config JSON must contain a top-level 'runs' list")

    out: list[RunSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each run in config must be an object")
        out.append(
            RunSpec(
                env=str(item["env"]),
                algo=str(item["algo"]),
                seed=int(item["seed"]),
                path=str(item["path"]),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overlay PPO/GBRP metrics across seeds and save plots (PNG)."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="seed_overlay_plots",
        help="Output directory for generated PNGs.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Workspace root used to resolve relative CSV paths.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file with {runs:[{env,algo,seed,path}, ...]}.",
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("ENV", "ALGO", "SEED", "REL_PATH"),
        help="Add one run spec. Repeat --run multiple times.",
    )

    args = parser.parse_args()

    root = Path(args.root) if args.root is not None else Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    run_specs: list[RunSpec] = []

    if args.config is not None:
        run_specs.extend(_parse_config_json(Path(args.config)))

    if args.run:
        for env, algo, seed, rel_path in args.run:
            run_specs.append(RunSpec(env=env, algo=algo, seed=int(seed), path=rel_path))

    if not run_specs:
        run_specs = list(RUNS)

    if not run_specs:
        print(
            "No runs provided. Either:\n"
            "- edit RUNS in plot_seed_overlays.py, or\n"
            "- pass --run ENV ALGO SEED REL_PATH (repeatable), or\n"
            "- pass --config config.json\n"
        )
        return 2

    grouped = _group_by_env_algo(run_specs)

    for (env, algo), specs in grouped.items():
        loaded = _load_runs(specs, root=root)

        # Column mapping (tries a few alternatives to be robust across files)
        reward_col = _first_existing_column(
            loaded[0][1],
            (
                "rollout/ep_rew_mean",
                "rewards/modified_mean_step",
                "rewards/raw_mean_step",
                "ep_rew_mean",
            ),
        )
        ep_len_col = _first_existing_column(
            loaded[0][1],
            ("rollout/ep_len_mean", "ep_len_mean"),
        )
        entropy_col = _first_existing_column(
            loaded[0][1],
            ("train/entropy_loss", "entropy_loss", "train/entropy"),
        )
        value_loss_col = _first_existing_column(
            loaded[0][1],
            ("train/value_loss", "value_loss"),
        )

        # Some runs might have columns that others don't; we pick based on first file,
        # but plotting functions will skip per-run if missing.
        env_slug = _sanitize_slug(env)
        algo_slug = _sanitize_slug(algo)
        base = outdir / env_slug / algo_slug

        if reward_col and ep_len_col:
            _plot_reward_and_len(
                loaded,
                reward_col=reward_col,
                ep_len_col=ep_len_col,
                title=f"{env} | {algo.upper()} | Reward & Episode length (overlay by seed)",
                outpath=base / "reward_and_ep_len.png",
            )

        if entropy_col and value_loss_col:
            _plot_two_metrics_overlay(
                loaded,
                top_col=entropy_col,
                top_ylabel="entropy",
                top_y_multiplier=-1.0,
                bottom_col=value_loss_col,
                bottom_ylabel="value loss",
                bottom_smooth_window=VALUE_LOSS_SMOOTH_WINDOW,
                title=f"{env} | {algo.upper()} | Entropy & Value loss (overlay by seed)",
                outpath=base / "entropy_and_value_loss.png",
            )
        else:
            if entropy_col:
                _plot_metric_overlay(
                    loaded,
                    metric_col=entropy_col,
                    title=f"{env} | {algo.upper()} | Entropy (overlay by seed)",
                    ylabel="entropy",
                    outpath=base / "entropy.png",
                    y_multiplier=-1.0,
                )

            if value_loss_col:
                _plot_metric_overlay(
                    loaded,
                    metric_col=value_loss_col,
                    title=f"{env} | {algo.upper()} | Value loss (overlay by seed)",
                    ylabel="value loss",
                    outpath=base / "value_loss.png",
                    smooth_window=VALUE_LOSS_SMOOTH_WINDOW,
                )

        if algo.lower() == "gbrp":
            lambda_col = _first_existing_column(
                loaded[0][1],
                ("gbrp/current_lambda", "current_lambda", "gbrp/lambda"),
            )
            level1_col = _first_existing_column(
                loaded[0][1],
                ("gbrp/level1_active", "level1_active"),
            )
            level2_col = _first_existing_column(
                loaded[0][1],
                ("gbrp/level2_active", "level2_active"),
            )

            if lambda_col:
                _plot_metric_overlay(
                    loaded,
                    metric_col=lambda_col,
                    title=f"{env} | GBRP | Lambda (overlay by seed)",
                    ylabel="lambda",
                    outpath=base / "lambda.png",
                    smooth_window=LAMBDA_SMOOTH_WINDOW,
                )

            if level1_col:
                _plot_metric_overlay(
                    loaded,
                    metric_col=level1_col,
                    title=f"{env} | GBRP | Level 1 active (overlay by seed)",
                    ylabel="active",
                    outpath=base / "level1_active.png",
                    as_step=LEVEL1_ACTIVE_SMOOTH_WINDOW <= 1,
                    smooth_window=LEVEL1_ACTIVE_SMOOTH_WINDOW,
                )

            if level2_col:
                _plot_metric_overlay(
                    loaded,
                    metric_col=level2_col,
                    title=f"{env} | GBRP | Level 2 active (overlay by seed)",
                    ylabel="active",
                    outpath=base / "level2_active.png",
                    as_step=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
