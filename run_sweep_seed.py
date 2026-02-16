"""Automate PPO + GBRP runs for a single seed.

Runs, in order, on a given seed:

- CoastRunner toy env (BoatRace-v0):
    - PPO
    - GBRP (group_size=5, baseline_window=3, E_len ON)
- BipedalWalker-v3:
    - PPO
    - GBRP (group_size=5, baseline_window=3, E_len OFF)
- Seaquest (ALE/Seaquest-v5, CNN policy):
    - PPO
    - GBRP (group_size=5, baseline_window=3, E_len OFF)

Usage example:

    python run_sweep_seed.py --seed 12345

This script will block and run each experiment sequentially. If any
run fails (non-zero exit), the script stops.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    """Run a subprocess, printing the command and stopping on error."""

    print("\n=== Running:", " ".join(cmd), "===", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO+GBRP sweep for a single seed.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use for all runs.")
    parser.add_argument(
        "--mlp-timesteps",
        type=int,
        default=2_000_000,
        help="Total env steps for MLP envs (BoatRace, BipedalWalker).",
    )
    parser.add_argument(
        "--seaquest-timesteps",
        type=int,
        default=2_000_000,
        help="Total env steps for Seaquest CNN run.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of vectorised envs for MLP runs (passed to network.py).",
    )

    args = parser.parse_args()

    seed = int(args.seed)
    seed_str = str(seed)
    py = sys.executable

    # Base layout mirrors existing runs/Seed*/ structure.
    # Each call lets network.py/cnn_network.py create a timestamped subdir.

    # ----- CoastRunner toy env (BoatRace-v0) -----
    coast_root = Path("runs") / f"Seed{seed_str}" / "CoastRunner-ToyEnvironment"

    # PPO
    _run(
        [
            py,
            "network.py",
            "--env",
            "BoatRace-v0",
            "--seed",
            seed_str,
            "--max-steps",
            str(args.mlp_timesteps),
            "--n-envs",
            str(args.n_envs),
            "--log-root",
            str(coast_root / "PPO"),
        ]
    )

    # GBRP (gs=5, bw=3, E_len ON)
    _run(
        [
            py,
            "network.py",
            "--env",
            "BoatRace-v0",
            "--seed",
            seed_str,
            "--max-steps",
            str(args.mlp_timesteps),
            "--n-envs",
            str(args.n_envs),
            "--gbrp",
            "--gbrp-group-size",
            "5",
            "--gbrp-baseline-window",
            "3",
            "--log-root",
            str(coast_root / "GBRP_gs5_bw3"),
        ]
    )

    # ----- BipedalWalker-v3 -----
    biped_root = Path("runs") / f"Seed{seed_str}" / "BipedalWalker-Gymnasium"

    # PPO
    _run(
        [
            py,
            "network.py",
            "--env",
            "BipedalWalker-v3",
            "--seed",
            seed_str,
            "--max-steps",
            str(args.mlp_timesteps),
            "--n-envs",
            str(args.n_envs),
            "--log-root",
            str(biped_root / "PPO"),
        ]
    )

    # GBRP (gs=5, bw=3, E_len OFF)
    _run(
        [
            py,
            "network.py",
            "--env",
            "BipedalWalker-v3",
            "--seed",
            seed_str,
            "--max-steps",
            str(args.mlp_timesteps),
            "--n-envs",
            str(args.n_envs),
            "--gbrp",
            "--gbrp-group-size",
            "5",
            "--gbrp-baseline-window",
            "3",
            "--gbrp-disable-e-len",
            "--log-root",
            str(biped_root / "GBRP_gs5_bw3_ElenOff"),
        ]
    )

    # ----- Seaquest (ALE/Seaquest-v5, CNN policy) -----
    sea_root = Path("runs") / f"Seed{seed_str}" / "Seaquest-ALEAtari"

    # PPO (no GBRP)
    _run(
        [
            py,
            "cnn_network.py",
            "--env",
            "ALE/Seaquest-v5",
            "--seed",
            seed_str,
            "--total-timesteps",
            str(args.seaquest_timesteps),
            "--n-envs",
            "4",
            "--log-root",
            str(sea_root / "PPO"),
        ]
    )

    # GBRP (gs=5, bw=3, E_len OFF)
    _run(
        [
            py,
            "cnn_network.py",
            "--env",
            "ALE/Seaquest-v5",
            "--seed",
            seed_str,
            "--total-timesteps",
            str(args.seaquest_timesteps),
            "--n-envs",
            "4",
            "--gbrp",
            "--gbrp-group-size",
            "5",
            "--gbrp-baseline-window",
            "3",
            "--gbrp-disable-e-len",
            "--log-root",
            str(sea_root / "GBRP_gs5_bw3_ElenOff"),
        ]
    )

    print("\nAll runs for seed", seed_str, "completed.")


if __name__ == "__main__":
    main()
