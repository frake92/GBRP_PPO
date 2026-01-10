from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _resolve_csv_path(csv_arg: Optional[str], run_dir_arg: str) -> Path:
    if csv_arg:
        p = Path(csv_arg)
        if p.is_dir():
            return p / "progress.csv"
        return p

    run_dir = Path(run_dir_arg)
    if run_dir.is_file():
        return run_dir

    direct = run_dir / "progress.csv"
    if direct.exists():
        return direct

    matches = sorted(run_dir.rglob("progress.csv"))
    if matches:
        return matches[0]

    any_csv = sorted(run_dir.rglob("*.csv"))
    if any_csv:
        return any_csv[0]

    raise FileNotFoundError(f"No CSV found in {run_dir}")


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1).mean()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _pick_x(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # Prefer SB3 standard axes.
    for key in ("time/total_timesteps", "time/iterations"):
        if key in df.columns:
            return _num(df, key), key
    return pd.Series(range(len(df)), index=df.index, dtype=float), "index"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SB3 progress.csv metrics (includes GBRP signals if present).")
    parser.add_argument("--csv", type=str, default=None, help="Path to SB3 progress.csv (or a run dir)")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/_buggy_smoketest",
        help="Run directory containing progress.csv (ignored if --csv provided)",
    )
    parser.add_argument("--window", type=int, default=10, help="Smoothing window (default: 10)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: <run_dir>/metrics.png)",
    )
    parser.add_argument("--show", action="store_true", help="Also show an interactive window")
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.csv, args.run_dir)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    x, x_label = _pick_x(df)
    window = max(1, int(args.window))

    metric_groups: list[dict[str, str]] = [
        {
            "rollout/ep_rew_mean": "Episode reward mean",
            "rollout/ep_len_mean": "Episode length mean",
            "rewards/raw_mean_step": "Raw reward mean (per step)",
            "rewards/modified_mean_step": "Modified reward mean (per step)",
            "time/fps": "FPS",
        },
        {
            "train/loss": "Total loss",
            "train/value_loss": "Value loss",
            "train/policy_gradient_loss": "Policy gradient loss",
            "train/entropy_loss": "Entropy loss (negative entropy)",
            "train/approx_kl": "Approx KL",
            "train/explained_variance": "Explained variance",
        },
        {
            # GBRP (optional)
            "gbrp/current_lambda": "GBRP current lambda",
            "gbrp/level1_active": "GBRP Level1 active",
            "gbrp/level2_active": "GBRP Level2 active",
            "gbrp/S1_count": "GBRP S1 count",
            "gbrp/S2_count": "GBRP S2 count",
            "gbrp/reward_grad": "GBRP reward gradient",
            "gbrp/reward_spike": "GBRP reward spike",
            "gbrp/episode_len_reward_coupling": "GBRP len-reward coupling",
            "gbrp/value_loss_instability": "GBRP value-loss instability",
            "gbrp/explained_variance": "GBRP explained variance (raw proxy)",
            "gbrp/entropy_oscillation": "GBRP entropy oscillation",
        },
    ]

    selected: list[tuple[str, str]] = []
    for group in metric_groups:
        for key, title in group.items():
            y = _num(df, key)
            if y.notna().any():
                selected.append((key, title))

    if not selected:
        raise ValueError(f"No known metrics found in CSV: {csv_path}")

    import math

    n_plots = len(selected)
    ncols = min(3, n_plots)
    nrows = int(math.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False)
    fig.suptitle(f"SB3 Metrics: {csv_path}", fontsize=14)

    ax_list = [ax for row in axs for ax in row]
    for ax, (key, title) in zip(ax_list, selected, strict=False):
        y = _num(df, key)

        ax.plot(x, y, alpha=0.35, linewidth=1.0, marker=".", markersize=3, label="raw")
        ax.plot(x, _smooth(y, window), linewidth=2.0, label=f"smooth({window})")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.2)

        if key in ("gbrp/level1_active", "gbrp/level2_active"):
            ax.set_ylim(-0.05, 1.05)

        if key == "train/entropy_loss":
            entropy = -y
            ax2 = ax.twinx()
            ax2.plot(x, _smooth(entropy, window), color="tab:green", alpha=0.6, linewidth=1.5)
            ax2.set_ylabel("entropy (approx)")

        ax.legend(loc="best", fontsize=8)

    for j in range(len(selected), len(ax_list)):
        ax_list[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = csv_path.parent / "metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    print(f"Read: {csv_path}")
    print(f"Wrote: {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
