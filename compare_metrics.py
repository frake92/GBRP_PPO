from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1).mean()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _pick_x(df: pd.DataFrame) -> tuple[pd.Series, str]:
    for key in ("time/total_timesteps", "time/iterations"):
        if key in df.columns:
            return _num(df, key), key
    return pd.Series(range(len(df)), index=df.index, dtype=float), "index"


def _resolve_csv_path(path_or_dir: str) -> Path:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "progress.csv"
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SB3 progress.csv runs (PPO vs GBRP).")
    parser.add_argument("--ppo", type=str, default="runs/BoatHackPPO2/progress.csv", help="PPO progress.csv (or run dir)")
    parser.add_argument(
        "--gbrp",
        type=str,
        default="runs/BoatHackPPOGBRP_2M/progress.csv",
        help="GBRP progress.csv (or run dir)",
    )
    parser.add_argument("--window", type=int, default=25, help="Smoothing window (default: 25)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: runs/compare_PPO_vs_GBRP.png)",
    )
    parser.add_argument("--show", action="store_true", help="Also show an interactive window")
    args = parser.parse_args()

    ppo_csv = _resolve_csv_path(args.ppo)
    gbrp_csv = _resolve_csv_path(args.gbrp)

    ppo = pd.read_csv(ppo_csv)
    gbrp = pd.read_csv(gbrp_csv)
    ppo.columns = [c.strip() for c in ppo.columns]
    gbrp.columns = [c.strip() for c in gbrp.columns]

    x_ppo, x_label_ppo = _pick_x(ppo)
    x_gbrp, x_label_gbrp = _pick_x(gbrp)

    # Prefer a common x axis if possible.
    if x_label_ppo == x_label_gbrp:
        x_label = x_label_ppo
        x_ppo_plot = x_ppo
        x_gbrp_plot = x_gbrp
    else:
        x_label = f"{x_label_ppo} / {x_label_gbrp}"
        x_ppo_plot = x_ppo
        x_gbrp_plot = x_gbrp

    window = max(1, int(args.window))

    # (key, title, transform)
    def entropy(df: pd.DataFrame) -> pd.Series:
        # SB3 logs entropy_loss as negative entropy; plot entropy.
        return -_num(df, "train/entropy_loss")

    plots: list[tuple[str, str, callable[[pd.DataFrame], pd.Series]]] = [
        ("rollout/ep_rew_mean", "Episode reward mean", lambda df: _num(df, "rollout/ep_rew_mean")),
        ("rollout/ep_len_mean", "Episode length mean", lambda df: _num(df, "rollout/ep_len_mean")),
        ("train/approx_kl", "Approx KL", lambda df: _num(df, "train/approx_kl")),
        ("train/entropy_loss", "Entropy (approx)", entropy),
        ("train/value_loss", "Value loss", lambda df: _num(df, "train/value_loss")),
        ("train/explained_variance", "Explained variance", lambda df: _num(df, "train/explained_variance")),
        ("gbrp/current_lambda", "GBRP lambda", lambda df: _num(df, "gbrp/current_lambda")),
        ("gbrp/level1_active", "GBRP level1 active", lambda df: _num(df, "gbrp/level1_active")),
    ]

    selected: list[tuple[str, str, callable[[pd.DataFrame], pd.Series]]] = []
    for key, title, fn in plots:
        has_any = (key in ppo.columns) or (key in gbrp.columns) or (key == "train/entropy_loss")
        if not has_any:
            continue
        # If it is entropy, it exists only if entropy_loss exists.
        if key == "train/entropy_loss" and ("train/entropy_loss" not in ppo.columns) and ("train/entropy_loss" not in gbrp.columns):
            continue
        selected.append((key, title, fn))

    if not selected:
        raise ValueError("No comparable metrics found.")

    import math

    n_plots = len(selected)
    ncols = 2
    nrows = int(math.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(12.5, 3.6 * nrows), squeeze=False)
    fig.suptitle(
        f"PPO vs GBRP metrics\nPPO: {ppo_csv} | GBRP: {gbrp_csv}",
        fontsize=13,
    )

    ax_list = [ax for row in axs for ax in row]
    for ax, (key, title, fn) in zip(ax_list, selected, strict=False):
        y_ppo = fn(ppo)
        y_gbrp = fn(gbrp)

        if key in ppo.columns or key == "train/entropy_loss":
            ax.plot(x_ppo_plot, _smooth(y_ppo, window), linewidth=2.0, label="PPO")
        if key in gbrp.columns or key == "train/entropy_loss":
            ax.plot(x_gbrp_plot, _smooth(y_gbrp, window), linewidth=2.0, label="GBRP")

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.2)

        if key in ("gbrp/level1_active",):
            ax.set_ylim(-0.05, 1.05)

        ax.legend(loc="best", fontsize=9)

    for j in range(len(selected), len(ax_list)):
        ax_list[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = Path(args.out) if args.out else Path("runs/compare_PPO_vs_GBRP.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    print(f"Read PPO:  {ppo_csv}")
    print(f"Read GBRP: {gbrp_csv}")
    print(f"Wrote:     {out_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
