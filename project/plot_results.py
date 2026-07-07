"""
Combines the per-algorithm reward curves (saved as .npy files by
reinforce/mujoco_reinforce.py, reinforce_baseline/mujoco_reinforce_baseline.py,
and a3c/mujoco_a3c.py) into the plots used in the report:

  1. One figure per algorithm: reward vs. episode, mean +/- std across
     the 5 Fibonacci seeds (same style as the tutorial's seaborn plot).
  2. One overlay figure comparing all three algorithms' mean curves
     (optionally smoothed with a rolling window) on the same axes.

Usage:
    python plot_results.py --results_dir results --out_dir plots
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 5)

ALGOS = {
    "reinforce": ("reinforce_all_seeds.npy", "REINFORCE"),
    "reinforce_baseline": ("reinforce_baseline_all_seeds.npy", "REINFORCE + Baseline"),
    "a3c": ("a3c_all_seeds.npy", "A3C"),
}


def load_all_seeds(results_dir: str, filename: str) -> np.ndarray | None:
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"[warn] missing {path}, skipping")
        return None
    return np.load(path)


def plot_single_algo(rewards_over_seeds: np.ndarray, title: str, out_path: str):
    df = pd.DataFrame(rewards_over_seeds).melt()
    df.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)

    sns.set(style="darkgrid", context="talk", palette="rainbow")
    plt.figure()
    sns.lineplot(x="episodes", y="reward", data=df).set(title=title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


def rolling_mean(x: np.ndarray, window: int = 50) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=1).mean().to_numpy()


def plot_overlay(all_results: dict, out_path: str, window: int = 50):
    plt.figure()
    sns.set(style="darkgrid", context="talk", palette="rainbow")

    for key, (arr, label) in all_results.items():
        if arr is None:
            continue
        mean_curve = arr.mean(axis=0)
        smoothed = rolling_mean(mean_curve, window=window)
        plt.plot(smoothed, label=label)

    plt.xlabel("episodes")
    plt.ylabel(f"reward (rolling mean, window={window})")
    plt.title("REINFORCE vs. REINFORCE+Baseline vs. A3C on InvertedPendulum-v4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="plots")
    parser.add_argument("--smoothing_window", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for key, (filename, title) in ALGOS.items():
        arr = load_all_seeds(args.results_dir, filename)
        all_results[key] = (arr, title)
        if arr is not None:
            plot_single_algo(arr, title + " for InvertedPendulum-v4",
                              os.path.join(args.out_dir, f"{key}.png"))

    plot_overlay(all_results, os.path.join(args.out_dir, "comparison_overlay.png"),
                 window=args.smoothing_window)


if __name__ == "__main__":
    main()
