from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_RESULTS = {
    "caltech": {"tda": 0.9424, "freetta": 0.9463},
    "dtd": {"tda": 0.4740, "freetta": 0.4696},
    "eurosat": {"tda": 0.5800, "freetta": 0.6293},
    "imagenet": {"tda": 0.6467, "freetta": 0.6492},
    "pets": {"tda": 0.8863, "freetta": 0.9011},
}


def plot_paper_vs_repo(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.18

    paper_tda = [PAPER_RESULTS[d]["tda"] for d in datasets]
    paper_freetta = [PAPER_RESULTS[d]["freetta"] for d in datasets]
    repo_tda = summary_df["tda_acc"].tolist()
    repo_freetta = summary_df["freetta_acc"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, paper_tda, width=width, label="Paper TDA", color="#4c72b0")
    ax.bar(x - 0.5 * width, paper_freetta, width=width, label="Paper FreeTTA", color="#55a868")
    ax.bar(x + 0.5 * width, repo_tda, width=width, label="Repo TDA", color="#9ecae9")
    ax.bar(x + 1.5 * width, repo_freetta, width=width, label="Repo FreeTTA", color="#a1d99b")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Paper Expectation vs Repo Reproduction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gap_breakdown(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, summary_df["tda_minus_clip"], width=width, label="TDA - CLIP", color="#4c72b0")
    ax.bar(x + width / 2, summary_df["freetta_minus_clip"], width=width, label="FreeTTA - CLIP", color="#55a868")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy Gain")
    ax.set_title("How Much Each Method Improves Over CLIP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_flip_quality(flip_df: pd.DataFrame, output_path: Path) -> None:
    pivot_bfp = flip_df.pivot(index="dataset", columns="method", values="beneficial_flip_precision")
    pivot_hfr = flip_df.pivot(index="dataset", columns="method", values="harmful_flip_rate_on_clip_correct")
    datasets = pivot_bfp.index.tolist()
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, pivot_bfp["tda"], width=width, label="TDA BFP", color="#4c72b0")
    ax.bar(x - 0.5 * width, pivot_bfp["freetta"], width=width, label="FreeTTA BFP", color="#55a868")
    ax.bar(x + 0.5 * width, pivot_hfr["tda"], width=width, label="TDA HFR", color="#9ecae9")
    ax.bar(x + 1.5 * width, pivot_hfr["freetta"], width=width, label="FreeTTA HFR", color="#a1d99b")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Rate")
    ax.set_title("How Each Method Changes CLIP Predictions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_statistics_drift(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x, summary_df["freetta_final_mu_drift"], color="#55a868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Final Mean Drift")
    axes[0].set_title("FreeTTA Statistic Change")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary_df["tda_final_positive_cache_size"], color="#4c72b0", label="Positive cache")
    axes[1].bar(x, summary_df["tda_final_negative_cache_size"], bottom=summary_df["tda_final_positive_cache_size"], color="#9ecae9", label="Negative cache")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Cache Size")
    axes[1].set_title("TDA Final Memory State")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-friendly figures from saved outputs")
    parser.add_argument("--comparative-dir", default="outputs/comparative_analysis")
    parser.add_argument("--final-dir", default="outputs/final_method_suite")
    parser.add_argument("--output-dir", default="outputs/presentation_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparative_dir = Path(args.comparative_dir)
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics = pd.read_csv(comparative_dir / "summary_metrics.csv")
    flip_metrics = pd.read_csv(comparative_dir / "flip_metrics.csv")
    summary_table = pd.read_csv(final_dir / "summary_table.csv")

    plot_paper_vs_repo(summary_table, output_dir / "paper_vs_repo.png")
    plot_gap_breakdown(summary_table, output_dir / "gain_over_clip.png")
    plot_flip_quality(flip_metrics, output_dir / "clip_change_behavior.png")
    plot_statistics_drift(summary_metrics, output_dir / "internal_state_summary.png")


if __name__ == "__main__":
    main()
