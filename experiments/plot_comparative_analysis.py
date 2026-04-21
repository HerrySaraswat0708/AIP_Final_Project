from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, summary_df["clip_acc"], width=width, label="CLIP")
    ax.bar(x, summary_df["tda_acc"], width=width, label="TDA")
    ax.bar(x + width, summary_df["freetta_acc"], width=width, label="FreeTTA")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Across Datasets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gain_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    datasets = summary_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, summary_df["tda_gain_vs_clip"], width=width, label="TDA - CLIP")
    ax.bar(x + width / 2, summary_df["freetta_gain_vs_clip"], width=width, label="FreeTTA - CLIP")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy Gain")
    ax.set_title("Method Gain Over CLIP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_geometry(geometry_df: pd.DataFrame, output_path: Path) -> None:
    datasets = geometry_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, geometry_df["oracle_centroid_acc"], width=width, label="Centroid Oracle")
    axes[0].bar(x + width / 2, geometry_df["oracle_1nn_acc"], width=width, label="1-NN Oracle")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Oracle Accuracy")
    axes[0].set_title("Geometry Probe")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    colors = ["#2f7ed8" if value >= 0 else "#d84a4a" for value in geometry_df["geometry_alignment_score"]]
    axes[1].bar(x, geometry_df["geometry_alignment_score"], color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Centroid - 1NN")
    axes[1].set_title("Geometry Alignment Score")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_flip_metrics(flip_df: pd.DataFrame, output_path: Path) -> None:
    pivot_change = flip_df.pivot(index="dataset", columns="method", values="change_rate").reset_index()
    pivot_precision = flip_df.pivot(index="dataset", columns="method", values="beneficial_flip_precision").reset_index()
    datasets = pivot_change["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, pivot_change["tda"], width=width, label="TDA")
    axes[0].bar(x + width / 2, pivot_change["freetta"], width=width, label="FreeTTA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Change Rate")
    axes[0].set_title("Prediction Change Rate")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, pivot_precision["tda"], width=width, label="TDA")
    axes[1].bar(x + width / 2, pivot_precision["freetta"], width=width, label="FreeTTA")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Beneficial Flip Precision")
    axes[1].set_title("Flip Quality")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_disagreement(disagreement_df: pd.DataFrame, output_path: Path) -> None:
    datasets = disagreement_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, disagreement_df["disagreement_rate"], width=0.5, color="#7aa6c2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("TDA vs FreeTTA Disagreement Rate")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, disagreement_df["tda_acc_on_disagreement"], width=width, label="TDA")
    axes[1].bar(x + width / 2, disagreement_df["freetta_acc_on_disagreement"], width=width, label="FreeTTA")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy on Disagreement Samples")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_latency(latency_df: pd.DataFrame, curves_df: pd.DataFrame, output_path: Path) -> None:
    datasets = latency_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, latency_df["tda_break_even_ratio"], width=width, label="TDA vs CLIP")
    axes[0].bar(x + width / 2, latency_df["freetta_break_even_ratio"].fillna(1.0), width=width, label="FreeTTA vs CLIP")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Break-even Ratio")
    axes[0].set_title("How Long Until Each Method Helps")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for dataset, group in curves_df.groupby("dataset", observed=True):
        axes[1].plot(group["progress_ratio"], group["rolling_freetta_vs_tda"], label=dataset, linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Progress Ratio")
    axes[1].set_ylabel("Rolling FreeTTA - TDA")
    axes[1].set_title("Late-Stream Advantage")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_difficulty(difficulty_df: pd.DataFrame, output_path: Path) -> None:
    datasets = difficulty_df["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4 * len(datasets)), squeeze=False)

    for ax, dataset in zip(axes[:, 0], datasets):
        group = difficulty_df[difficulty_df["dataset"] == dataset].copy()
        x = np.arange(len(group))
        width = 0.25
        ax.bar(x - width, group["clip_acc"], width=width, label="CLIP")
        ax.bar(x, group["tda_acc"], width=width, label="TDA")
        ax.bar(x + width, group["freetta_acc"], width=width, label="FreeTTA")
        ax.set_xticks(x)
        ax.set_xticklabels(group["difficulty_bin"].astype(str).tolist())
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{dataset}: Accuracy by Difficulty Bin")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from outputs/comparative_analysis CSVs")
    parser.add_argument("--input-dir", default="outputs/comparative_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    summary_df = pd.read_csv(input_dir / "summary_metrics.csv")
    geometry_df = pd.read_csv(input_dir / "geometry_metrics.csv")
    flip_df = pd.read_csv(input_dir / "flip_metrics.csv")
    disagreement_df = pd.read_csv(input_dir / "disagreement_metrics.csv")
    latency_df = pd.read_csv(input_dir / "latency_metrics.csv")
    curves_df = pd.read_csv(input_dir / "latency_curves.csv")
    difficulty_df = pd.read_csv(input_dir / "difficulty_metrics.csv")

    plot_accuracy_summary(summary_df, input_dir / "accuracy_summary.png")
    plot_gain_summary(summary_df, input_dir / "gain_summary.png")
    plot_geometry(geometry_df, input_dir / "geometry_analysis.png")
    plot_flip_metrics(flip_df, input_dir / "flip_analysis.png")
    plot_disagreement(disagreement_df, input_dir / "disagreement_analysis.png")
    plot_latency(latency_df, curves_df, input_dir / "latency_analysis.png")
    plot_difficulty(difficulty_df, input_dir / "difficulty_analysis.png")

    print(f"Saved plots to: {input_dir}")


if __name__ == "__main__":
    main()
