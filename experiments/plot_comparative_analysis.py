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
    datasets = sorted(flip_df["dataset"].unique().tolist())
    methods = ["tda", "freetta"]
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    stack_columns = [
        ("unchanged_correct_rate", "Unchanged Correct", "#66c2a5"),
        ("unchanged_wrong_rate", "Unchanged Wrong", "#bdbdbd"),
        (
            "beneficial_flip_rate",
            "Beneficial Flip",
            "#4daf4a",
        ),
        ("harmful_flip_rate", "Harmful Flip", "#e41a1c"),
        ("other_changed_wrong_rate", "Changed Wrong->Wrong", "#ffb347"),
    ]

    for method_idx, method in enumerate(methods):
        group = flip_df[flip_df["method"] == method].set_index("dataset").reindex(datasets)
        bottoms = np.zeros(len(datasets))
        positions = x + (-width / 2 if method == "tda" else width / 2)
        for column, label, color in stack_columns:
            if column == "beneficial_flip_rate":
                values = group["beneficial_flip_count"].to_numpy(dtype=float) / np.maximum(
                    group["samples"].to_numpy(dtype=float), 1.0
                )
            elif column == "harmful_flip_rate":
                values = group["harmful_flip_count"].to_numpy(dtype=float) / np.maximum(
                    group["samples"].to_numpy(dtype=float), 1.0
                )
            elif column == "other_changed_wrong_rate":
                values = group["other_changed_wrong_count"].to_numpy(dtype=float) / np.maximum(
                    group["samples"].to_numpy(dtype=float), 1.0
                )
            else:
                values = group[column].to_numpy(dtype=float)
            axes[0].bar(
                positions,
                values,
                width=width,
                bottom=bottoms,
                label=f"{method.upper()} {label}",
                color=color,
                alpha=0.85 if method == "tda" else 0.55,
            )
            bottoms += values

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction of Samples")
    axes[0].set_title("Prediction Change Breakdown")
    axes[0].grid(axis="y", alpha=0.25)

    tda_group = flip_df[flip_df["method"] == "tda"].set_index("dataset").reindex(datasets)
    freetta_group = flip_df[flip_df["method"] == "freetta"].set_index("dataset").reindex(datasets)
    axes[1].bar(x - width / 2, tda_group["change_rate"], width=width, label="TDA Change Rate", color="#4c72b0")
    axes[1].bar(x + width / 2, freetta_group["change_rate"], width=width, label="FreeTTA Change Rate", color="#55a868")
    axes[1].plot(x, tda_group["net_correction_rate"], marker="o", color="#1f3d7a", label="TDA Net Correction")
    axes[1].plot(x, freetta_group["net_correction_rate"], marker="o", color="#2d6a3d", label="FreeTTA Net Correction")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Change Rate and Net Correction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

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

    for ax, dataset in zip(axes[:, 0], datasets, strict=False):
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


def plot_entropy_confidence_summary(entropy_df: pd.DataFrame, output_path: Path) -> None:
    wrong_df = entropy_df[entropy_df["subset"] == "wrong"].copy()
    datasets = sorted(wrong_df["dataset"].unique().tolist())
    methods = ["clip", "tda", "freetta"]
    x = np.arange(len(datasets))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, method in enumerate(methods):
        group = wrong_df[wrong_df["method"] == method].set_index("dataset").reindex(datasets)
        axes[0].bar(x + (idx - 1) * width, group["mean_entropy"], width=width, label=method.upper())
        axes[1].bar(x + (idx - 1) * width, group["mean_confidence"], width=width, label=method.upper())

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Mean Entropy")
    axes[0].set_title("Entropy on Wrong Predictions")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_title("Confidence on Wrong Predictions")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_failure_bucket_summary(failure_df: pd.DataFrame, output_path: Path) -> None:
    datasets = sorted(failure_df["dataset"].unique().tolist())
    buckets = failure_df["bucket"].unique().tolist()
    x = np.arange(len(datasets))
    width = 0.8 / max(len(buckets), 1)

    fig, ax = plt.subplots(figsize=(15, 6))
    for idx, bucket in enumerate(buckets):
        group = failure_df[failure_df["bucket"] == bucket].set_index("dataset").reindex(datasets)
        ax.bar(x + (idx - (len(buckets) - 1) / 2) * width, group["rate"], width=width, label=bucket)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Rate")
    ax.set_title("Failure Bucket Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
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

    entropy_path = input_dir / "entropy_confidence_metrics.csv"
    if entropy_path.exists():
        entropy_df = pd.read_csv(entropy_path)
        plot_entropy_confidence_summary(entropy_df, input_dir / "entropy_confidence_summary.png")

    failure_path = input_dir / "failure_bucket_summary.csv"
    if failure_path.exists():
        failure_df = pd.read_csv(failure_path)
        plot_failure_bucket_summary(failure_df, input_dir / "failure_bucket_summary.png")

    print(f"Saved plots to: {input_dir}")


if __name__ == "__main__":
    main()
