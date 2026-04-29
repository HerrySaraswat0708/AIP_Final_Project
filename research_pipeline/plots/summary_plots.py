from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "clip":      "#4c72b0",
    "tda":       "#dd8452",
    "freetta":   "#55a868",
    "conf_ftta": "#c44e52",
    "ent_tda":   "#8172b2",
    "hybrid":    "#937860",
}
ALL_METHODS = ("clip", "tda", "freetta", "conf_ftta", "ent_tda", "hybrid")


def plot_accuracy_summary(acc_df: pd.DataFrame, output_dir: Path) -> None:
    """Cross-dataset accuracy heatmap."""
    methods = [m for m in ALL_METHODS if f"{m}_acc" in acc_df.columns]
    datasets = acc_df["dataset"].tolist()
    matrix = acc_df[[f"{m}_acc" for m in methods]].to_numpy()

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.4), max(4, len(datasets) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Top-1 Accuracy Heatmap — All Methods × Datasets")
    plt.colorbar(im, ax=ax)

    for i in range(len(datasets)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "accuracy_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_split(diff_df: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy by difficulty bin (easy/medium/hard)."""
    methods = [m for m in ALL_METHODS if f"{m}_acc" in diff_df.columns]
    datasets = diff_df["dataset"].unique()
    bins = ["easy", "medium", "hard"]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = diff_df[diff_df["dataset"] == ds]
        x = np.arange(len(bins))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = []
            for b in bins:
                row = sub[sub["difficulty"] == b]
                vals.append(float(row[f"{m}_acc"].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w * 0.9,
                   color=COLORS.get(m, "gray"), label=m, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_title(ds.upper())
        ax.set_ylabel("Accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.05)
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1))

    fig.suptitle("Accuracy by Difficulty Bin (CLIP entropy-based)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "difficulty_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement_summary(disagree_df: pd.DataFrame, output_dir: Path) -> None:
    """Section 7 — disagreement rate + who wins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    datasets = disagree_df["dataset"].tolist()
    x = np.arange(len(datasets))

    axes[0].bar(x, disagree_df["disagreement_rate"], color="#984ea3", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Disagreement Rate")
    axes[0].set_title("Section 7 — TDA vs FreeTTA Disagreement Rate")
    axes[0].grid(axis="y", alpha=0.25)

    w = 0.35
    axes[1].bar(x - w / 2, disagree_df["tda_acc_on_disagree"], w, label="TDA acc", color="#1f77b4")
    axes[1].bar(x + w / 2, disagree_df["freetta_acc_on_disagree"], w, label="FreeTTA acc", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylabel("Accuracy on Disagreements")
    axes[1].set_title("Accuracy when TDA ≠ FreeTTA")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "disagreement_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_failure_bucket_summary(fail_df: pd.DataFrame, output_dir: Path) -> None:
    """Section 8 — failure bucket rates across datasets."""
    buckets = fail_df["bucket"].unique()
    datasets = fail_df["dataset"].unique()
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(datasets))
    w = 0.8 / len(buckets)
    bucket_colors = {
        "only_freetta_correct":   "#ff7f0e",
        "only_tda_correct":       "#1f77b4",
        "clip_correct_both_fail": "#9467bd",
        "all_fail":               "#333333",
        "all_correct":            "#4daf4a",
    }
    for i, bucket in enumerate(sorted(buckets)):
        rates = []
        for ds in datasets:
            row = fail_df[(fail_df["dataset"] == ds) & (fail_df["bucket"] == bucket)]
            rates.append(float(row["rate"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - len(buckets) / 2 + 0.5) * w, rates, w * 0.9,
               color=bucket_colors.get(bucket, "gray"), label=bucket, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Rate")
    ax.set_title("Section 8 — Failure Bucket Rates")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_buckets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
