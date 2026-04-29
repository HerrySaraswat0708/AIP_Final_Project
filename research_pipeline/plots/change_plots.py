from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_pipeline.analysis.metrics import ADAPT_METHODS

COLORS = {
    "clip":      "#4c72b0",
    "tda":       "#dd8452",
    "freetta":   "#55a868",
    "conf_ftta": "#c44e52",
    "ent_tda":   "#8172b2",
    "hybrid":    "#937860",
}


def plot_prediction_change_breakdown(
    df: pd.DataFrame, flip_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    Section 3 — Stacked bar chart: unchanged_correct | unchanged_wrong |
                                    beneficial | harmful | other_changed_wrong
    One figure per dataset.
    """
    for ds, g_flip in flip_df.groupby("dataset", observed=True):
        methods = g_flip["method"].tolist()
        n_samples = g_flip["n_samples"].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Prediction Change Analysis — {ds.upper()}", fontsize=13)

        # Left: stacked bar
        layers = [
            ("n_unchanged_correct",    "Unchanged Correct",     "#66c2a5"),
            ("n_unchanged_wrong",      "Unchanged Wrong",       "#bdbdbd"),
            ("n_beneficial",           "Beneficial Flip",       "#4daf4a"),
            ("n_harmful",              "Harmful Flip",          "#e41a1c"),
            ("n_other_changed_wrong",  "Changed Wrong→Wrong",   "#ffb347"),
        ]
        bottoms = np.zeros(len(methods))
        for col, label, color in layers:
            vals = g_flip[col].to_numpy(dtype=float) / max(float(n_samples), 1)
            axes[0].bar(methods, vals, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
            bottoms += vals

        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Fraction of samples")
        axes[0].set_title("Prediction Breakdown")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8, loc="upper right")

        # Right: change rate + net correction rate
        x = np.arange(len(methods))
        w = 0.35
        axes[1].bar(
            x - w / 2, g_flip["change_rate"], width=w,
            label="Change Rate", color="#6baed6"
        )
        axes[1].bar(
            x + w / 2, g_flip["net_correction_rate"], width=w,
            label="Net Correction Rate", color="#31a354"
        )
        axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=15, ha="right")
        axes[1].set_ylabel("Rate")
        axes[1].set_title("Change Rate vs Net Correction")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()

        fig.tight_layout()
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "prediction_change.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_correction_efficiency_comparison(
    novel_ce: pd.DataFrame, output_dir: Path
) -> None:
    """Section 10 CE — grouped bar across datasets."""
    datasets = novel_ce["dataset"].unique() if "dataset" in novel_ce.columns else []
    methods = novel_ce["method"].unique()

    # If CE comes from flip_df (no dataset column yet), add groupby
    if "dataset" not in novel_ce.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n_ds = len(datasets)
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(n_ds)

    for i, m in enumerate(sorted(methods)):
        vals = []
        for ds in sorted(datasets):
            sub = novel_ce[(novel_ce["dataset"] == ds) & (novel_ce["method"] == m)]
            vals.append(float(sub["correction_efficiency"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + (i - n_m / 2 + 0.5) * width, vals, width=width * 0.9,
               label=m, color=COLORS.get(m, "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels(sorted(datasets))
    ax.set_ylabel("Correction Efficiency (CE)")
    ax.set_title("Section 10 — Correction Efficiency per Dataset / Method")
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5, label="CE=0.5")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "correction_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
