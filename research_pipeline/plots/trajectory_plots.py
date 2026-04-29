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
CORE = ("clip", "tda", "freetta")
ALL_METHODS = ("clip", "tda", "freetta", "conf_ftta", "ent_tda", "hybrid")


def plot_trajectory(traj: pd.DataFrame, output_dir: Path) -> None:
    """Section 5 — rolling accuracy, confidence, entropy curves."""
    for ds, g in traj.groupby("dataset", observed=True):
        g = g.sort_values("stream_step")
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)

        x = g["stream_step"].to_numpy() + 1

        # Core 3-panel (CLIP / TDA / FreeTTA)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Trajectory Analysis — {ds.upper()}", fontsize=13)

        for m in CORE:
            lw = 2.5 if m == "clip" else 2.0
            ls = "--" if m == "clip" else "-"
            axes[0].plot(x, g[f"rolling_{m}_acc"], label=m.upper(),
                         color=COLORS[m], lw=lw, ls=ls)
            axes[1].plot(x, g[f"rolling_{m}_conf"], label=m.upper(),
                         color=COLORS[m], lw=lw, ls=ls)
            axes[2].plot(x, g[f"rolling_{m}_ent"], label=m.upper(),
                         color=COLORS[m], lw=lw, ls=ls)

        axes[0].set_ylabel("Rolling Accuracy")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.25)
        axes[1].set_ylabel("Rolling Confidence")
        axes[1].grid(alpha=0.25)
        axes[2].set_ylabel("Rolling Entropy")
        axes[2].set_xlabel("Stream Step")
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(out / "trajectory_core.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # All 6 methods — accuracy only
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.suptitle(f"Rolling Accuracy (all methods) — {ds.upper()}", fontsize=13)
        for m in ALL_METHODS:
            col = f"rolling_{m}_acc"
            if col not in g.columns:
                continue
            lw = 2.5 if m == "clip" else 1.8
            ls = "--" if m == "clip" else "-"
            ax2.plot(x, g[col], label=m.upper(), color=COLORS.get(m, "gray"), lw=lw, ls=ls)
        ax2.set_ylabel("Rolling Accuracy")
        ax2.set_xlabel("Stream Step")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        fig2.savefig(out / "trajectory_all_methods.png", dpi=180, bbox_inches="tight")
        plt.close(fig2)


def plot_stability_scores(stability_df: pd.DataFrame, output_dir: Path) -> None:
    """Section 10 SS — heatmap of stability scores per dataset x method."""
    methods = [m for m in ALL_METHODS if f"{m}_stability" in stability_df.columns]
    datasets = stability_df["dataset"].tolist()
    matrix = stability_df[[f"{m}_stability" for m in methods]].to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), max(4, len(datasets) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Stability Score (1 / (1 + std(rolling_acc)))")
    plt.colorbar(im, ax=ax)

    for i in range(len(datasets)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "stability_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
