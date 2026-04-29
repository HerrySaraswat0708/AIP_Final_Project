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


def plot_improvement_comparison(acc_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Section 12 — Bar chart: accuracy of all 6 methods per dataset.
    Highlights improvement vs degradation vs baseline.
    """
    datasets = acc_df["dataset"].tolist()
    n_ds = len(datasets)
    n_m = len(ALL_METHODS)
    width = 0.8 / n_m
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(10, n_ds * 2.5), 6))
    for i, m in enumerate(ALL_METHODS):
        col = f"{m}_acc"
        if col not in acc_df.columns:
            continue
        vals = acc_df[col].to_numpy()
        bars = ax.bar(
            x + (i - n_m / 2 + 0.5) * width,
            vals,
            width=width * 0.9,
            color=COLORS.get(m, "gray"),
            label=m.upper(),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Section 12 — Improvement Methods: Accuracy Comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, min(1.05, acc_df[[f"{m}_acc" for m in ALL_METHODS if f"{m}_acc" in acc_df.columns]].max().max() + 0.05))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "improvement_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Delta plot: each adapter vs its base
    pairs = [
        ("conf_ftta", "freetta", "ConfGatedFreeTTA vs FreeTTA"),
        ("ent_tda",   "tda",     "EntropyGatedTDA vs TDA"),
        ("hybrid",    "clip",    "Hybrid vs CLIP"),
    ]
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig2.suptitle("Section 12 — Improvement Delta (method − baseline)", fontsize=13)
    for ax2, (m_new, m_base, title) in zip(axes, pairs):
        col_new  = f"{m_new}_acc"
        col_base = f"{m_base}_acc"
        if col_new not in acc_df.columns or col_base not in acc_df.columns:
            ax2.set_title(title + "\n(data unavailable)")
            continue
        deltas = (acc_df[col_new] - acc_df[col_base]).to_numpy()
        bar_colors = ["#4daf4a" if d > 0 else "#e41a1c" for d in deltas]
        ax2.bar(acc_df["dataset"], deltas, color=bar_colors)
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_title(title)
        ax2.set_ylabel("Accuracy Delta")
        ax2.grid(axis="y", alpha=0.25)
        ax2.tick_params(axis="x", rotation=30)

    fig2.tight_layout()
    fig2.savefig(output_dir / "improvement_deltas.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def plot_skip_rate_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """ConfGatedFreeTTA: skip rate vs accuracy over the stream."""
    if "conf_ftta_skip_rate" not in df.columns:
        return

    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step")
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()

        win = min(50, max(5, len(g) // 20))
        x = g["stream_step"].to_numpy() + 1
        roll_skip = g["conf_ftta_skip_rate"].rolling(win, min_periods=1).mean()
        roll_acc = g["conf_ftta_correct"].astype(float).rolling(win, min_periods=1).mean()
        roll_ftta = g["freetta_correct"].astype(float).rolling(win, min_periods=1).mean()

        ax1.plot(x, roll_skip, color="#c44e52", lw=2, label="Skip rate (rolling)")
        ax2.plot(x, roll_acc, color="#c44e52", lw=2, ls="--", label="ConfGated acc (rolling)")
        ax2.plot(x, roll_ftta, color="#55a868", lw=2, ls=":", label="FreeTTA acc (rolling)")

        ax1.set_ylabel("M-step skip rate", color="#c44e52")
        ax2.set_ylabel("Rolling accuracy")
        ax1.set_xlabel("Stream Step")
        ax1.set_title(f"ConfGatedFreeTTA: Skip Rate vs Accuracy — {ds.upper()}")
        lines1, lbls1 = ax1.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8)
        ax1.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(out / "conf_ftta_skip_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
