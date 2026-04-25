"""
Generate clean final results visualization comparing CLIP / TDA / FreeTTA.
Reads from outputs/comparative_analysis/summary_metrics.csv produced by
run_comparative_analysis.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PAPER_TARGETS = {
    "caltech":  dict(tda=94.24, freetta=94.63),
    "dtd":      dict(tda=47.40, freetta=46.96),
    "eurosat":  dict(tda=58.00, freetta=62.93),
    "pets":     dict(tda=88.63, freetta=90.11),
    "imagenet": dict(tda=64.67, freetta=64.92),
}

COLORS = {
    "clip":    "#7f7f7f",
    "tda":     "#1f77b4",
    "freetta": "#d62728",
}

DATASET_LABELS = {
    "caltech":  "Caltech-101",
    "dtd":      "DTD",
    "eurosat":  "EuroSAT",
    "pets":     "Oxford Pets",
    "imagenet": "ImageNetV2",
}


def load_results(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "summary_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"summary_metrics.csv not found at {path}")
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].str.lower()
    ordered = ["caltech", "dtd", "eurosat", "pets", "imagenet"]
    df["sort_key"] = df["dataset"].map({d: i for i, d in enumerate(ordered)})
    return df.sort_values("sort_key").reset_index(drop=True)


def plot_accuracy_bars(df: pd.DataFrame, output_path: Path) -> None:
    datasets = df["dataset"].tolist()
    n = len(datasets)
    x = np.arange(n)
    width = 0.22
    labels = [DATASET_LABELS.get(d, d) for d in datasets]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: absolute accuracy bars ──────────────────────────────────────
    ax = axes[0]
    bars_clip = ax.bar(x - width, df["clip_acc"] * 100, width, label="CLIP (zero-shot)",
                       color=COLORS["clip"], alpha=0.85, edgecolor="white")
    bars_tda = ax.bar(x, df["tda_acc"] * 100, width, label="TDA",
                      color=COLORS["tda"], alpha=0.85, edgecolor="white")
    bars_ft = ax.bar(x + width, df["freetta_acc"] * 100, width, label="FreeTTA (ours)",
                     color=COLORS["freetta"], alpha=0.85, edgecolor="white")

    # Star markers for paper targets
    for i, ds in enumerate(datasets):
        pt = PAPER_TARGETS.get(ds, {})
        if pt:
            ax.plot(i, pt["tda"], marker="*", color=COLORS["tda"], markersize=10,
                    label="_Paper TDA target" if i > 0 else "Paper target (★)")
            ax.plot(i, pt["freetta"], marker="*", color=COLORS["freetta"], markersize=10,
                    label="_Paper FreeTTA target")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CLIP vs TDA vs FreeTTA — Absolute Accuracy\n(★ = paper-reported target)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ymin = max(0, df[["clip_acc", "tda_acc", "freetta_acc"]].min().min() * 100 - 5)
    ax.set_ylim(ymin, 100)

    # ── Right: gain vs CLIP, bars ──────────────────────────────────────────
    ax = axes[1]
    tda_gain = df["tda_gain_vs_clip"] * 100
    ft_gain = df["freetta_gain_vs_clip"] * 100

    bars_tda_g = ax.bar(x - width / 2, tda_gain, width, label="TDA gain vs CLIP",
                        color=COLORS["tda"], alpha=0.85, edgecolor="white")
    bars_ft_g = ax.bar(x + width / 2, ft_gain, width, label="FreeTTA gain vs CLIP",
                       color=COLORS["freetta"], alpha=0.85, edgecolor="white")

    ax.axhline(0.0, color="black", linewidth=0.8)

    # Delta arrow for FreeTTA-TDA comparison
    for i, ds in enumerate(datasets):
        delta = float(df[df["dataset"] == ds]["freetta_minus_tda"].values[0]) * 100
        color = "#2ca25f" if delta >= 0 else "#e34a33"
        x_pos = x[i] + width / 2
        ax.annotate(f"{delta:+.2f}%", xy=(x_pos, ft_gain.iloc[i]),
                    xytext=(x_pos, ft_gain.iloc[i] + 0.5),
                    ha="center", va="bottom", fontsize=8, color=color,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy Gain vs CLIP (%)")
    ax.set_title("Gain over CLIP Zero-Shot\n(labels = FreeTTA − TDA)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def plot_delta_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart showing FreeTTA − TDA delta per dataset with colour coding."""
    datasets = df["dataset"].tolist()
    n = len(datasets)
    deltas = df["freetta_minus_tda"] * 100
    colors = ["#2ca25f" if d >= 0 else "#e34a33" for d in deltas]
    labels = [DATASET_LABELS.get(d, d) for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels[::-1], deltas.tolist()[::-1], color=colors[::-1],
                   edgecolor="white", height=0.6)
    ax.axvline(0.0, color="black", linewidth=0.8)

    for bar, val in zip(bars, deltas.tolist()[::-1]):
        sign = "+" if val >= 0 else ""
        ax.text(val + (0.1 if val >= 0 else -0.1), bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=11, fontweight="bold")

    win = (deltas >= 0).sum()
    avg = deltas.mean()
    ax.set_xlabel("Accuracy Difference: FreeTTA − TDA (%)")
    ax.set_title(f"FreeTTA vs TDA Per-Dataset Advantage\n"
                 f"FreeTTA wins {win}/{n} datasets  |  Average: {avg:+.2f}%")
    ax.grid(axis="x", alpha=0.3)

    green_patch = mpatches.Patch(color="#2ca25f", label="FreeTTA wins")
    red_patch = mpatches.Patch(color="#e34a33", label="TDA wins")
    ax.legend(handles=[green_patch, red_patch], fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("  FINAL RESULTS: FreeTTA vs TDA on 5 Datasets (CLIP ViT-B/16)")
    print("=" * 80)
    print(f"{'Dataset':<14} {'CLIP':>8} {'TDA':>9} {'FreeTTA':>9} {'Δ(F-T)':>10} {'Winner':>10}")
    print("-" * 80)

    freetta_wins, tda_wins = 0, 0
    for _, row in df.iterrows():
        ds = DATASET_LABELS.get(row["dataset"], row["dataset"])
        clip = row["clip_acc"] * 100
        tda = row["tda_acc"] * 100
        ft = row["freetta_acc"] * 100
        delta = row["freetta_minus_tda"] * 100
        if delta > 0.01:
            winner = "FreeTTA ✓"
            freetta_wins += 1
        elif delta < -0.01:
            winner = "TDA ✓"
            tda_wins += 1
        else:
            winner = "Tie ≈"
        sign = "+" if delta >= 0 else ""
        print(f"{ds:<14} {clip:>7.2f}%  {tda:>8.2f}%  {ft:>8.2f}%  {sign+f'{delta:.2f}':>8}%  {winner:>10}")

    print("-" * 80)
    avg_tda = df["tda_acc"].mean() * 100
    avg_ft = df["freetta_acc"].mean() * 100
    avg_delta = avg_ft - avg_tda
    print(f"{'Average':<14} {'':>8}  {avg_tda:>8.2f}%  {avg_ft:>8.2f}%  {'+'+f'{avg_delta:.2f}' if avg_delta >= 0 else f'{avg_delta:.2f}':>8}%")
    print(f"\nFreeTTA wins: {freetta_wins}/5 datasets  |  Average advantage: {avg_delta:+.2f}%")
    print("=" * 80)


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "comparative_analysis"
    df = load_results(out_dir)
    plot_accuracy_bars(df, out_dir / "accuracy_summary.png")
    plot_delta_heatmap(df, out_dir / "freetta_vs_tda_delta.png")
    print_summary_table(df)


if __name__ == "__main__":
    main()
