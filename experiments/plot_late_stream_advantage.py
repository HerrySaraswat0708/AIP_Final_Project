"""
Per-dataset late-stream advantage plots.

For each dataset: rolling (adapter accuracy - CLIP accuracy) over the full stream,
with the late-stream region (final 30%) shaded and annotated with mean advantage.

Outputs: outputs/dynamics_analysis/late_stream_advantage_{dataset}.png  (one per dataset)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["caltech", "dtd", "eurosat", "pets", "imagenet"]
DATASET_LABELS = {
    "caltech": "Caltech101",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "pets": "Oxford Pets",
    "imagenet": "ImageNet-V2",
}
COLORS = {
    "tda":     "#DD8452",
    "freetta": "#55A868",
    "zero":    "#888888",
    "shade":   "#E8E8F8",
}
WINDOW = 50
LATE_FRAC = 0.30   # define "late stream" as the last 30%


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr.astype(float))
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = (cs[i] - (cs[start - 1] if start > 0 else 0)) / (i - start + 1)
    return out


def first_crossover(adv: np.ndarray) -> int | None:
    """First index where rolling advantage >= 0 (break-even vs CLIP)."""
    for i, v in enumerate(adv):
        if not np.isnan(v) and v >= 0:
            return i
    return None


def mean_late(arr: np.ndarray, late_frac: float) -> float:
    n = len(arr)
    late_start = int((1 - late_frac) * n)
    return float(np.nanmean(arr[late_start:]))


# ── single-dataset plot ───────────────────────────────────────────────────────

def plot_one(ds: str, df: pd.DataFrame, out_dir: Path):
    n = len(df)
    x_pct = np.arange(n) / n * 100            # x-axis: stream %

    r_clip  = rolling_mean(df["clip_correct"].values,    WINDOW)
    r_tda   = rolling_mean(df["tda_correct"].values,     WINDOW)
    r_ftta  = rolling_mean(df["freetta_correct"].values, WINDOW)

    adv_tda  = (r_tda  - r_clip) * 100   # percentage-point advantage
    adv_ftta = (r_ftta - r_clip) * 100

    late_start_pct = (1 - LATE_FRAC) * 100   # x value where late region starts

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # shaded late-stream region
    ax.axvspan(late_start_pct, 100, color=COLORS["shade"], alpha=0.6,
               label=f"Late stream (last {int(LATE_FRAC*100)}%)", zorder=0)

    # zero baseline
    ax.axhline(0, color=COLORS["zero"], lw=1.2, ls="--", zorder=1)

    # advantage curves
    ax.plot(x_pct, adv_tda,  color=COLORS["tda"],     lw=1.8, label="TDA − CLIP",     zorder=3)
    ax.plot(x_pct, adv_ftta, color=COLORS["freetta"],  lw=1.8, label="FreeTTA − CLIP", zorder=3)

    # break-even markers
    be_tda  = first_crossover(adv_tda)
    be_ftta = first_crossover(adv_ftta)
    for be, color, name in [(be_tda, COLORS["tda"], "TDA"),
                             (be_ftta, COLORS["freetta"], "FreeTTA")]:
        if be is not None and be < n:
            bx = x_pct[be]
            by = adv_tda[be] if name == "TDA" else adv_ftta[be]
            ax.axvline(bx, color=color, lw=0.8, ls=":", alpha=0.7, zorder=2)
            ax.annotate(f"{name} BE\n{bx:.1f}%", xy=(bx, by),
                        xytext=(bx + 1.5, by + 0.3),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.6))

    # late-stream mean advantage annotations (horizontal bars)
    ml_tda  = mean_late(adv_tda,  LATE_FRAC)
    ml_ftta = mean_late(adv_ftta, LATE_FRAC)
    for ml, color, offset in [(ml_tda, COLORS["tda"], 0.15),
                               (ml_ftta, COLORS["freetta"], -0.15)]:
        ax.hlines(ml, late_start_pct, 100, colors=color, lw=2.5,
                  linestyles="solid", alpha=0.55, zorder=4)
        sign = "+" if ml >= 0 else ""
        ax.text(100.3, ml + offset, f"{sign}{ml:.2f}pp",
                color=color, fontsize=8, va="center", fontweight="bold")

    # axis labels / title
    ax.set_xlabel("Stream progress (%)", fontsize=11)
    ax.set_ylabel("Rolling accuracy advantage\nover CLIP (pp, window=50)", fontsize=10)
    ax.set_title(f"Late-Stream Advantage — {DATASET_LABELS[ds]}  (N={n:,})",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 107)   # leave room for right-side labels
    ax.grid(alpha=0.25, zorder=0)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    # append mean-advantage lines as legend entries
    handles.append(mpatches.Patch(color=COLORS["tda"],    alpha=0.55,
                                  label=f"TDA late mean: {'+' if ml_tda>=0 else ''}{ml_tda:.2f}pp"))
    handles.append(mpatches.Patch(color=COLORS["freetta"], alpha=0.55,
                                  label=f"FreeTTA late mean: {'+' if ml_ftta>=0 else ''}{ml_ftta:.2f}pp"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out_path = out_dir / f"late_stream_advantage_{ds}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}  "
          f"(TDA late={'+' if ml_tda>=0 else ''}{ml_tda:.3f}pp, "
          f"FreeTTA late={'+' if ml_ftta>=0 else ''}{ml_ftta:.3f}pp)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",  default="outputs/comparative_analysis",
                        help="Directory containing per-dataset comparative analysis outputs")
    parser.add_argument("--output-dir", default="outputs/dynamics_analysis",
                        help="Directory to write PNG plots")
    parser.add_argument("--datasets",   nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--window",     type=int, default=WINDOW,
                        help="Rolling window size (samples)")
    args = parser.parse_args()

    global WINDOW
    WINDOW = args.window

    input_dir = PROJECT_ROOT / args.input_dir
    out_dir   = PROJECT_ROOT / args.output_dir

    generated = []
    for ds in args.datasets:
        psm = input_dir / ds / "per_sample_metrics.csv"
        if not psm.exists():
            print(f"  Skipping {ds}: {psm} not found")
            continue
        print(f"Processing {DATASET_LABELS[ds]} ...", end=" ", flush=True)
        df = pd.read_csv(psm)
        plot_one(ds, df, out_dir)
        generated.append(ds)

    print(f"\nDone. {len(generated)} plots written to {out_dir}/")


if __name__ == "__main__":
    main()
