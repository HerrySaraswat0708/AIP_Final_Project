"""Four-category analysis:
  1. Adaptation Dynamics   – rolling accuracy, stability, adaptation speed
  2. Uncertainty Analysis  – entropy distributions, entropy-accuracy correlation, EM-weight
  3. Distribution Modeling – PCA of logit space, μ-drift, geometry alignment
  4. Computational Efficiency – per-sample time, cache growth, scaling
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["caltech", "dtd", "eurosat", "pets", "imagenet"]
DATASET_LABELS = {
    "caltech": "Caltech101",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "pets": "Pets",
    "imagenet": "ImageNet",
}
COLORS = {"clip": "#4472B4", "tda": "#DD8452", "freetta": "#55A868"}
WINDOW = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr.astype(float))
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = (cs[i] - (cs[start - 1] if start > 0 else 0)) / (i - start + 1)
    return out


def load_per_sample(ds: str, input_dir: Path) -> pd.DataFrame | None:
    p = input_dir / ds / "per_sample_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def savefig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Section 1: Adaptation Dynamics ───────────────────────────────────────────

def plot_rolling_accuracy(dfs: dict[str, pd.DataFrame], out_dir: Path) -> dict:
    """Rolling accuracy curves + stability metrics for all datasets."""
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (ds, df) in enumerate(dfs.items()):
        ax = axes[idx]
        n = len(df)
        x = np.arange(n) / n * 100  # % of stream

        r_clip = rolling_mean(df["clip_correct"].values, WINDOW)
        r_tda = rolling_mean(df["tda_correct"].values, WINDOW)
        r_ftta = rolling_mean(df["freetta_correct"].values, WINDOW)

        ax.plot(x, r_clip * 100, color=COLORS["clip"], lw=1.5, label="CLIP", alpha=0.85)
        ax.plot(x, r_tda * 100, color=COLORS["tda"], lw=1.5, label="TDA", alpha=0.85)
        ax.plot(x, r_ftta * 100, color=COLORS["freetta"], lw=1.5, label="FreeTTA", alpha=0.85)
        ax.set_title(DATASET_LABELS[ds], fontweight="bold")
        ax.set_xlabel("Stream progress (%)")
        ax.set_ylabel("Rolling acc. (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # stability: std-dev of rolling acc in last 30% of stream
        late_start = int(0.7 * n)
        stability_clip = np.std(r_clip[late_start:]) * 100
        stability_tda = np.std(r_tda[late_start:]) * 100
        stability_ftta = np.std(r_ftta[late_start:]) * 100

        # time-to-beat-clip: first step where rolling acc ≥ rolling clip
        tbc_tda = next((i for i in range(n) if r_tda[i] >= r_clip[i]), n)
        tbc_ftta = next((i for i in range(n) if r_ftta[i] >= r_clip[i]), n)

        rows.append({
            "dataset": ds,
            "final_clip_acc": df["clip_correct"].mean() * 100,
            "final_tda_acc": df["tda_correct"].mean() * 100,
            "final_ftta_acc": df["freetta_correct"].mean() * 100,
            "stability_clip": stability_clip,
            "stability_tda": stability_tda,
            "stability_ftta": stability_ftta,
            "time_to_beat_clip_tda_pct": tbc_tda / n * 100,
            "time_to_beat_clip_ftta_pct": tbc_ftta / n * 100,
        })

    # hide unused subplot
    for ax in axes[len(dfs):]:
        ax.set_visible(False)

    fig.suptitle("Rolling Accuracy (window=50) Over Stream", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "adaptation_rolling_accuracy.png")
    return pd.DataFrame(rows).set_index("dataset")


def plot_mu_drift(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """FreeTTA μ-drift over stream progress."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 4), sharey=False)
    if len(dfs) == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, dfs.items()):
        x = np.arange(len(df)) / len(df) * 100
        ax.plot(x, df["freetta_mu_drift"].values, color=COLORS["freetta"], lw=1.2)
        ax.set_title(f"{DATASET_LABELS[ds]}: μ Drift", fontweight="bold")
        ax.set_xlabel("Stream progress (%)")
        ax.set_ylabel("|μ − μ₀|")
        ax.grid(alpha=0.3)

    fig.suptitle("FreeTTA Class-Mean Drift Over Time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "adaptation_mu_drift.png")


def plot_cache_growth(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """TDA positive cache size over stream."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 4), sharey=False)
    if len(dfs) == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, dfs.items()):
        x = np.arange(len(df)) / len(df) * 100
        ax.plot(x, df["tda_positive_cache_size"].values, color=COLORS["tda"], lw=1.2, label="Pos cache")
        ax.plot(x, df["tda_negative_cache_size"].values, color="gray", lw=1.0, linestyle="--", label="Neg cache")
        ax.set_title(f"{DATASET_LABELS[ds]}: Cache Growth", fontweight="bold")
        ax.set_xlabel("Stream progress (%)")
        ax.set_ylabel("Total slots used")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("TDA Cache Size Over Stream", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "adaptation_cache_growth.png")


# ── Section 2: Uncertainty Analysis ──────────────────────────────────────────

def compute_entropy_stats(dfs: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Entropy distributions and entropy–accuracy correlation."""
    rows = []
    fig, axes = plt.subplots(len(dfs), 3, figsize=(15, 3.5 * len(dfs)))
    if len(dfs) == 1:
        axes = axes.reshape(1, 3)

    bins = np.linspace(0, 5, 40)

    for row_idx, (ds, df) in enumerate(dfs.items()):
        for col_idx, (method, ent_col, lbl) in enumerate([
            ("clip",    "clip_entropy",    "CLIP"),
            ("tda",     "tda_entropy",     "TDA"),
            ("freetta", "freetta_entropy", "FreeTTA"),
        ]):
            ax = axes[row_idx, col_idx]
            corr_col = f"{method}_correct"
            ent_corr  = df.loc[df[corr_col] == 1, ent_col].values
            ent_wrong = df.loc[df[corr_col] == 0, ent_col].values

            ax.hist(ent_corr,  bins=bins, alpha=0.6, color="green",  label="Correct", density=True)
            ax.hist(ent_wrong, bins=bins, alpha=0.6, color="red",    label="Wrong",   density=True)
            ax.set_title(f"{DATASET_LABELS[ds]} — {lbl}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Entropy")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # entropy–accuracy correlation (rank)
            rho, pval = stats.spearmanr(df[ent_col], 1 - df[corr_col])

            rows.append({
                "dataset": ds,
                "method": method,
                "mean_entropy_correct": float(np.mean(ent_corr)),
                "mean_entropy_wrong":   float(np.mean(ent_wrong)),
                "entropy_gap":          float(np.mean(ent_wrong) - np.mean(ent_corr)),
                "spearman_rho":         float(rho),
                "spearman_pval":        float(pval),
            })

    fig.suptitle("Entropy Distributions: Correct vs Wrong Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "uncertainty_entropy_distributions.png")
    return pd.DataFrame(rows)


def plot_em_weight_distribution(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """FreeTTA EM weight (α) distribution — how much each sample contributes."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(4 * len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, dfs.items()):
        weights = df["freetta_em_weight"].values
        ax.hist(weights, bins=40, color=COLORS["freetta"], alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(weights), color="red", lw=1.5, linestyle="--", label=f"Mean={np.mean(weights):.3f}")
        ax.set_title(f"{DATASET_LABELS[ds]}", fontweight="bold")
        ax.set_xlabel("EM weight α = 1 − H(x)/log(C)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("FreeTTA EM Weight Distribution (Confidence Gating)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "uncertainty_em_weight.png")


def plot_entropy_vs_accuracy(dfs: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Bin entropy into deciles, plot accuracy per decile (calibration plot)."""
    fig, axes = plt.subplots(len(dfs), 3, figsize=(15, 3.5 * len(dfs)))
    if len(dfs) == 1:
        axes = axes.reshape(1, 3)

    rows = []
    for row_idx, (ds, df) in enumerate(dfs.items()):
        for col_idx, (method, ent_col) in enumerate([
            ("clip",    "clip_entropy"),
            ("tda",     "tda_entropy"),
            ("freetta", "freetta_entropy"),
        ]):
            ax = axes[row_idx, col_idx]
            corr_col = f"{method}_correct"

            # bin into deciles
            df["_bin"] = pd.qcut(df[ent_col], q=10, labels=False, duplicates="drop")
            gb = df.groupby("_bin")
            bin_ent = gb[ent_col].mean().values
            bin_acc = gb[corr_col].mean().values * 100

            ax.plot(bin_ent, bin_acc, "o-", color=COLORS[method], lw=1.5, markersize=5)
            ax.set_title(f"{DATASET_LABELS[ds]} — {method.upper()}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Mean entropy (decile)")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(alpha=0.3)

            # log for table
            rho, _ = stats.spearmanr(df[ent_col], 1 - df[corr_col])
            rows.append({"dataset": ds, "method": method,
                         "low_ent_acc": float(bin_acc[0]) if len(bin_acc) > 0 else np.nan,
                         "high_ent_acc": float(bin_acc[-1]) if len(bin_acc) > 0 else np.nan,
                         "entropy_acc_rho": float(rho)})

            df.drop(columns=["_bin"], inplace=True)

    fig.suptitle("Accuracy vs Entropy (decile bins)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "uncertainty_entropy_accuracy_curve.png")
    return pd.DataFrame(rows)


# ── Section 3: Distribution Modeling Analysis ─────────────────────────────────

def plot_pca_logit_space(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """PCA of logit space colored by correctness category."""
    for ds, df in dfs.items():
        pca_path = (PROJECT_ROOT / "outputs" / "comparative_analysis" / ds / "pca_projection.csv")
        if not pca_path.exists():
            continue
        pca_df = pd.read_csv(pca_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (method, pc1, pc2, title) in zip(axes, [
            ("clip",    "clip_pc1",    "clip_pc2",    "CLIP"),
            ("tda",     "tda_pc1",     "tda_pc2",     "TDA"),
            ("freetta", "freetta_pc1", "freetta_pc2", "FreeTTA"),
        ]):
            corr_col = f"{method}_correct" if method != "clip" else "clip_correct"
            # join with df to get correct/wrong mask
            merged = pd.merge(pca_df, df[["sample_index", "clip_correct", "tda_correct", "freetta_correct"]],
                              on="sample_index", how="left")
            correct_mask = merged[corr_col].values == 1

            ax.scatter(pca_df.loc[~correct_mask, pc1], pca_df.loc[~correct_mask, pc2],
                       c="red", alpha=0.3, s=5, label="Wrong")
            ax.scatter(pca_df.loc[correct_mask, pc1], pca_df.loc[correct_mask, pc2],
                       c="green", alpha=0.3, s=5, label="Correct")
            ax.set_title(f"{title} logit PCA", fontweight="bold")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(fontsize=8, markerscale=3)

        fig.suptitle(f"{DATASET_LABELS[ds]} — PCA of Logit Space", fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig(fig, out_dir / f"distribution_pca_{ds}.png")


def compute_geometry_table(input_dir: Path) -> pd.DataFrame:
    """Oracle centroid vs 1-NN accuracy and GAS for all datasets."""
    rows = []
    for ds in DATASETS:
        gpath = input_dir / ds / "geometry_metrics.csv"
        ipath = input_dir / ds / "internal_metrics.csv"
        if not gpath.exists() and not ipath.exists():
            continue
        p = ipath if ipath.exists() else gpath
        d = pd.read_csv(p).iloc[0]
        rows.append({
            "dataset": ds,
            "oracle_centroid_acc": float(d.get("oracle_centroid_acc", np.nan)) * 100,
            "oracle_1nn_acc":      float(d.get("oracle_1nn_acc", np.nan)) * 100,
            "GAS":                 float(d.get("geometry_alignment_score", np.nan)),
            "mu_drift_final":      float(d.get("freetta_final_mu_drift", d.get("freetta_mean_mu_update_norm", np.nan))),
            "cache_pressure":      float(d.get("tda_cache_pressure_ratio", np.nan)),
        })
    return pd.DataFrame(rows).set_index("dataset")


def plot_oracle_comparison(geo_df: pd.DataFrame, out_dir: Path):
    """Bar chart: oracle centroid vs 1-NN acc by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ds_labels = [DATASET_LABELS.get(d, d) for d in geo_df.index]
    x = np.arange(len(ds_labels))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, geo_df["oracle_centroid_acc"], w, label="Oracle Centroid", color="#4472B4")
    ax.bar(x + w/2, geo_df["oracle_1nn_acc"],      w, label="Oracle 1-NN",    color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, rotation=20)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Oracle Centroid vs 1-NN Accuracy\n(upper bound for FreeTTA vs TDA)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["green" if g > 0 else "red" for g in geo_df["GAS"]]
    ax.bar(ds_labels, geo_df["GAS"], color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("GAS = OracleCentroid − Oracle1NN")
    ax.set_title("Geometry Alignment Score (GAS)\n>0 favours FreeTTA, <0 favours TDA", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    savefig(fig, out_dir / "distribution_oracle_geometry.png")


def plot_freetta_distribution_evolution(dfs: dict[str, pd.DataFrame], out_dir: Path):
    """FreeTTA sigma_trace and prior_entropy evolution over stream."""
    fig, axes = plt.subplots(2, len(dfs), figsize=(4 * len(dfs), 7))
    if len(dfs) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, (ds, df) in enumerate(dfs.items()):
        x = np.arange(len(df)) / len(df) * 100

        axes[0, col_idx].plot(x, df["freetta_sigma_trace"], color=COLORS["freetta"], lw=1.0)
        axes[0, col_idx].set_title(f"{DATASET_LABELS[ds]}: σ-trace", fontweight="bold")
        axes[0, col_idx].set_xlabel("Stream (%)")
        axes[0, col_idx].set_ylabel("Σ trace (spread)")
        axes[0, col_idx].grid(alpha=0.3)

        axes[1, col_idx].plot(x, df["freetta_prior_entropy"], color="#9467bd", lw=1.0)
        axes[1, col_idx].set_title(f"{DATASET_LABELS[ds]}: prior entropy", fontweight="bold")
        axes[1, col_idx].set_xlabel("Stream (%)")
        axes[1, col_idx].set_ylabel("H(π) — class prior entropy")
        axes[1, col_idx].grid(alpha=0.3)

    fig.suptitle("FreeTTA Distribution Evolution: Covariance Trace & Class Prior Entropy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "distribution_freetta_evolution.png")


# ── Section 4: Computational Efficiency ──────────────────────────────────────

def compute_efficiency_table(comparison_json: Path) -> pd.DataFrame:
    """Per-sample timing and throughput from comparison_results.json."""
    if not comparison_json.exists():
        return pd.DataFrame()

    with open(comparison_json) as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        ds = r["dataset"]
        n = r["n_samples"]
        t_tda  = r["time_tda"]
        t_ftta = r["time_freetta"]
        rows.append({
            "dataset":              ds,
            "n_samples":            n,
            "tda_total_s":          t_tda,
            "ftta_total_s":         t_ftta,
            "tda_ms_per_sample":    t_tda / n * 1000,
            "ftta_ms_per_sample":   t_ftta / n * 1000,
            "tda_samples_per_sec":  n / t_tda,
            "ftta_samples_per_sec": n / t_ftta,
            "speedup_ftta_vs_tda":  t_tda / t_ftta,
        })
    return pd.DataFrame(rows).set_index("dataset")


def plot_efficiency(eff_df: pd.DataFrame, out_dir: Path):
    """Plot timing comparison and scaling."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ds_labels = [DATASET_LABELS.get(d, d) for d in eff_df.index]
    x = np.arange(len(ds_labels))
    w = 0.35

    # per-sample latency
    ax = axes[0]
    ax.bar(x - w/2, eff_df["tda_ms_per_sample"],  w, label="TDA",     color=COLORS["tda"])
    ax.bar(x + w/2, eff_df["ftta_ms_per_sample"], w, label="FreeTTA", color=COLORS["freetta"])
    ax.set_xticks(x); ax.set_xticklabels(ds_labels, rotation=20)
    ax.set_ylabel("ms / sample")
    ax.set_title("Per-Sample Latency", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # throughput
    ax = axes[1]
    ax.bar(x - w/2, eff_df["tda_samples_per_sec"],  w, label="TDA",     color=COLORS["tda"])
    ax.bar(x + w/2, eff_df["ftta_samples_per_sec"], w, label="FreeTTA", color=COLORS["freetta"])
    ax.set_xticks(x); ax.set_xticklabels(ds_labels, rotation=20)
    ax.set_ylabel("Samples / second")
    ax.set_title("Throughput", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # speedup FreeTTA vs TDA
    ax = axes[2]
    colors = ["green" if v >= 1 else "red" for v in eff_df["speedup_ftta_vs_tda"]]
    ax.bar(ds_labels, eff_df["speedup_ftta_vs_tda"], color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax.axhline(1.0, color="black", lw=0.8, linestyle="--", label="Equal speed")
    ax.set_ylabel("TDA time / FreeTTA time  (>1 = FreeTTA faster)")
    ax.set_title("FreeTTA Speedup vs TDA", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Computational Efficiency Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(fig, out_dir / "efficiency_timing.png")


def plot_scaling(eff_df: pd.DataFrame, out_dir: Path):
    """Log-scale latency vs dataset size."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = eff_df["n_samples"].values
    for method, col, c in [("TDA", "tda_ms_per_sample", COLORS["tda"]),
                            ("FreeTTA", "ftta_ms_per_sample", COLORS["freetta"])]:
        ax.scatter(ns, eff_df[col], label=method, color=c, s=80, zorder=5)
        # linear fit in log-log
        if len(ns) >= 2:
            m, b = np.polyfit(np.log10(ns), np.log10(eff_df[col]), 1)
            xs = np.logspace(np.log10(ns.min()), np.log10(ns.max()), 100)
            ax.plot(xs, 10**(b + m * np.log10(xs)), "--", color=c, alpha=0.5, lw=1.2,
                    label=f"{method} slope≈{m:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dataset size (samples)"); ax.set_ylabel("ms / sample")
    ax.set_title("Latency Scaling with Dataset Size", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    savefig(fig, out_dir / "efficiency_scaling.png")


# ── Summary markdown ──────────────────────────────────────────────────────────

def write_summary(
    adapt_df: pd.DataFrame,
    ent_df: pd.DataFrame,
    ent_acc_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    eff_df: pd.DataFrame,
    out_dir: Path,
):
    lines = ["# Dynamics Analysis Summary\n"]

    lines.append("## 1. Adaptation Dynamics\n")
    lines.append(adapt_df.round(3).to_markdown() + "\n")

    lines.append("\n## 2. Uncertainty Analysis — Entropy Statistics\n")
    lines.append(ent_df.round(4).to_markdown(index=False) + "\n")
    lines.append("\n### Entropy–Accuracy Calibration\n")
    lines.append(ent_acc_df.round(3).to_markdown(index=False) + "\n")

    lines.append("\n## 3. Distribution Modeling — Geometry\n")
    lines.append(geo_df.round(3).to_markdown() + "\n")

    lines.append("\n## 4. Computational Efficiency\n")
    if not eff_df.empty:
        lines.append(eff_df[["n_samples", "tda_ms_per_sample", "ftta_ms_per_sample",
                              "speedup_ftta_vs_tda"]].round(3).to_markdown() + "\n")
    else:
        lines.append("(comparison_results.json not found)\n")

    (out_dir / "dynamics_analysis_summary.md").write_text("".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  default="outputs/comparative_analysis")
    p.add_argument("--output-dir", default="outputs/dynamics_analysis")
    p.add_argument("--datasets",   nargs="+", default=DATASETS)
    args = p.parse_args()

    input_dir  = PROJECT_ROOT / args.input_dir
    out_dir    = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # load per_sample data
    dfs: dict[str, pd.DataFrame] = {}
    for ds in args.datasets:
        df = load_per_sample(ds, input_dir)
        if df is not None:
            dfs[ds] = df
            print(f"  Loaded {ds}: {len(df)} samples")
        else:
            print(f"  [skip] {ds}: per_sample_metrics.csv not found")

    if not dfs:
        print("No data loaded — exiting.")
        return

    print("\n[1] Adaptation Dynamics")
    adapt_df = plot_rolling_accuracy(dfs, out_dir)
    plot_mu_drift(dfs, out_dir)
    plot_cache_growth(dfs, out_dir)
    adapt_df.to_csv(out_dir / "adaptation_dynamics.csv")
    print(adapt_df.to_string())

    print("\n[2] Uncertainty Analysis")
    ent_df = compute_entropy_stats(dfs, out_dir)
    plot_em_weight_distribution(dfs, out_dir)
    ent_acc_df = plot_entropy_vs_accuracy(dfs, out_dir)
    ent_df.to_csv(out_dir / "uncertainty_entropy_stats.csv", index=False)
    print(ent_df.to_string())

    print("\n[3] Distribution Modeling")
    plot_pca_logit_space(dfs, out_dir)
    geo_df = compute_geometry_table(input_dir)
    if not geo_df.empty:
        plot_oracle_comparison(geo_df, out_dir)
    plot_freetta_distribution_evolution(dfs, out_dir)
    if not geo_df.empty:
        geo_df.to_csv(out_dir / "geometry_alignment.csv")
        print(geo_df.to_string())

    print("\n[4] Computational Efficiency")
    cmp_json = PROJECT_ROOT / "outputs" / "comparison_results.json"
    eff_df = compute_efficiency_table(cmp_json)
    if not eff_df.empty:
        plot_efficiency(eff_df, out_dir)
        plot_scaling(eff_df, out_dir)
        eff_df.to_csv(out_dir / "efficiency_metrics.csv")
        print(eff_df[["n_samples", "tda_ms_per_sample", "ftta_ms_per_sample",
                       "speedup_ftta_vs_tda"]].to_string())

    write_summary(adapt_df, ent_df, ent_acc_df,
                  geo_df if not geo_df.empty else pd.DataFrame(),
                  eff_df, out_dir)

    print(f"\nAll outputs written to: {out_dir}")
    print("  Plots: adaptation_rolling_accuracy.png, adaptation_mu_drift.png, "
          "adaptation_cache_growth.png")
    print("         uncertainty_entropy_distributions.png, uncertainty_em_weight.png, "
          "uncertainty_entropy_accuracy_curve.png")
    print("         distribution_pca_<ds>.png, distribution_oracle_geometry.png, "
          "distribution_freetta_evolution.png")
    print("         efficiency_timing.png, efficiency_scaling.png")


if __name__ == "__main__":
    main()
