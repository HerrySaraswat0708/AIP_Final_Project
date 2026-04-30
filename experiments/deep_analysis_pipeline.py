"""
Deep Analysis Pipeline – Sections 1–13
Works from existing per_sample_metrics.csv, logits.npz, freetta_internal.npz.
Outputs all required plots to outputs/deep_analysis/.
"""
from __future__ import annotations

import sys
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
COMP    = ROOT / "outputs" / "comparative_analysis"
OUT     = ROOT / "outputs" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["caltech", "dtd", "eurosat", "pets", "imagenet"]
DS_LABEL = {
    "caltech":  "Caltech-101",
    "dtd":      "DTD",
    "eurosat":  "EuroSAT",
    "pets":     "Oxford Pets",
    "imagenet": "ImageNetV2",
}
COLORS = {"clip": "#636363", "tda": "#1f77b4", "freetta": "#d62728"}
C_CLIP, C_TDA, C_FT = COLORS["clip"], COLORS["tda"], COLORS["freetta"]

NUM_CLASSES = {
    "caltech": 100, "dtd": 47, "eurosat": 10, "pets": 37, "imagenet": 1000
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ── loaders ────────────────────────────────────────────────────────────────────

def load_ps(ds: str) -> pd.DataFrame:
    return pd.read_csv(COMP / ds / "per_sample_metrics.csv")

def load_traj(ds: str) -> pd.DataFrame:
    return pd.read_csv(COMP / ds / "trajectory_metrics.csv")

def load_logits(ds: str) -> dict:
    p = COMP / ds / "logits.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.files}

def load_freetta_internal(ds: str) -> dict:
    p = COMP / ds / "freetta_internal.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    return {k: d[k] for k in d.files}

def load_failure(ds: str) -> pd.DataFrame:
    p = COMP / ds / "failure_buckets.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # drop duplicate header rows
    df = df[df["bucket"] != "bucket"]
    df["count"] = df["count"].astype(int)
    df["rate"]  = df["rate"].astype(float)
    return df

def load_all_ps() -> dict[str, pd.DataFrame]:
    return {ds: load_ps(ds) for ds in DATASETS}

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)

def savefig(name: str):
    p = OUT / name
    plt.savefig(p, dpi=150)
    plt.close("all")
    print(f"  saved: {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – CORE METRICS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def sec1_metrics_validation(all_ps: dict):
    print("\n[S1] Core metrics validation …")

    rows = []
    for ds, df in all_ps.items():
        n = len(df)
        C = NUM_CLASSES[ds]

        clip_acc  = df["clip_correct"].mean() * 100
        tda_acc   = df["tda_correct"].mean()  * 100
        ft_acc    = df["freetta_correct"].mean() * 100

        # change rate
        tda_cr  = df["tda_changed_prediction"].mean()
        ft_cr   = df["freetta_changed_prediction"].mean()

        # BFP
        tda_flips   = df["tda_beneficial_flip"].sum() + df["tda_harmful_flip"].sum()
        ft_flips    = df["freetta_beneficial_flip"].sum() + df["freetta_harmful_flip"].sum()
        tda_bfp  = df["tda_beneficial_flip"].sum() / tda_flips  if tda_flips  > 0 else float("nan")
        ft_bfp   = df["freetta_beneficial_flip"].sum() / ft_flips if ft_flips > 0 else float("nan")

        # entropy
        tda_ent = df["tda_entropy"].mean()
        ft_ent  = df["freetta_entropy"].mean()

        # GAS
        gas = df["geometry_alignment_score"].iloc[0]
        oracle_c = df["oracle_centroid_acc"].iloc[0]
        oracle_1nn = df["oracle_1nn_acc"].iloc[0]

        # cache pressure
        total_slots = df["tda_total_cache_slots"].iloc[-1]
        pos_slots   = df["tda_positive_cache_size"].iloc[-1]
        cache_pressure = pos_slots / total_slots if total_slots > 0 else float("nan")

        # mean EM weight
        mean_em = df["freetta_em_weight"].mean()

        rows.append(dict(
            dataset=ds, n=n, C=C,
            clip_acc=clip_acc, tda_acc=tda_acc, ft_acc=ft_acc,
            tda_gain=tda_acc-clip_acc, ft_gain=ft_acc-clip_acc,
            winner="FreeTTA" if ft_acc > tda_acc else "TDA",
            tda_cr=tda_cr, ft_cr=ft_cr,
            tda_bfp=tda_bfp, ft_bfp=ft_bfp,
            tda_ent=tda_ent, ft_ent=ft_ent,
            gas=gas, oracle_c=oracle_c, oracle_1nn=oracle_1nn,
            cache_pressure=cache_pressure,
            mean_em=mean_em,
        ))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "sec1_metrics_summary.csv", index=False)

    # ── Plot: 10 metrics in one figure ─────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    ds_labels = [DS_LABEL[ds] for ds in summary["dataset"]]
    x = np.arange(len(DATASETS))
    w = 0.35

    def bar2(ax, a, b, la, lb, ca, cb, title, ylabel):
        ax.bar(x - w/2, a, w, label=la, color=ca)
        ax.bar(x + w/2, b, w, label=lb, color=cb)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels(ds_labels, rotation=30, ha="right")
        ax.legend(fontsize=8)

    # 1 Accuracy
    bar2(axes[0],
         summary["tda_acc"], summary["ft_acc"],
         "TDA", "FreeTTA", C_TDA, C_FT,
         "1. Accuracy", "Accuracy (%)")
    axes[0].bar(x - w*1.5, summary["clip_acc"], w*0.8, color=C_CLIP, alpha=0.6, label="CLIP")
    axes[0].legend(fontsize=7)

    # 2 Change Rate
    bar2(axes[1], summary["tda_cr"]*100, summary["ft_cr"]*100,
         "TDA", "FreeTTA", C_TDA, C_FT, "2. Change Rate (%)", "%")

    # 3 BFP
    bar2(axes[2], summary["tda_bfp"]*100, summary["ft_bfp"]*100,
         "TDA", "FreeTTA", C_TDA, C_FT, "3. Benef. Flip Precision (%)", "%")

    # 4 Mean Entropy
    bar2(axes[3], summary["tda_ent"], summary["ft_ent"],
         "TDA", "FreeTTA", C_TDA, C_FT, "4. Mean Entropy (fused)", "nats")

    # 5 Gain vs CLIP
    bar2(axes[4], summary["tda_gain"], summary["ft_gain"],
         "TDA", "FreeTTA", C_TDA, C_FT, "5. Gain vs CLIP (%)", "Δ acc (%)")
    axes[4].axhline(0, color="k", lw=0.8)

    # 6 GAS
    axes[5].bar(x, summary["gas"]*100, color="#2ca02c")
    axes[5].set_title("8. Geometry Align. Score (%)")
    axes[5].set_ylabel("GAS = OracleCentroid – Oracle1NN")
    axes[5].set_xticks(x); axes[5].set_xticklabels(ds_labels, rotation=30, ha="right")

    # 7 Cache Pressure
    axes[6].bar(x, summary["cache_pressure"]*100, color=C_TDA)
    axes[6].set_title("9. Cache Fill Rate (%)")
    axes[6].set_ylabel("Pos cache filled / total slots")
    axes[6].set_xticks(x); axes[6].set_xticklabels(ds_labels, rotation=30, ha="right")

    # 8 Mean EM Weight
    axes[7].bar(x, summary["mean_em"], color=C_FT)
    axes[7].set_title("10. Mean EM Weight α_t")
    axes[7].set_ylabel("E[exp(-β·H_norm)]")
    axes[7].set_xticks(x); axes[7].set_xticklabels(ds_labels, rotation=30, ha="right")

    # 9 Oracle accuracy comparison
    axes[8].bar(x - w/2, summary["oracle_c"]*100, w, label="Oracle Centroid", color="#9467bd")
    axes[8].bar(x + w/2, summary["oracle_1nn"]*100, w, label="Oracle 1-NN", color="#8c564b")
    axes[8].set_title("11. Oracle Acc (GAS probe)")
    axes[8].set_ylabel("Accuracy (%)")
    axes[8].set_xticks(x); axes[8].set_xticklabels(ds_labels, rotation=30, ha="right")
    axes[8].legend(fontsize=8)

    # 10 Winner summary
    winners = summary["winner"].value_counts()
    axes[9].bar(winners.index, winners.values, color=[C_FT, C_TDA][:len(winners)])
    axes[9].set_title("Dataset Winners")
    axes[9].set_ylabel("# datasets")

    fig.suptitle("Section 1 – All 10 Core Metrics Across 5 Datasets", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig("sec1_all_metrics.png")

    print(f"  Summary:\n{summary[['dataset','clip_acc','tda_acc','ft_acc','tda_gain','ft_gain','gas']].to_string(index=False)}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – CONTROLLED EXPERIMENT: SAMPLES-PER-CLASS GRID
# ══════════════════════════════════════════════════════════════════════════════

def sec2_controlled_grid(all_ps: dict):
    """Simulate different sample sizes by subsampling the stream."""
    print("\n[S2] Controlled experiment grid (subsampling) …")

    sample_fracs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    results = []

    for ds, df in all_ps.items():
        n_total = len(df)
        for frac in sample_fracs:
            n = max(10, int(n_total * frac))
            sub = df.iloc[:n]
            results.append(dict(
                dataset=ds, frac=frac, n=n,
                clip_acc=sub["clip_correct"].mean()*100,
                tda_acc=sub["tda_correct"].mean()*100,
                ft_acc=sub["freetta_correct"].mean()*100,
                tda_gain=(sub["tda_correct"].mean()-sub["clip_correct"].mean())*100,
                ft_gain=(sub["freetta_correct"].mean()-sub["clip_correct"].mean())*100,
                tda_cr=sub["tda_changed_prediction"].mean()*100,
                ft_cr=sub["freetta_changed_prediction"].mean()*100,
            ))

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / "sec2_sample_grid.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for col, ds in enumerate(DATASETS):
        sub = rdf[rdf["dataset"] == ds]
        ax_acc = axes[0, col]
        ax_gain = axes[1, col]

        ax_acc.plot(sub["frac"]*100, sub["clip_acc"], "o-", color=C_CLIP, label="CLIP")
        ax_acc.plot(sub["frac"]*100, sub["tda_acc"],  "s-", color=C_TDA,  label="TDA")
        ax_acc.plot(sub["frac"]*100, sub["ft_acc"],   "^-", color=C_FT,   label="FreeTTA")
        ax_acc.set_title(DS_LABEL[ds])
        ax_acc.set_xlabel("Stream fraction (%)")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=7)
        ax_acc.grid(alpha=0.3)

        ax_gain.plot(sub["frac"]*100, sub["tda_gain"], "s-", color=C_TDA,  label="TDA gain")
        ax_gain.plot(sub["frac"]*100, sub["ft_gain"],  "^-", color=C_FT,   label="FreeTTA gain")
        ax_gain.axhline(0, color="k", lw=0.8, ls="--")
        ax_gain.set_xlabel("Stream fraction (%)")
        ax_gain.set_ylabel("Gain vs CLIP (%)")
        ax_gain.legend(fontsize=7)
        ax_gain.grid(alpha=0.3)

    fig.suptitle("Section 2 – Accuracy vs. Stream Fraction (Controlled Sample Grid)", fontsize=13)
    plt.tight_layout()
    savefig("sec2_sample_grid.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – ADAPTATION DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def sec3_adaptation_dynamics(all_ps: dict):
    print("\n[S3] Adaptation dynamics …")

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    stats_rows = []

    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        n  = len(df)
        window = max(50, n // 40)

        # rolling accuracy
        roll_clip = df["clip_correct"].rolling(window, min_periods=1).mean() * 100
        roll_tda  = df["tda_correct"].rolling(window, min_periods=1).mean()  * 100
        roll_ft   = df["freetta_correct"].rolling(window, min_periods=1).mean() * 100
        x_idx     = np.arange(n)

        # cumulative gain
        cum_tda_gain = (df["tda_correct"].cumsum() - df["clip_correct"].cumsum())
        cum_ft_gain  = (df["freetta_correct"].cumsum() - df["clip_correct"].cumsum())

        # flip frequency
        flip_rate_tda = df["tda_changed_prediction"].rolling(window, min_periods=1).mean() * 100
        flip_rate_ft  = df["freetta_changed_prediction"].rolling(window, min_periods=1).mean() * 100

        ax1, ax2, ax3 = axes[0, col], axes[1, col], axes[2, col]

        # Row 0: rolling accuracy
        ax1.plot(x_idx, roll_clip, color=C_CLIP, lw=0.8, label="CLIP", alpha=0.8)
        ax1.plot(x_idx, roll_tda,  color=C_TDA,  lw=1.0, label="TDA")
        ax1.plot(x_idx, roll_ft,   color=C_FT,   lw=1.0, label="FreeTTA")
        ax1.set_title(DS_LABEL[ds])
        if col == 0:
            ax1.set_ylabel("Rolling acc (%)")
        ax1.legend(fontsize=7)
        ax1.grid(alpha=0.2)

        # Row 1: cumulative gain
        ax2.plot(x_idx, cum_tda_gain, color=C_TDA, label="TDA")
        ax2.plot(x_idx, cum_ft_gain,  color=C_FT,  label="FreeTTA")
        ax2.axhline(0, color="k", lw=0.7)
        if col == 0:
            ax2.set_ylabel("Cumul. gain over CLIP")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.2)

        # Row 2: flip/change rate
        ax3.plot(x_idx, flip_rate_tda, color=C_TDA, label="TDA changes")
        ax3.plot(x_idx, flip_rate_ft,  color=C_FT,  label="FreeTTA changes")
        ax3.set_xlabel("Sample index")
        if col == 0:
            ax3.set_ylabel("Change rate (%)")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.2)

        # Break-even (first sample where cumulative gain > 0)
        def break_even(cum_gain):
            pos = np.where(cum_gain.values > 0)[0]
            return int(pos[0]) if len(pos) else n

        tda_be  = break_even(cum_tda_gain)
        ft_be   = break_even(cum_ft_gain)

        # early vs late accuracy (first/last 20%)
        split = n // 5
        early_tda  = df["tda_correct"].iloc[:split].mean()
        late_tda   = df["tda_correct"].iloc[-split:].mean()
        early_ft   = df["freetta_correct"].iloc[:split].mean()
        late_ft    = df["freetta_correct"].iloc[-split:].mean()

        stats_rows.append(dict(
            dataset=ds, n=n, window=window,
            tda_break_even=tda_be, ft_break_even=ft_be,
            early_tda_acc=early_tda*100, late_tda_acc=late_tda*100,
            early_ft_acc=early_ft*100,   late_ft_acc=late_ft*100,
            tda_adaptation_speed=late_tda-early_tda,
            ft_adaptation_speed=late_ft-early_ft,
        ))

    fig.suptitle("Section 3 – Adaptation Dynamics Over Stream", fontsize=13)
    plt.tight_layout()
    savefig("sec3_adaptation_dynamics.png")

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT / "sec3_adaptation_stats.csv", index=False)
    print(f"  Break-even samples: TDA={stats['tda_break_even'].values}, FreeTTA={stats['ft_break_even'].values}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – UNCERTAINTY / ENTROPY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec4_uncertainty_analysis(all_ps: dict):
    print("\n[S4] Uncertainty / entropy analysis …")

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
    rows = []

    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        clip_ent = df["clip_entropy"].values
        ft_em    = df["freetta_em_weight"].values

        # Entropy buckets (CLIP entropy used as uncertainty signal)
        C = NUM_CLASSES[ds]
        H_max = math.log(max(C, 2))
        norm_h = clip_ent / H_max

        lo_mask = norm_h < 0.33
        mi_mask = (norm_h >= 0.33) & (norm_h < 0.67)
        hi_mask = norm_h >= 0.67

        def bucket_stats(mask):
            n = mask.sum()
            if n == 0:
                return 0, float("nan"), float("nan"), float("nan")
            sub = df[mask]
            return n, sub["clip_correct"].mean()*100, sub["tda_correct"].mean()*100, sub["freetta_correct"].mean()*100

        n_lo, lo_clip, lo_tda, lo_ft = bucket_stats(lo_mask)
        n_mi, mi_clip, mi_tda, mi_ft = bucket_stats(mi_mask)
        n_hi, hi_clip, hi_tda, hi_ft = bucket_stats(hi_mask)

        rows.append(dict(
            dataset=ds,
            n_lo=n_lo, lo_clip=lo_clip, lo_tda=lo_tda, lo_ft=lo_ft,
            n_mi=n_mi, mi_clip=mi_clip, mi_tda=mi_tda, mi_ft=mi_ft,
            n_hi=n_hi, hi_clip=hi_clip, hi_tda=hi_tda, hi_ft=hi_ft,
            corr_ent_clip_acc=sp_stats.spearmanr(clip_ent, df["clip_correct"].values)[0],
            corr_ent_ft_gain=sp_stats.spearmanr(clip_ent, (df["freetta_correct"]-df["clip_correct"]).values)[0],
            mean_em=ft_em.mean(), std_em=ft_em.std(),
        ))

        ax1 = axes[0, col]
        ax2 = axes[1, col]
        ax3 = axes[2, col]

        # Row 0: entropy histogram
        ax1.hist(norm_h, bins=40, color=C_CLIP, edgecolor="none", alpha=0.8)
        ax1.axvline(0.33, color="orange", ls="--", lw=1)
        ax1.axvline(0.67, color="red",    ls="--", lw=1)
        ax1.set_title(DS_LABEL[ds])
        if col == 0:
            ax1.set_ylabel("Count")
        ax1.set_xlabel("Norm. entropy H/log(C)")

        # Row 1: accuracy by entropy bucket
        buckets  = ["Low\nH<0.33", "Mid\n0.33–0.67", "High\nH>0.67"]
        clip_acc = [lo_clip, mi_clip, hi_clip]
        tda_acc  = [lo_tda,  mi_tda,  hi_tda]
        ft_acc   = [lo_ft,   mi_ft,   hi_ft]
        b_x      = np.arange(3)
        for vals, col_c, lbl in [(clip_acc, C_CLIP, "CLIP"), (tda_acc, C_TDA, "TDA"), (ft_acc, C_FT, "FreeTTA")]:
            ax2.plot(b_x, vals, "o-", color=col_c, label=lbl)
        ax2.set_xticks(b_x); ax2.set_xticklabels(buckets)
        if col == 0:
            ax2.set_ylabel("Accuracy (%)")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        # Row 2: EM weight distribution
        ax3.hist(ft_em, bins=40, color=C_FT, edgecolor="none", alpha=0.8)
        ax3.set_xlabel("EM weight α_t")
        if col == 0:
            ax3.set_ylabel("Count")

    fig.suptitle("Section 4 – Entropy/Uncertainty Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec4_uncertainty_analysis.png")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec4_entropy_buckets.csv", index=False)
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – DISTRIBUTION MODELING / PCA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec5_distribution_modeling(all_ps: dict):
    print("\n[S5] Distribution modeling / PCA of logits …")

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    rows = []

    for col, ds in enumerate(DATASETS):
        logits_d = load_logits(ds)
        if not logits_d:
            print(f"  skipping {ds} – no logits.npz")
            continue

        clip_log = logits_d["clip_logits"].astype(np.float32)   # (N, C)
        tda_log  = logits_d["tda_logits"].astype(np.float32)
        ft_log   = logits_d["freetta_logits"].astype(np.float32)
        labels   = logits_d["labels"].astype(int)

        # PCA on CLIP logits
        N, C = clip_log.shape
        pca  = PCA(n_components=2, random_state=42)
        clip_2d = pca.fit_transform(clip_log)

        # colour by label; just plot up to 2000 pts
        idx = np.random.RandomState(0).choice(N, min(2000, N), replace=False)

        ax1, ax2 = axes[0, col], axes[1, col]

        sc = ax1.scatter(clip_2d[idx, 0], clip_2d[idx, 1],
                         c=labels[idx], cmap="tab20", s=4, alpha=0.6, linewidths=0)
        ax1.set_title(f"{DS_LABEL[ds]}\nCLIP logit PCA")
        ax1.axis("off")

        # FreeTTA centroid drift: use freetta_mu_drift from per_sample
        df = all_ps[ds]
        drift = df["freetta_mu_drift"].values
        ax2.hist(drift, bins=40, color=C_FT, edgecolor="none", alpha=0.8)
        ax2.set_title(f"{DS_LABEL[ds]}\nFreeTTA μ drift")
        ax2.set_xlabel("‖μ_t – μ_0‖ (cosine)")
        if col == 0:
            ax2.set_ylabel("Count")

        # Cosine similarity: TDA vs FreeTTA logit direction
        tda_prob  = softmax(tda_log[idx])
        ft_prob   = softmax(ft_log[idx])
        clip_prob = softmax(clip_log[idx])

        tda_div  = float(np.mean(np.sum(np.abs(tda_prob - clip_prob), axis=-1)))
        ft_div   = float(np.mean(np.sum(np.abs(ft_prob  - clip_prob), axis=-1)))

        rows.append(dict(
            dataset=ds,
            pca_var_ratio=float(pca.explained_variance_ratio_.sum()),
            mean_drift=float(drift.mean()), max_drift=float(drift.max()),
            tda_l1_div_from_clip=tda_div, ft_l1_div_from_clip=ft_div,
        ))

    fig.suptitle("Section 5 – Distribution Modeling: Logit PCA + Centroid Drift", fontsize=13)
    plt.tight_layout()
    savefig("sec5_distribution_modeling.png")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec5_distribution_stats.csv", index=False)
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – COMPUTATIONAL EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════

def sec6_efficiency():
    print("\n[S6] Computational efficiency …")

    # Load latency data from existing latency_metrics.csv
    rows = []
    for ds in DATASETS:
        p = COMP / ds / "latency_metrics.csv"
        if p.exists():
            d = pd.read_csv(p).iloc[0].to_dict()
            rows.append({**d, "dataset": ds})

    if not rows:
        print("  no latency data found; using summary")
        ldf = pd.read_csv(COMP / "latency_metrics.csv")
        rows = ldf.to_dict("records")

    ldf = pd.DataFrame(rows)
    ldf.to_csv(OUT / "sec6_latency.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ds_labels = [DS_LABEL.get(r["dataset"], r["dataset"]) for _, r in ldf.iterrows()]
    x = np.arange(len(ldf))
    w = 0.35

    if "tda_break_even_vs_clip" in ldf.columns:
        axes[0].bar(x - w/2, ldf["tda_break_even_vs_clip"], w, color=C_TDA, label="TDA")
        axes[0].bar(x + w/2, ldf["freetta_break_even_vs_clip"], w, color=C_FT,  label="FreeTTA")
        axes[0].set_title("Break-Even Sample Index\n(vs CLIP baseline)")
        axes[0].set_ylabel("Sample index")
        axes[0].set_xticks(x); axes[0].set_xticklabels(ds_labels, rotation=30, ha="right")
        axes[0].legend()

    if "tda_break_even_ratio" in ldf.columns:
        axes[1].bar(x - w/2, ldf["tda_break_even_ratio"]*100, w, color=C_TDA, label="TDA")
        axes[1].bar(x + w/2, ldf["freetta_break_even_ratio"]*100, w, color=C_FT,  label="FreeTTA")
        axes[1].set_title("Break-Even / Stream Length (%)")
        axes[1].set_ylabel("Fraction of stream (%)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(ds_labels, rotation=30, ha="right")
        axes[1].legend()

    # Memory: TDA uses C × K × D, FreeTTA uses C × D
    ns = [NUM_CLASSES[r["dataset"]] for _, r in ldf.iterrows()]
    D  = 512
    K_pos, K_neg = 3, 2
    tda_mem  = [c*(K_pos+K_neg)*D*4/1024 for c in ns]  # KB
    ft_mem   = [c*D*4/1024 for c in ns]
    axes[2].bar(x - w/2, tda_mem, w, color=C_TDA, label=f"TDA (K={K_pos+K_neg})")
    axes[2].bar(x + w/2, ft_mem,  w, color=C_FT,  label="FreeTTA (μ only)")
    axes[2].set_title("Estimated Memory (KB)\n512-d features")
    axes[2].set_ylabel("KB")
    axes[2].set_xticks(x); axes[2].set_xticklabels(ds_labels, rotation=30, ha="right")
    axes[2].legend()

    fig.suptitle("Section 6 – Computational Efficiency", fontsize=13)
    plt.tight_layout()
    savefig("sec6_efficiency.png")
    return ldf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – ARCHITECTURE MECHANISM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec7_architecture_analysis(all_ps: dict):
    print("\n[S7] Architecture mechanism analysis …")

    rows = []
    for ds, df in all_ps.items():
        n = len(df)
        # TDA mechanism: how does cache size influence gain?
        pos_cache = df["tda_positive_cache_size"].values
        neg_cache = df["tda_negative_cache_size"].values
        gate_open = df["tda_negative_gate_open"].values

        # correlation: cache size vs per-sample correctness delta
        tda_delta = df["tda_correct"].values - df["clip_correct"].values
        ft_delta  = df["freetta_correct"].values - df["clip_correct"].values

        corr_pos_cache_tda = sp_stats.spearmanr(pos_cache, tda_delta)[0]
        corr_neg_gate_tda  = sp_stats.spearmanr(gate_open, tda_delta)[0]

        # FreeTTA mechanism: how does EM weight relate to update quality?
        em_w   = df["freetta_em_weight"].values
        mu_upd = df["freetta_mu_update_norm"].values
        mu_drift = df["freetta_mu_drift"].values

        corr_em_ft_delta   = sp_stats.spearmanr(em_w, ft_delta)[0]
        corr_drift_ft_acc  = sp_stats.spearmanr(mu_drift, df["freetta_correct"].values)[0]

        # Cache pressure: fraction of positive cache that is filled
        total_slots = df["tda_total_cache_slots"].iloc[-1]
        pos_final   = df["tda_positive_cache_size"].iloc[-1]
        pressure    = pos_final / total_slots if total_slots > 0 else float("nan")

        rows.append(dict(
            dataset=ds,
            corr_pos_cache_tda_delta=corr_pos_cache_tda,
            corr_neg_gate_tda_delta=corr_neg_gate_tda,
            corr_em_weight_ft_delta=corr_em_ft_delta,
            corr_drift_ft_acc=corr_drift_ft_acc,
            cache_pressure=pressure,
            neg_gate_rate=gate_open.mean(),
            mean_mu_drift=mu_drift[-100:].mean() if len(mu_drift) > 100 else mu_drift.mean(),
        ))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec7_mechanism_correlations.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        em_w   = df["freetta_em_weight"].values
        drift  = df["freetta_mu_drift"].values
        ft_acc = df["freetta_correct"].values
        tda_acc = df["tda_correct"].values
        clip_acc = df["clip_correct"].values
        pos_cache = df["tda_positive_cache_size"].values

        # Row 0: EM weight vs FreeTTA correctness (binned)
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(em_w, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins)-2)
        bin_ft  = [ft_acc[bin_idx == b].mean()*100 if (bin_idx==b).sum()>0 else float("nan")
                   for b in range(len(bins)-1)]
        bin_mid = (bins[:-1] + bins[1:]) / 2
        axes[0, col].plot(bin_mid, bin_ft, "o-", color=C_FT, lw=2)
        axes[0, col].set_title(DS_LABEL[ds])
        if col == 0:
            axes[0, col].set_ylabel("FreeTTA acc (%) by EM weight")
        axes[0, col].set_xlabel("EM weight bin")
        axes[0, col].grid(alpha=0.3)

        # Row 1: TDA positive cache size vs rolling accuracy gain
        window = max(50, len(df) // 40)
        roll_tda_gain = (df["tda_correct"] - df["clip_correct"]).rolling(window, min_periods=1).mean() * 100
        axes[1, col].plot(np.arange(len(df)), pos_cache / pos_cache.max(), color=C_TDA, alpha=0.6, label="pos cache (norm)")
        ax_twin = axes[1, col].twinx()
        ax_twin.plot(np.arange(len(df)), roll_tda_gain, color="orange", lw=1, label="TDA gain")
        ax_twin.axhline(0, color="k", lw=0.5)
        axes[1, col].set_xlabel("Sample index")
        if col == 0:
            axes[1, col].set_ylabel("Cache size (norm)")
            ax_twin.set_ylabel("Rolling gain (%)")

    fig.suptitle("Section 7 – Architecture Mechanism Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec7_architecture_analysis.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – CONFIDENCE-BASED SUBSET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec8_confidence_subset(all_ps: dict):
    print("\n[S8] Confidence-based subset analysis …")

    rows = []
    for ds, df in all_ps.items():
        conf = df["clip_confidence"].values
        q33 = np.percentile(conf, 33)
        q67 = np.percentile(conf, 67)
        lo_mask = conf <= q33
        mi_mask = (conf > q33) & (conf <= q67)
        hi_mask = conf > q67

        for bucket, mask, label in [("low", lo_mask, "Low conf"),
                                    ("mid", mi_mask, "Mid conf"),
                                    ("high", hi_mask, "High conf")]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            rows.append(dict(
                dataset=ds, confidence_bucket=bucket, n=len(sub),
                clip_acc=sub["clip_correct"].mean()*100,
                tda_acc=sub["tda_correct"].mean()*100,
                ft_acc=sub["freetta_correct"].mean()*100,
                tda_gain=(sub["tda_correct"].mean()-sub["clip_correct"].mean())*100,
                ft_gain=(sub["freetta_correct"].mean()-sub["clip_correct"].mean())*100,
                tda_cr=sub["tda_changed_prediction"].mean()*100,
                ft_cr=sub["freetta_changed_prediction"].mean()*100,
                tda_bfp=(sub["tda_beneficial_flip"].sum() /
                         max(1, sub["tda_beneficial_flip"].sum()+sub["tda_harmful_flip"].sum()))*100,
                ft_bfp=(sub["freetta_beneficial_flip"].sum() /
                        max(1, sub["freetta_beneficial_flip"].sum()+sub["freetta_harmful_flip"].sum()))*100,
            ))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec8_confidence_subset.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for col, ds in enumerate(DATASETS):
        sub = rdf[rdf["dataset"] == ds]
        buckets = ["Low\n(≤33%)", "Mid\n(33–67%)", "High\n(≥67%)"]

        ax_acc  = axes[0, col]
        ax_gain = axes[1, col]

        for meth, col_c in [("clip_acc", C_CLIP), ("tda_acc", C_TDA), ("ft_acc", C_FT)]:
            label = meth.split("_")[0].upper() if meth != "ft_acc" else "FreeTTA"
            ax_acc.plot(np.arange(3), sub[meth].values, "o-", color=col_c, label=label)
        ax_acc.set_title(DS_LABEL[ds])
        ax_acc.set_xticks(np.arange(3)); ax_acc.set_xticklabels(buckets)
        if col == 0:
            ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=7); ax_acc.grid(alpha=0.3)

        ax_gain.plot(np.arange(3), sub["tda_gain"].values, "s-", color=C_TDA, label="TDA gain")
        ax_gain.plot(np.arange(3), sub["ft_gain"].values,  "^-", color=C_FT,  label="FreeTTA gain")
        ax_gain.axhline(0, color="k", lw=0.7, ls="--")
        ax_gain.set_xticks(np.arange(3)); ax_gain.set_xticklabels(buckets)
        if col == 0:
            ax_gain.set_ylabel("Gain vs CLIP (%)")
        ax_gain.legend(fontsize=7); ax_gain.grid(alpha=0.3)

    fig.suptitle("Section 8 – Confidence-Based Subset Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec8_confidence_subset.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – SAMPLES-PER-CLASS REGIME
# ══════════════════════════════════════════════════════════════════════════════

def sec9_spc_regime(all_ps: dict):
    print("\n[S9] Samples-per-class regime …")

    rows = []
    for ds, df in all_ps.items():
        C   = NUM_CLASSES[ds]
        N   = len(df)
        spc_full = N / C

        # Simulate different SPC by random subsampling at fixed fracs
        rng = np.random.RandomState(42)
        for frac in [0.05, 0.1, 0.2, 0.5, 1.0]:
            n_sub = max(C, int(N * frac))
            idx   = rng.choice(N, n_sub, replace=False)
            sub   = df.iloc[idx]
            spc   = n_sub / C

            # Centroid quality proxy: mean FreeTTA confidence
            ft_conf = sub["freetta_confidence"].mean()
            rows.append(dict(
                dataset=ds, spc=spc, n_sub=n_sub, frac=frac,
                clip_acc=sub["clip_correct"].mean()*100,
                tda_acc=sub["tda_correct"].mean()*100,
                ft_acc=sub["freetta_correct"].mean()*100,
                ft_gain=(sub["freetta_correct"].mean()-sub["clip_correct"].mean())*100,
                tda_gain=(sub["tda_correct"].mean()-sub["clip_correct"].mean())*100,
                ft_conf_proxy=ft_conf,
                mean_em=sub["freetta_em_weight"].mean(),
                mean_drift=sub["freetta_mu_drift"].mean(),
            ))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec9_spc_regime.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for col, ds in enumerate(DATASETS):
        sub = rdf[rdf["dataset"] == ds].sort_values("spc")
        ax1 = axes[0, col]
        ax2 = axes[1, col]

        ax1.semilogx(sub["spc"], sub["tda_gain"],  "s-", color=C_TDA, label="TDA gain")
        ax1.semilogx(sub["spc"], sub["ft_gain"],   "^-", color=C_FT,  label="FreeTTA gain")
        ax1.axhline(0, color="k", lw=0.7, ls="--")
        ax1.set_title(DS_LABEL[ds])
        ax1.set_xlabel("Samples per class (log)")
        if col == 0:
            ax1.set_ylabel("Gain vs CLIP (%)")
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

        ax2.semilogx(sub["spc"], sub["mean_drift"], "o-", color=C_FT, label="Mean drift")
        ax2.set_xlabel("Samples per class (log)")
        if col == 0:
            ax2.set_ylabel("FreeTTA μ drift")
        ax2.grid(alpha=0.3)

    fig.suptitle("Section 9 – Samples-per-Class Regime Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec9_spc_regime.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – INITIALIZATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec10_initialization(all_ps: dict):
    print("\n[S10] Initialization analysis …")

    rows = []
    for ds, df in all_ps.items():
        fi = load_freetta_internal(ds)
        if not fi:
            continue

        drift = fi.get("mu_drift_by_class", np.array([[]])).mean(axis=0)
        prior_ent = fi.get("prior_entropy", np.array([]))
        sigma_tr  = fi.get("sigma_trace", np.array([]))

        # How quickly does drift stabilize? (early vs late 25%)
        n = len(df)
        split = n // 4
        early_drift = df["freetta_mu_drift"].iloc[:split].mean()
        late_drift  = df["freetta_mu_drift"].iloc[-split:].mean()

        # Convergence: is drift monotonically increasing or does it plateau?
        d_vals = df["freetta_mu_drift"].values
        # simple: fit a log curve to drift
        x_log = np.log(np.arange(1, n+1))
        if x_log.std() > 0:
            r, _ = sp_stats.pearsonr(x_log, d_vals)
        else:
            r = float("nan")

        rows.append(dict(
            dataset=ds,
            early_drift=early_drift, late_drift=late_drift,
            drift_log_corr=r,
            convergence_ratio=late_drift/max(early_drift, 1e-8),
            final_prior_entropy=df["freetta_prior_entropy"].iloc[-1],
            final_sigma_trace=df["freetta_sigma_trace"].iloc[-1],
        ))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "sec10_initialization.csv", index=False)

    fig, axes = plt.subplots(2, len(rdf), figsize=(4*len(rdf), 8))
    if len(rdf) == 1:
        axes = axes.reshape(2, 1)

    for col, (_, row) in enumerate(rdf.iterrows()):
        ds = row["dataset"]
        df = all_ps[ds]
        drift = df["freetta_mu_drift"].values
        prior_ent = df["freetta_prior_entropy"].values

        ax1 = axes[0, col]
        ax2 = axes[1, col]

        ax1.plot(drift, color=C_FT, lw=0.8)
        ax1.set_title(DS_LABEL[ds])
        if col == 0:
            ax1.set_ylabel("Centroid drift ‖μ_t – μ_0‖")
        ax1.set_xlabel("Sample")
        ax1.grid(alpha=0.3)

        ax2.plot(prior_ent, color="#9467bd", lw=0.8)
        if col == 0:
            ax2.set_ylabel("Prior entropy H(Ny)")
        ax2.set_xlabel("Sample")
        ax2.grid(alpha=0.3)

    fig.suptitle("Section 10 – Initialization & Convergence Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec10_initialization.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – GAS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def sec11_gas_validation(all_ps: dict):
    print("\n[S11] GAS validation …")

    agg = []
    for ds, df in all_ps.items():
        gas = df["geometry_alignment_score"].iloc[0]
        oracle_c   = df["oracle_centroid_acc"].iloc[0] * 100
        oracle_1nn = df["oracle_1nn_acc"].iloc[0] * 100
        ft_gain    = (df["freetta_correct"].mean() - df["clip_correct"].mean()) * 100
        tda_gain   = (df["tda_correct"].mean() - df["clip_correct"].mean()) * 100
        ft_vs_tda  = ft_gain - tda_gain
        agg.append(dict(
            dataset=ds,
            gas=gas, oracle_c=oracle_c, oracle_1nn=oracle_1nn,
            ft_gain=ft_gain, tda_gain=tda_gain, ft_vs_tda=ft_vs_tda,
        ))

    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(OUT / "sec11_gas_validation.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ds_labels = [DS_LABEL[d] for d in agg_df["dataset"]]
    x = np.arange(len(agg_df))
    w = 0.35

    # GAS vs FreeTTA-TDA delta
    axes[0].scatter(agg_df["gas"]*100, agg_df["ft_vs_tda"], s=80, c=[
        C_FT if v > 0 else C_TDA for v in agg_df["ft_vs_tda"]], zorder=3)
    for _, r in agg_df.iterrows():
        axes[0].annotate(DS_LABEL[r["dataset"]][:3], (r["gas"]*100, r["ft_vs_tda"]),
                         fontsize=8, xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("GAS = OracleCentroid – Oracle1NN (%)")
    axes[0].set_ylabel("FreeTTA gain – TDA gain (Δ%)")
    axes[0].set_title("GAS vs FreeTTA Advantage")
    axes[0].grid(alpha=0.3)

    # Oracle accuracy comparison
    axes[1].bar(x - w/2, agg_df["oracle_c"],   w, color="#9467bd", label="Oracle Centroid")
    axes[1].bar(x + w/2, agg_df["oracle_1nn"],  w, color="#8c564b", label="Oracle 1-NN")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ds_labels, rotation=25, ha="right")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Oracle Geometry Probes")
    axes[1].legend()

    # GAS bar
    axes[2].bar(x, agg_df["gas"]*100, color="#2ca02c")
    axes[2].set_xticks(x); axes[2].set_xticklabels(ds_labels, rotation=25, ha="right")
    axes[2].set_ylabel("GAS (%)")
    axes[2].set_title("GAS per Dataset\n(+ = centroid better, – = 1-NN better)")

    fig.suptitle("Section 11 – Geometry Alignment Score Validation", fontsize=13)
    plt.tight_layout()
    savefig("sec11_gas_validation.png")
    return agg_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 – FAILURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def sec12_failure_analysis(all_ps: dict):
    print("\n[S12] Failure analysis …")

    BUCKET_LABELS = {
        "clip_wrong_tda_wrong_freetta_correct":  "CLIP✗ TDA✗ FT✓",
        "clip_wrong_tda_correct_freetta_wrong":  "CLIP✗ TDA✓ FT✗",
        "clip_correct_tda_wrong_freetta_correct":"CLIP✓ TDA✗ FT✓",
        "clip_correct_tda_correct_freetta_wrong":"CLIP✓ TDA✓ FT✗",
        "all_wrong":                             "All ✗",
        "all_correct":                           "All ✓",
    }

    records = []
    for ds, df in all_ps.items():
        n = len(df)

        buckets = {
            "clip_wrong_tda_wrong_freetta_correct":
                ((df["clip_correct"]==0) & (df["tda_correct"]==0) & (df["freetta_correct"]==1)),
            "clip_wrong_tda_correct_freetta_wrong":
                ((df["clip_correct"]==0) & (df["tda_correct"]==1) & (df["freetta_correct"]==0)),
            "clip_correct_tda_wrong_freetta_correct":
                ((df["clip_correct"]==1) & (df["tda_correct"]==0) & (df["freetta_correct"]==1)),
            "clip_correct_tda_correct_freetta_wrong":
                ((df["clip_correct"]==1) & (df["tda_correct"]==1) & (df["freetta_correct"]==0)),
            "all_wrong":
                ((df["clip_correct"]==0) & (df["tda_correct"]==0) & (df["freetta_correct"]==0)),
            "all_correct":
                ((df["clip_correct"]==1) & (df["tda_correct"]==1) & (df["freetta_correct"]==1)),
        }

        for bname, mask in buckets.items():
            cnt  = int(mask.sum())
            rate = cnt / n
            # entropy/confidence stats in this bucket
            sub  = df[mask]
            mean_ent  = sub["clip_entropy"].mean() if cnt > 0 else float("nan")
            mean_conf = sub["clip_confidence"].mean() if cnt > 0 else float("nan")
            records.append(dict(
                dataset=ds, bucket=bname, count=cnt, rate=rate,
                mean_clip_entropy=mean_ent, mean_clip_confidence=mean_conf,
            ))

    rdf = pd.DataFrame(records)
    rdf.to_csv(OUT / "sec12_failure_analysis.csv", index=False)

    # Stacked bar per dataset
    bucket_order = list(BUCKET_LABELS.keys())
    pal = ["#d62728","#1f77b4","#ff7f0e","#9467bd","#2ca02c","#8c564b"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: rate per bucket per dataset
    ax = axes[0]
    width = 0.15
    for bi, (bname, blabel) in enumerate(BUCKET_LABELS.items()):
        vals = [rdf[(rdf["dataset"]==ds) & (rdf["bucket"]==bname)]["rate"].values[0]*100
                if len(rdf[(rdf["dataset"]==ds) & (rdf["bucket"]==bname)])>0 else 0
                for ds in DATASETS]
        x = np.arange(len(DATASETS))
        ax.bar(x + bi*width - 2.5*width, vals, width, label=blabel, color=pal[bi])
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=25, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Failure Bucket Distribution")
    ax.legend(fontsize=7, loc="upper right")

    # Right: stacked bar (normalized)
    pivot = rdf.pivot_table(index="dataset", columns="bucket", values="rate", aggfunc="sum").fillna(0)
    pivot = pivot[bucket_order] if all(b in pivot.columns for b in bucket_order) else pivot
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct[[c for c in pivot_pct.columns if c in bucket_order]].plot(
        kind="bar", stacked=True, ax=axes[1], color=pal[:len(pivot_pct.columns)],
        legend=True)
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Share of all samples (%)")
    axes[1].set_title("Stacked Failure Composition")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend([BUCKET_LABELS.get(c, c) for c in pivot_pct.columns], fontsize=6, loc="upper right")

    fig.suptitle("Section 12 – Failure Analysis", fontsize=13)
    plt.tight_layout()
    savefig("sec12_failure_analysis.png")
    return rdf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 – STANDARD REQUIRED PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def sec13_standard_plots(all_ps: dict, sec1: pd.DataFrame, sec3: pd.DataFrame,
                          sec4: pd.DataFrame, sec11: pd.DataFrame, sec12: pd.DataFrame):
    print("\n[S13] Generating all required standard plots …")

    # 1. accuracy_vs_samples.png
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        n  = len(df)
        w  = max(50, n // 40)
        roll_tda = df["tda_correct"].rolling(w, min_periods=1).mean()*100
        roll_ft  = df["freetta_correct"].rolling(w, min_periods=1).mean()*100
        roll_cl  = df["clip_correct"].rolling(w, min_periods=1).mean()*100
        axes[col].plot(roll_cl, color=C_CLIP, lw=0.8, alpha=0.7, label="CLIP")
        axes[col].plot(roll_tda, color=C_TDA, lw=1, label="TDA")
        axes[col].plot(roll_ft,  color=C_FT,  lw=1, label="FreeTTA")
        axes[col].set_title(DS_LABEL[ds])
        axes[col].set_xlabel("Samples")
        if col == 0: axes[col].set_ylabel("Rolling acc (%)")
        axes[col].legend(fontsize=7)
        axes[col].grid(alpha=0.2)
    plt.suptitle("Accuracy vs. Number of Processed Samples")
    plt.tight_layout()
    savefig("accuracy_vs_samples.png")

    # 2. change_rate_vs_accuracy.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, meth, col_c in [(axes[0], "tda", C_TDA), (axes[1], "freetta", C_FT)]:
        cr_col  = "tda_cr" if meth == "tda" else "ft_cr"
        acc_col = "tda_acc" if meth == "tda" else "ft_acc"
        xs = [sec1[sec1["dataset"]==ds][cr_col].values[0]*100 for ds in DATASETS]
        ys = [sec1[sec1["dataset"]==ds][acc_col].values[0] for ds in DATASETS]
        ax.scatter(xs, ys, s=90, color=col_c, zorder=3)
        for i, ds in enumerate(DATASETS):
            ax.annotate(DS_LABEL[ds][:3], (xs[i], ys[i]), fontsize=8, xytext=(3,2), textcoords="offset points")
        ax.set_xlabel("Change Rate (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{meth.upper()} – Change Rate vs Accuracy")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("change_rate_vs_accuracy.png")

    # 3. bfp_vs_thresholds.png  (proxy: BFP across datasets)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DATASETS))
    w = 0.35
    tda_bfp = sec1["tda_bfp"].values * 100
    ft_bfp  = sec1["ft_bfp"].values * 100
    ax.bar(x - w/2, tda_bfp, w, color=C_TDA, label="TDA BFP")
    ax.bar(x + w/2, ft_bfp,  w, color=C_FT,  label="FreeTTA BFP")
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=25, ha="right")
    ax.set_ylabel("Beneficial Flip Precision (%)")
    ax.set_title("Beneficial Flip Precision by Dataset")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("bfp_vs_thresholds.png")

    # 4. entropy_confidence_plots.png
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        axes[0, col].hist(df["clip_entropy"], bins=40, color=C_CLIP, alpha=0.7, label="CLIP")
        axes[0, col].hist(df["tda_entropy"],  bins=40, color=C_TDA,  alpha=0.5, label="TDA")
        axes[0, col].hist(df["freetta_entropy"], bins=40, color=C_FT, alpha=0.5, label="FT")
        axes[0, col].set_title(DS_LABEL[ds])
        if col == 0: axes[0, col].set_ylabel("Count")
        axes[0, col].legend(fontsize=7)
        axes[1, col].scatter(df["clip_confidence"], df["clip_correct"], s=1, alpha=0.3, color=C_CLIP)
        axes[1, col].set_xlabel("CLIP confidence")
        if col == 0: axes[1, col].set_ylabel("Correct (0/1)")
    axes[0, 0].set_ylabel("Entropy distribution")
    plt.suptitle("Entropy & Confidence Distributions")
    plt.tight_layout()
    savefig("entropy_confidence_plots.png")

    # 5. break_even_plots.png
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DATASETS))
    w = 0.35
    ax.bar(x - w/2, sec3["tda_break_even"].values,  w, color=C_TDA, label="TDA break-even")
    ax.bar(x + w/2, sec3["ft_break_even"].values,   w, color=C_FT,  label="FreeTTA break-even")
    ax.set_xticks(x); ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=25, ha="right")
    ax.set_ylabel("Sample index of first net gain over CLIP")
    ax.set_title("Break-Even Point: When Adaptation Pays Off")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("break_even_plots.png")

    # 6. disagreement_analysis.png
    dis_rows = []
    for ds, df in all_ps.items():
        mask = df["tda_pred"] != df["freetta_pred"]
        dis_n = mask.sum()
        dis_rows.append(dict(
            dataset=ds,
            disagreement_rate=mask.mean()*100,
            tda_wins_disagreement=(df[mask]["tda_correct"] > df[mask]["freetta_correct"]).mean()*100 if dis_n>0 else 0,
            ft_wins_disagreement=(df[mask]["freetta_correct"] > df[mask]["tda_correct"]).mean()*100 if dis_n>0 else 0,
        ))
    dis_df = pd.DataFrame(dis_rows)
    dis_df.to_csv(OUT / "disagreement_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(DATASETS))
    axes[0].bar(x, dis_df["disagreement_rate"], color="#7f7f7f")
    axes[0].set_xticks(x); axes[0].set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=25, ha="right")
    axes[0].set_ylabel("Disagreement Rate (%)"); axes[0].set_title("TDA vs FreeTTA Disagreement Rate")
    axes[1].bar(x - 0.2, dis_df["tda_wins_disagreement"],  0.35, color=C_TDA, label="TDA wins")
    axes[1].bar(x + 0.2, dis_df["ft_wins_disagreement"],   0.35, color=C_FT,  label="FreeTTA wins")
    axes[1].set_xticks(x); axes[1].set_xticklabels([DS_LABEL[d] for d in DATASETS], rotation=25, ha="right")
    axes[1].set_ylabel("% of disagreements"); axes[1].set_title("Who Wins on Disagreements?")
    axes[1].legend()
    plt.tight_layout()
    savefig("disagreement_analysis.png")

    # 7. failure_buckets.png  (from sec12)
    fig, ax = plt.subplots(figsize=(10, 5))
    bucket_order = ["all_correct", "clip_wrong_tda_wrong_freetta_correct",
                    "clip_wrong_tda_correct_freetta_wrong",
                    "clip_correct_tda_wrong_freetta_correct",
                    "clip_correct_tda_correct_freetta_wrong", "all_wrong"]
    pal = ["#2ca02c","#d62728","#1f77b4","#ff7f0e","#9467bd","#8c564b"]
    pivot = (sec12.pivot_table(index="dataset", columns="bucket", values="rate", aggfunc="sum")
             .fillna(0))
    cols = [b for b in bucket_order if b in pivot.columns]
    pivot[cols].plot(kind="bar", stacked=True, ax=ax,
                     color=pal[:len(cols)], legend=True)
    ax.set_xticklabels([DS_LABEL[d] for d in pivot.index], rotation=25, ha="right")
    ax.set_ylabel("Sample fraction"); ax.set_title("Failure Bucket Composition")
    short = {"all_correct":"All✓","clip_wrong_tda_wrong_freetta_correct":"CLIP✗TDA✗FT✓",
             "clip_wrong_tda_correct_freetta_wrong":"CLIP✗TDA✓FT✗",
             "clip_correct_tda_wrong_freetta_correct":"CLIP✓TDA✗FT✓",
             "clip_correct_tda_correct_freetta_wrong":"CLIP✓TDA✓FT✗","all_wrong":"All✗"}
    ax.legend([short.get(c,c) for c in cols], fontsize=7, loc="upper right")
    plt.tight_layout()
    savefig("failure_buckets.png")

    # 8. gas_vs_performance.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gas_vals = sec11["gas"].values * 100
    ft_g  = sec11["ft_gain"].values
    tda_g = sec11["tda_gain"].values
    axes[0].scatter(gas_vals, ft_g,  s=90, c=C_FT,  label="FreeTTA gain", zorder=3)
    axes[0].scatter(gas_vals, tda_g, s=90, c=C_TDA, label="TDA gain", marker="s", zorder=3)
    for i, ds in enumerate(DATASETS):
        axes[0].annotate(DS_LABEL[ds][:3], (gas_vals[i], ft_g[i]), fontsize=8)
    axes[0].axhline(0, color="k", lw=0.7, ls="--")
    axes[0].set_xlabel("GAS (%)"); axes[0].set_ylabel("Accuracy gain vs CLIP (%)")
    axes[0].set_title("GAS vs Accuracy Gain"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].scatter(gas_vals, sec11["ft_vs_tda"].values, s=90, c="#e377c2", zorder=3)
    for i, ds in enumerate(DATASETS):
        axes[1].annotate(DS_LABEL[ds][:3], (gas_vals[i], sec11["ft_vs_tda"].values[i]), fontsize=8)
    axes[1].axhline(0, color="k", lw=0.7, ls="--")
    axes[1].set_xlabel("GAS (%)"); axes[1].set_ylabel("FreeTTA – TDA gain (%)")
    axes[1].set_title("GAS vs FreeTTA Relative Advantage"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    savefig("gas_vs_performance.png")

    # 9. cache_pressure_plots.png
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, ds in enumerate(["caltech", "eurosat"]):
        df = all_ps[ds]
        ax = axes[col]
        ax.plot(df["tda_positive_cache_size"], color=C_TDA, lw=0.9, label="Pos cache")
        ax.plot(df["tda_negative_cache_size"], color="orange", lw=0.9, label="Neg cache", alpha=0.8)
        ax.set_title(f"{DS_LABEL[ds]} – Cache Growth")
        ax.set_xlabel("Sample"); ax.set_ylabel("Cache size (entries)")
        ax.legend(); ax.grid(alpha=0.2)
    plt.suptitle("Cache Pressure Plots")
    plt.tight_layout()
    savefig("cache_pressure_plots.png")

    # 10. em_weight_analysis.png
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for col, ds in enumerate(DATASETS):
        df = all_ps[ds]
        axes[col].scatter(range(len(df)), df["freetta_em_weight"], s=1, alpha=0.3, color=C_FT)
        axes[col].set_title(DS_LABEL[ds]); axes[col].set_xlabel("Sample")
        if col == 0: axes[col].set_ylabel("EM weight α_t")
    plt.suptitle("Mean EM Weight (α_t) Over Stream")
    plt.tight_layout()
    savefig("em_weight_analysis.png")

    print("  All 10 standard plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════

def write_master_summary(all_ps: dict, sec1: pd.DataFrame, sec3: pd.DataFrame,
                          sec4: pd.DataFrame, sec11: pd.DataFrame):
    rows = []
    for ds, df in all_ps.items():
        n = len(df)
        C = NUM_CLASSES[ds]

        s1 = sec1[sec1["dataset"]==ds].iloc[0]
        s3 = sec3[sec3["dataset"]==ds].iloc[0]
        s4 = sec4[sec4["dataset"]==ds].iloc[0]
        s11 = sec11[sec11["dataset"]==ds].iloc[0]

        rows.append(dict(
            dataset=ds, n_samples=n, n_classes=C,
            clip_acc=s1["clip_acc"], tda_acc=s1["tda_acc"], ft_acc=s1["ft_acc"],
            tda_gain=s1["tda_gain"], ft_gain=s1["ft_gain"], winner=s1["winner"],
            tda_change_rate=s1["tda_cr"], ft_change_rate=s1["ft_cr"],
            tda_bfp=s1["tda_bfp"], ft_bfp=s1["ft_bfp"],
            tda_break_even=s3["tda_break_even"], ft_break_even=s3["ft_break_even"],
            tda_early_acc=s3["early_tda_acc"], tda_late_acc=s3["late_tda_acc"],
            ft_early_acc=s3["early_ft_acc"],   ft_late_acc=s3["late_ft_acc"],
            corr_ent_clip=s4["corr_ent_clip_acc"], mean_em=s4["mean_em"],
            gas=s11["gas"], oracle_c=s11["oracle_c"], oracle_1nn=s11["oracle_1nn"],
            ft_vs_tda_gain=s11["ft_vs_tda"],
        ))

    master = pd.DataFrame(rows)
    master.to_csv(OUT / "master_summary.csv", index=False)
    print(f"\n  Master summary saved: {OUT}/master_summary.csv")
    print(master[["dataset","clip_acc","tda_acc","ft_acc","tda_gain","ft_gain","winner","gas"]].to_string(index=False))
    return master


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Deep Analysis Pipeline – 14-Section TTA Study")
    print("=" * 70)

    all_ps = load_all_ps()

    sec1  = sec1_metrics_validation(all_ps)
    sec2  = sec2_controlled_grid(all_ps)
    sec3  = sec3_adaptation_dynamics(all_ps)
    sec4  = sec4_uncertainty_analysis(all_ps)
    sec5  = sec5_distribution_modeling(all_ps)
    sec6  = sec6_efficiency()
    sec7  = sec7_architecture_analysis(all_ps)
    sec8  = sec8_confidence_subset(all_ps)
    sec9  = sec9_spc_regime(all_ps)
    sec10 = sec10_initialization(all_ps)
    sec11 = sec11_gas_validation(all_ps)
    sec12 = sec12_failure_analysis(all_ps)
    sec13_standard_plots(all_ps, sec1, sec3, sec4, sec11, sec12)
    master = write_master_summary(all_ps, sec1, sec3, sec4, sec11)

    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {OUT}")
    print(f"  Plots: {len(list(OUT.glob('*.png')))}, CSVs: {len(list(OUT.glob('*.csv')))}")
    print("=" * 70)

if __name__ == "__main__":
    main()
