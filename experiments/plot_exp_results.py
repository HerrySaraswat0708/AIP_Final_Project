"""Generate plots for Experiments 1 and 2."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

OUT = Path("outputs/deep_analysis")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

COLORS = {"caltech": "#2196F3", "dtd": "#FF5722", "eurosat": "#4CAF50", "pets": "#9C27B0"}
MARKERS = {"caltech": "o", "dtd": "s", "eurosat": "^", "pets": "D"}

# ─── EXP 1A: Threshold sweep ──────────────────────────────────────────────────
e1 = pd.read_csv(OUT / "exp1_negative_cache_sweep.csv")
thresh = e1[e1.experiment == "thresh"].copy()
alpha_sw = e1[e1.experiment == "neg_alpha"].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Experiment 1: Negative-Cache Sensitivity Analysis", fontweight="bold", fontsize=13)

# Panel A – threshold configs
ax = axes[0]
thresh_labels = ["Baseline\n[0.2-0.50]", "All-open\n[0.0-1.01]",
                 "High-H\n[0.5-1.01]", "Very-high\n[0.8-1.01]"]
x = np.arange(len(thresh_labels))
ds_list = ["caltech", "dtd", "eurosat", "pets"]
for ds in ds_list:
    sub = thresh[thresh.dataset == ds]
    ax.plot(x, sub.acc.values, marker=MARKERS[ds], color=COLORS[ds], label=ds.upper(), linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(thresh_labels, fontsize=9)
ax.set_xlabel("Threshold Configuration")
ax.set_ylabel("Accuracy (%)")
ax.set_title("(A) Accuracy vs Entropy Threshold Window")
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel B – neg_alpha sweep
ax = axes[1]
for ds in ds_list:
    sub = alpha_sw[alpha_sw.dataset == ds]
    ax.plot(sub.neg_alpha.values, sub.acc.values, marker=MARKERS[ds],
            color=COLORS[ds], label=ds.upper(), linewidth=2)
ax.set_xlabel("neg_alpha (negative cache weight)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("(B) Accuracy vs neg_alpha (gate fully open)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Annotate neg_alpha=0 as best
ax.axvline(0.117, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.text(0.13, ax.get_ylim()[0] + 0.5, "paper\ndefault", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(OUT / "exp1_negative_cache_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: exp1_negative_cache_sweep.png")

# ─── EXP 1A: neg_used bar ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("Experiment 1: Negative Cache Entries Used (Threshold Sweep)", fontweight="bold")
w = 0.2
x = np.arange(len(thresh_labels))
for i, ds in enumerate(ds_list):
    sub = thresh[thresh.dataset == ds]
    ax.bar(x + i*w, sub.neg_cache_used.values, width=w, color=COLORS[ds], label=ds.upper(), alpha=0.85)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(thresh_labels, fontsize=9)
ax.set_ylabel("Negative Cache Entries Used")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "exp1a_neg_cache_used.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: exp1a_neg_cache_used.png")

# ─── EXP 2: K majority vote ───────────────────────────────────────────────────
e2 = pd.read_csv(OUT / "exp2_majority_vote.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Experiment 2: K-NN Majority Vote vs TDA Affinity-Sum", fontweight="bold", fontsize=13)
axes = axes.flatten()

for idx, ds in enumerate(ds_list):
    sub = e2[e2.dataset == ds]
    ax = axes[idx]
    ax.plot(sub.K, sub.tda_affinity_acc, marker="o", color="#2196F3",
            label="TDA Affinity-Sum", linewidth=2.5)
    ax.plot(sub.K, sub.majority_vote_acc, marker="s", color="#FF5722",
            label="K-NN Majority Vote", linewidth=2.5, linestyle="--")
    ax.axhline(sub.clip_acc.iloc[0], color="gray", linestyle=":", linewidth=1.5, label="CLIP Baseline")
    ax.set_title(f"{ds.upper()}", fontweight="bold")
    ax.set_xlabel("K (neighbors)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(sub.K.values)

plt.tight_layout()
plt.savefig(OUT / "exp2_majority_vote.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: exp2_majority_vote.png")

# ─── EXP 2: MV-TDA delta heatmap style ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for ds in ds_list:
    sub = e2[e2.dataset == ds]
    ax.plot(sub.K, sub.mv_vs_tda, marker=MARKERS[ds], color=COLORS[ds], label=ds.upper(), linewidth=2)
ax.axhline(0, color="black", linewidth=1, linestyle="-")
ax.fill_between(range(1, 22), 0, -100, alpha=0.05, color="red")
ax.set_xlabel("K (neighbors)")
ax.set_ylabel("Majority-Vote Acc − TDA-Affinity Acc (pp)")
ax.set_title("Experiment 2: MV vs Affinity-Sum Gap", fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_xlim(0, 21)
ax.set_ylim(-95, 5)
plt.tight_layout()
plt.savefig(OUT / "exp2_mv_vs_tda_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: exp2_mv_vs_tda_delta.png")

print("\nAll plots saved to", OUT)
