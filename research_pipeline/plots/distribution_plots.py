from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("clip", "tda", "freetta", "conf_ftta", "ent_tda", "hybrid")
METHOD_COLORS = {
    "clip":      "#4c72b0",
    "tda":       "#dd8452",
    "freetta":   "#55a868",
    "conf_ftta": "#c44e52",
    "ent_tda":   "#8172b2",
    "hybrid":    "#937860",
}


def plot_entropy_confidence(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Section 4 — Per-dataset violin/box plots of entropy (correct vs wrong)
    and confidence histogram.
    """
    for ds, g in df.groupby("dataset", observed=True):
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)

        # ── Entropy box plots ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(f"Entropy & Confidence — {ds.upper()}", fontsize=13)

        core_methods = ("clip", "tda", "freetta")
        for ax, subset, title in zip(
            axes[:2],
            ("correct", "wrong"),
            ("Entropy on Correct", "Entropy on Wrong"),
        ):
            data = [
                g.loc[g[f"{m}_correct"] == (1 if subset == "correct" else 0),
                       f"{m}_entropy"].to_numpy()
                for m in core_methods
            ]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            for patch, m in zip(bp["boxes"], core_methods):
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_alpha(0.7)
            ax.set_xticklabels([m.upper() for m in core_methods])
            ax.set_title(title)
            ax.set_ylabel("Entropy")
            ax.grid(axis="y", alpha=0.25)

        # Confidence histogram
        bins = np.linspace(0.0, 1.0, 30)
        for m in core_methods:
            axes[2].hist(
                g[f"{m}_confidence"].to_numpy(),
                bins=bins,
                alpha=0.45,
                density=True,
                label=m.upper(),
                color=METHOD_COLORS[m],
            )
        axes[2].set_title("Confidence Distribution")
        axes[2].set_xlabel("Max Softmax")
        axes[2].set_ylabel("Density")
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(out / "entropy_confidence.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # ── OER comparison (all 6 methods) ────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for i, m in enumerate(METHODS):
            wrong = g[g[f"{m}_correct"] == 0]
            if len(wrong) == 0:
                continue
            oer = float((wrong[f"{m}_confidence"] > 0.9).mean())
            ax2.bar(i, oer, color=METHOD_COLORS.get(m, "gray"), label=m)
        ax2.set_xticks(range(len(METHODS)))
        ax2.set_xticklabels(METHODS, rotation=15)
        ax2.set_ylabel("OER (confidence > 0.9 | wrong)")
        ax2.set_title(f"Overconfidence Error Rate — {ds.upper()}")
        ax2.axhline(0.1, color="red", linestyle="--", linewidth=1, alpha=0.5, label="OER=0.1")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)
        ax2.set_ylim(0, min(1.05, ax2.get_ylim()[1] + 0.05))
        fig2.tight_layout()
        fig2.savefig(out / "oer_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig2)


def plot_lmm_analysis(df: pd.DataFrame, prob_arrays: dict, output_dir: Path) -> None:
    """
    Section 10 LMM — Logit Movement Magnitude by flip category.
    """
    clip_p = prob_arrays.get("clip")
    if clip_p is None:
        return

    adapt_methods = ("tda", "freetta", "conf_ftta", "ent_tda", "hybrid")

    for ds, g in df.groupby("dataset", observed=True):
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        idxs = g.index.to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Logit Movement Magnitude — {ds.upper()}", fontsize=13)

        lmm_means = {"beneficial": [], "harmful": [], "unchanged": []}
        method_labels = []

        for m in adapt_methods:
            if m not in prob_arrays:
                continue
            m_p = prob_arrays[m]
            lmm = np.linalg.norm(m_p[idxs] - clip_p[idxs], axis=1)

            ben = g[f"{m}_beneficial"].to_numpy().astype(bool)
            harm = g[f"{m}_harmful"].to_numpy().astype(bool)
            unch = g[f"{m}_changed"].to_numpy() == 0

            lmm_means["beneficial"].append(float(lmm[ben].mean()) if ben.any() else 0.0)
            lmm_means["harmful"].append(float(lmm[harm].mean()) if harm.any() else 0.0)
            lmm_means["unchanged"].append(float(lmm[unch].mean()) if unch.any() else 0.0)
            method_labels.append(m)

        x = np.arange(len(method_labels))
        w = 0.25
        axes[0].bar(x - w, lmm_means["beneficial"], w, label="Beneficial", color="#4daf4a")
        axes[0].bar(x,     lmm_means["harmful"],    w, label="Harmful",    color="#e41a1c")
        axes[0].bar(x + w, lmm_means["unchanged"],  w, label="Unchanged",  color="#bdbdbd")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(method_labels, rotation=15)
        axes[0].set_ylabel("Mean LMM (||p_method - p_clip||)")
        axes[0].set_title("LMM by Flip Category")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.25)

        # Scatter: LMM vs entropy for TDA/FreeTTA
        for m, color in (("tda", "#dd8452"), ("freetta", "#55a868")):
            if m not in prob_arrays:
                continue
            m_p = prob_arrays[m]
            lmm = np.linalg.norm(m_p[idxs] - clip_p[idxs], axis=1)
            correct = g[f"{m}_correct"].to_numpy().astype(bool)
            axes[1].scatter(
                g["clip_entropy"].to_numpy()[correct],
                lmm[correct],
                s=5, alpha=0.2, color=color, label=f"{m} correct",
            )
            axes[1].scatter(
                g["clip_entropy"].to_numpy()[~correct],
                lmm[~correct],
                s=5, alpha=0.2, color=color, marker="x", label=f"{m} wrong",
            )
        axes[1].set_xlabel("CLIP Entropy")
        axes[1].set_ylabel("LMM")
        axes[1].set_title("LMM vs CLIP Entropy")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(out / "lmm_analysis.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
