from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_freetta_internals(df: pd.DataFrame, logit_arrs: dict, output_dir: Path) -> None:
    """Section 6 — FreeTTA: mu drift, prior entropy, sigma trace."""
    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        x = g["stream_step"].to_numpy() + 1

        mu_drift_matrix = logit_arrs.get("mu_drift_by_class")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"FreeTTA Internal Statistics — {ds.upper()}", fontsize=13)

        if mu_drift_matrix is not None and mu_drift_matrix.shape[0] == len(g):
            mean_drift = mu_drift_matrix.mean(axis=1)
            axes[0].plot(x, mean_drift, label="Mean drift", lw=2, color="#111")
            # Top-5 most drifted classes
            final_drift = mu_drift_matrix[-1]
            top_k = min(5, mu_drift_matrix.shape[1])
            for ci in np.argsort(final_drift)[-top_k:]:
                axes[0].plot(x, mu_drift_matrix[:, ci], lw=1, alpha=0.7, label=f"class {ci}")
        else:
            axes[0].plot(x, g["freetta_mu_drift"], label="Mean drift", lw=2, color="#111")

        axes[0].set_ylabel("||μ_y(t) − μ_y(0)||")
        axes[0].set_title("Class Mean Drift")
        axes[0].legend(fontsize=8, ncols=2)
        axes[0].grid(alpha=0.25)

        axes[1].plot(x, g["freetta_prior_entropy"], lw=2, color="#4c72b0", label="H(π)")
        axes[1].set_ylabel("Prior Entropy H(π)")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        axes[2].plot(x, g["freetta_sigma_trace"], lw=2, color="#55a868", label="Σ trace proxy")
        axes[2].set_ylabel("Covariance Trace Proxy")
        axes[2].set_xlabel("Stream Step")
        axes[2].grid(alpha=0.25)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(out / "freetta_internals.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Conf-gated FreeTTA comparison
        if "conf_ftta_mu_drift" in g.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(x, g["freetta_mu_drift"], label="FreeTTA (std)", lw=2, color="#55a868")
            ax2.plot(x, g["conf_ftta_mu_drift"], label="ConfGated FreeTTA", lw=2, color="#c44e52", ls="--")
            ax2.set_ylabel("Mean μ Drift")
            ax2.set_xlabel("Stream Step")
            ax2.set_title(f"Mean Drift: FreeTTA vs Conf-Gated — {ds.upper()}")
            ax2.legend()
            ax2.grid(alpha=0.25)
            fig2.tight_layout()
            fig2.savefig(out / "freetta_drift_comparison.png", dpi=180, bbox_inches="tight")
            plt.close(fig2)


def plot_tda_internals(df: pd.DataFrame, output_dir: Path) -> None:
    """Section 6 — TDA: cache sizes, gate activation."""
    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        x = g["stream_step"].to_numpy() + 1
        win = min(50, max(5, len(g) // 20))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"TDA Internal Statistics — {ds.upper()}", fontsize=13)

        axes[0].plot(x, g["tda_pos_cache_size"], label="Positive cache", lw=2, color="#4c72b0")
        axes[0].plot(x, g["tda_neg_cache_size"], label="Negative cache", lw=2, color="#dd8452")
        if "ent_tda_pos_cache_size" in g.columns:
            axes[0].plot(x, g["ent_tda_pos_cache_size"], label="EntropyGated pos cache",
                         lw=1.5, color="#8172b2", ls="--")
        axes[0].set_ylabel("Cache Size")
        axes[0].set_title("Cache Growth Over Stream")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        gate_roll = g["tda_gate_open"].astype(float).rolling(win, min_periods=1).mean()
        gate_cum = g["tda_gate_open"].astype(float).expanding().mean()
        axes[1].plot(x, gate_roll, label=f"Rolling gate rate (w={win})", lw=2)
        axes[1].plot(x, gate_cum, label="Cumulative gate rate", lw=2, ls="--")
        axes[1].set_ylabel("Negative Gate Activation Rate")
        axes[1].set_xlabel("Stream Step")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(out / "tda_internals.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
