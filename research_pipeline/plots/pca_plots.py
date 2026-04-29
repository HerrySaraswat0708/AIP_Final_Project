from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA as _PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _pca2d(mat: np.ndarray) -> np.ndarray:
    if _HAS_SKLEARN:
        return _PCA(n_components=2, svd_solver="randomized", random_state=0).fit_transform(mat)
    centered = mat - mat.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ Vt[:2].T).astype(np.float32)


def plot_pca_logits(
    df: pd.DataFrame,
    prob_arrays: dict[str, np.ndarray],
    output_dir: Path,
    max_arrows: int = 300,
) -> None:
    """
    Section 9 — PCA of CLIP / TDA / FreeTTA probability vectors.

    Panel 1: All three methods' points, coloured by correctness.
    Panel 2: Arrow plot: CLIP → TDA (blue), CLIP → FreeTTA (orange).
    Panel 3: Special cases (TDA-only correct, FreeTTA-only correct, both wrong).
    Panel 4: Sample categories in CLIP space.
    """
    clip_p = prob_arrays.get("clip")
    tda_p = prob_arrays.get("tda")
    ftta_p = prob_arrays.get("freetta")
    if clip_p is None or tda_p is None or ftta_p is None:
        return

    for ds, g in df.groupby("dataset", observed=True):
        out = output_dir / ds
        out.mkdir(parents=True, exist_ok=True)
        idxs = g.index.to_numpy()

        c = clip_p[idxs]
        t = tda_p[idxs]
        f = ftta_p[idxs]

        all_pts = np.concatenate([c, t, f], axis=0).astype(np.float32)
        proj = _pca2d(all_pts)
        n = len(idxs)
        c2d, t2d, f2d = proj[:n], proj[n:2*n], proj[2*n:]

        # Category masks
        tda_correct = g["tda_correct"].to_numpy().astype(bool)
        ftta_correct = g["freetta_correct"].to_numpy().astype(bool)
        clip_correct = g["clip_correct"].to_numpy().astype(bool)
        tda_changed = g["tda_changed"].to_numpy().astype(bool)
        ftta_changed = g["freetta_changed"].to_numpy().astype(bool)

        tda_wins = tda_correct & ~ftta_correct
        ftta_wins = ftta_correct & ~tda_correct
        both_wrong = ~tda_correct & ~ftta_correct
        all_correct_mask = tda_correct & ftta_correct & clip_correct
        changed_mask = tda_changed | ftta_changed

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"PCA Logit Visualization — {ds.upper()}", fontsize=13)

        # Panel 1: CLIP/TDA/FreeTTA points colored by correctness
        for method, pts, correct_m in (
            ("clip",    c2d, g["clip_correct"].to_numpy().astype(bool)),
            ("tda",     t2d, tda_correct),
            ("freetta", f2d, ftta_correct),
        ):
            markers = {"clip": "o", "tda": "s", "freetta": "^"}[method]
            axes[0, 0].scatter(pts[correct_m, 0],  pts[correct_m, 1],
                               s=8, alpha=0.15, marker=markers, color="#2ca25f", label=f"{method} ✓")
            axes[0, 0].scatter(pts[~correct_m, 0], pts[~correct_m, 1],
                               s=8, alpha=0.15, marker=markers, color="#de2d26", label=f"{method} ✗")
        axes[0, 0].set_title("Points: CLIP/TDA/FreeTTA (green=correct, red=wrong)")
        axes[0, 0].grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            fontsize=7, ncols=2
        )

        # Panel 2: Arrows CLIP→TDA and CLIP→FreeTTA
        axes[0, 1].scatter(c2d[:, 0], c2d[:, 1], s=6, alpha=0.06, color="#7f7f7f")
        cand_idx = np.flatnonzero(changed_mask)
        entropy_scores = g["clip_entropy"].to_numpy()[changed_mask]
        if len(cand_idx):
            ordered = cand_idx[np.argsort(entropy_scores)[::-1]]
            chosen = ordered[:max_arrows]
            for i in chosen:
                if tda_changed[i]:
                    axes[0, 1].annotate(
                        "", xy=(t2d[i, 0], t2d[i, 1]),
                        xytext=(c2d[i, 0], c2d[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.2, lw=0.7),
                    )
                if ftta_changed[i]:
                    axes[0, 1].annotate(
                        "", xy=(f2d[i, 0], f2d[i, 1]),
                        xytext=(c2d[i, 0], c2d[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="#ff7f0e", alpha=0.2, lw=0.7),
                    )
        axes[0, 1].set_title("Movement Arrows: CLIP→TDA (blue), CLIP→FreeTTA (orange)")
        axes[0, 1].grid(alpha=0.2)
        from matplotlib.patches import Patch
        axes[0, 1].legend(
            handles=[Patch(color="#1f77b4", label="→TDA"), Patch(color="#ff7f0e", label="→FreeTTA")],
            fontsize=8,
        )

        # Panel 3: Special cases
        axes[1, 0].scatter(t2d[tda_wins, 0], t2d[tda_wins, 1], s=14, alpha=0.7,
                           color="#1f77b4", label="TDA only correct")
        axes[1, 0].scatter(f2d[ftta_wins, 0], f2d[ftta_wins, 1], s=14, alpha=0.7,
                           color="#ff7f0e", label="FreeTTA only correct")
        axes[1, 0].scatter(c2d[both_wrong, 0], c2d[both_wrong, 1], s=8, alpha=0.18,
                           color="#444", label="Both wrong")
        axes[1, 0].set_title("Special Cases")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.2)

        # Panel 4: Category map in CLIP space
        cat_map = {
            "all_correct":      (all_correct_mask,    "#4daf4a"),
            "tda_only":         (tda_wins,             "#1f77b4"),
            "freetta_only":     (ftta_wins,             "#ff7f0e"),
            "both_wrong":       (both_wrong,            "#333333"),
            "different_preds":  (tda_changed | ftta_changed, "#984ea3"),
        }
        for label, (mask, color) in cat_map.items():
            if mask.any():
                axes[1, 1].scatter(c2d[mask, 0], c2d[mask, 1], s=10, alpha=0.35,
                                   color=color, label=label)
        axes[1, 1].set_title("Sample Categories in CLIP Space")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(out / "pca_logit_visualization.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
