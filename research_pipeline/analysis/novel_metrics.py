from __future__ import annotations

"""
Section 10 — Novel Metrics

CE   Correction Efficiency       = beneficial_flips / total_changes
OER  Overconfidence Error Rate   = wrong predictions with confidence > 0.9 / total_wrong
LMM  Logit Movement Magnitude    = ||prob_method - prob_clip||_2
SS   Stability Score             = 1 / (1 + std(rolling_accuracy))
"""

import numpy as np
import pandas as pd

from research_pipeline.analysis.metrics import ADAPT_METHODS, METHODS


def compute_correction_efficiency(flip_df: pd.DataFrame) -> pd.DataFrame:
    """
    CE = n_beneficial / (n_beneficial + n_harmful + n_other_changed_wrong)
    Values close to 1.0 mean almost all changes are beneficial.
    """
    df = flip_df.copy()
    total_changes = (
        df["n_beneficial"] + df["n_harmful"] + df["n_other_changed_wrong"]
    ).clip(lower=1)
    df["correction_efficiency"] = df["n_beneficial"] / total_changes
    return df[["dataset", "method", "n_beneficial", "n_harmful",
               "correction_efficiency"]].copy()


def compute_overconfidence_error_rate(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    OER = |{wrong predictions with confidence > threshold}| / |wrong predictions|

    High OER: method makes confident mistakes — a calibration problem.
    """
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        for m in METHODS:
            wrong = g[g[f"{m}_correct"] == 0]
            n_wrong = max(len(wrong), 1)
            n_overconf_wrong = int((wrong[f"{m}_confidence"] > threshold).sum())
            rows.append({
                "dataset": ds,
                "method": m,
                "n_wrong": len(wrong),
                "n_overconfident_wrong": n_overconf_wrong,
                "oer": n_overconf_wrong / n_wrong,
            })
    return pd.DataFrame(rows)



def compute_stability_score(traj: pd.DataFrame) -> pd.DataFrame:
    """
    SS = 1 / (1 + std(rolling_accuracy))

    High SS = steady rolling accuracy = stable adaptation.
    """
    rows = []
    for ds, g in traj.groupby("dataset", observed=True):
        row = {"dataset": ds}
        for m in METHODS:
            col = f"rolling_{m}_acc"
            if col in g.columns:
                std = float(g[col].std(ddof=0))
                row[f"{m}_stability"] = 1.0 / (1.0 + std)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_lmm_per_dataset(
    df: pd.DataFrame,
    per_ds_arrs: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Compute LMM per-dataset (handles different class counts per dataset).
    per_ds_arrs: {dataset: {method: (N, C) array}}
    """
    rows = []
    for ds, arrs in per_ds_arrs.items():
        ds_df = df[df["dataset"] == ds].reset_index(drop=True)
        if ds_df.empty or "clip" not in arrs:
            continue
        for m in ADAPT_METHODS:
            if m not in arrs:
                continue
            clip_p = arrs["clip"]
            m_p = arrs[m]
            if clip_p.shape != m_p.shape:
                continue
            lmm = np.linalg.norm(m_p - clip_p, axis=1)

            beneficial = ds_df[f"{m}_beneficial"].to_numpy().astype(bool)
            harmful = ds_df[f"{m}_harmful"].to_numpy().astype(bool)
            unchanged = ds_df[f"{m}_changed"].to_numpy() == 0

            rows.append({
                "dataset": ds,
                "method": m,
                "lmm_mean_all": float(lmm.mean()),
                "lmm_std_all": float(lmm.std()),
                "lmm_mean_beneficial": float(lmm[beneficial].mean()) if beneficial.any() else float("nan"),
                "lmm_mean_harmful": float(lmm[harmful].mean()) if harmful.any() else float("nan"),
                "lmm_mean_unchanged": float(lmm[unchanged].mean()) if unchanged.any() else float("nan"),
            })
    return pd.DataFrame(rows)


def compute_all_novel_metrics(
    df: pd.DataFrame,
    flip_df: pd.DataFrame,
    traj: pd.DataFrame,
    per_ds_arrs: dict[str, dict[str, np.ndarray]],
) -> dict[str, pd.DataFrame]:
    """Compute and return all Section 10 metrics."""
    return {
        "correction_efficiency": compute_correction_efficiency(flip_df),
        "overconfidence_error_rate": compute_overconfidence_error_rate(df),
        "logit_movement_magnitude": compute_lmm_per_dataset(df, per_ds_arrs),
        "stability_score": compute_stability_score(traj),
    }
