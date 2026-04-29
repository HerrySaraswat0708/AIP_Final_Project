from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METHODS = ("clip", "tda", "freetta", "conf_ftta", "ent_tda", "hybrid")
ADAPT_METHODS = ("tda", "freetta", "conf_ftta", "ent_tda", "hybrid")


# ── Section 2: Baseline Metrics ──────────────────────────────────────────────

def compute_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 accuracy per method per dataset."""
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        row = {"dataset": ds, "n_samples": len(g)}
        for m in METHODS:
            row[f"{m}_acc"] = float(g[f"{m}_correct"].mean())
        for m in ADAPT_METHODS:
            row[f"{m}_gain"] = row[f"{m}_acc"] - row["clip_acc"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def compute_per_class_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class accuracy for all methods."""
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        for cls, cg in g.groupby("label", observed=True):
            row = {"dataset": ds, "class_id": cls, "n_samples": len(cg)}
            for m in METHODS:
                row[f"{m}_acc"] = float(cg[f"{m}_correct"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ── Section 3: Prediction Change Analysis ────────────────────────────────────

def compute_flip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        total = max(len(g), 1)
        for m in ADAPT_METHODS:
            changed = g[f"{m}_changed"] == 1
            beneficial = g[f"{m}_beneficial"] == 1
            harmful = g[f"{m}_harmful"] == 1
            other_chg_wrong = (
                changed & ~beneficial & ~harmful & (g["clip_correct"] == 0)
            )
            unchanged_correct = (g["clip_correct"] == 1) & ~changed
            unchanged_wrong = (g["clip_correct"] == 0) & ~changed
            n_changed = int(changed.sum())
            n_clip_correct = int((g["clip_correct"] == 1).sum())
            rows.append({
                "dataset": ds,
                "method": m,
                "n_samples": total,
                "n_unchanged_correct": int(unchanged_correct.sum()),
                "n_unchanged_wrong": int(unchanged_wrong.sum()),
                "n_beneficial": int(beneficial.sum()),
                "n_harmful": int(harmful.sum()),
                "n_other_changed_wrong": int(other_chg_wrong.sum()),
                "change_rate": float(changed.mean()),
                "beneficial_rate": float(beneficial.mean()),
                "harmful_rate": float(harmful.mean()),
                "net_correction_score": int(beneficial.sum()) - int(harmful.sum()),
                "net_correction_rate": float(
                    (beneficial.sum() - harmful.sum()) / total
                ),
                "correction_efficiency": float(
                    beneficial.sum() / max(n_changed, 1)
                ),
                "harm_rate_on_clip_correct": float(
                    harmful.sum() / max(n_clip_correct, 1)
                ),
            })
    return pd.DataFrame(rows)


# ── Section 4: Entropy & Confidence ──────────────────────────────────────────

def compute_entropy_confidence_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        for m in METHODS:
            for subset, mask in (
                ("all",     np.ones(len(g), dtype=bool)),
                ("correct", g[f"{m}_correct"].to_numpy() == 1),
                ("wrong",   g[f"{m}_correct"].to_numpy() == 0),
            ):
                sub = g.loc[mask]
                if len(sub) == 0:
                    continue
                ent_col = sub[f"{m}_entropy"]
                conf_col = sub[f"{m}_confidence"]
                rows.append({
                    "dataset": ds, "method": m, "subset": subset,
                    "n": len(sub),
                    "mean_entropy": float(ent_col.mean()),
                    "median_entropy": float(ent_col.median()),
                    "std_entropy": float(ent_col.std(ddof=0)),
                    "mean_confidence": float(conf_col.mean()),
                    "median_confidence": float(conf_col.median()),
                    "std_confidence": float(conf_col.std(ddof=0)),
                })
    return pd.DataFrame(rows)


# ── Section 5: Trajectory Analysis ───────────────────────────────────────────

def compute_trajectory(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    frames = []
    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        n = len(g)
        win = min(max(window, 5), n)
        curve: dict[str, object] = {
            "dataset": g["dataset"],
            "stream_step": g["stream_step"],
            "progress": (g["stream_step"] + 1) / max(n, 1),
        }
        for m in METHODS:
            roll = g[f"{m}_correct"].astype(float).rolling(win, min_periods=1).mean()
            curve[f"rolling_{m}_acc"] = roll
            curve[f"rolling_{m}_conf"] = (
                g[f"{m}_confidence"].astype(float).rolling(win, min_periods=1).mean()
            )
            curve[f"rolling_{m}_ent"] = (
                g[f"{m}_entropy"].astype(float).rolling(win, min_periods=1).mean()
            )
        frames.append(pd.DataFrame(curve))
    return pd.concat(frames, ignore_index=True)


def compute_break_even(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in traj.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        n = len(g)
        row: dict = {"dataset": ds}
        for m in ADAPT_METHODS:
            diff = g[f"rolling_{m}_acc"] - g["rolling_clip_acc"]
            positive_idx = np.flatnonzero(diff.to_numpy() > 0)
            be = int(positive_idx[0] + 1) if len(positive_idx) else None
            row[f"{m}_break_even_step"] = be
            row[f"{m}_break_even_ratio"] = (
                be / n if be is not None else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ── Section 6: Internal Method Analysis ──────────────────────────────────────

def compute_freetta_internal(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        rows.append({
            "dataset": ds,
            "mean_em_weight": float(g["freetta_em_weight"].mean()),
            "final_mu_drift": float(g["freetta_mu_drift"].iloc[-1]),
            "final_prior_entropy": float(g["freetta_prior_entropy"].iloc[-1]),
            "final_sigma_trace": float(g["freetta_sigma_trace"].iloc[-1]),
            "mean_mu_update_norm": float(g["freetta_mu_update_norm"].mean()),
        })
    return pd.DataFrame(rows)


def compute_tda_internal(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        g = g.sort_values("stream_step").reset_index(drop=True)
        rows.append({
            "dataset": ds,
            "final_pos_cache": int(g["tda_pos_cache_size"].iloc[-1]),
            "final_neg_cache": int(g["tda_neg_cache_size"].iloc[-1]),
            "mean_gate_rate": float(g["tda_gate_open"].mean()),
            "max_pos_cache": int(g["tda_pos_cache_size"].max()),
        })
    return pd.DataFrame(rows)


# ── Section 7: Disagreement Analysis ─────────────────────────────────────────

def compute_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        dis = g[g["tda_pred"] != g["freetta_pred"]]
        rows.append({
            "dataset": ds,
            "disagreement_rate": float(len(dis) / max(len(g), 1)),
            "n_disagree": len(dis),
            "tda_acc_on_disagree": float(dis["tda_correct"].mean()) if len(dis) else 0.0,
            "freetta_acc_on_disagree": float(dis["freetta_correct"].mean()) if len(dis) else 0.0,
            "clip_acc_on_disagree": float(dis["clip_correct"].mean()) if len(dis) else 0.0,
            "mean_clip_entropy_on_disagree": float(dis["clip_entropy"].mean()) if len(dis) else 0.0,
            "tda_wins": int(
                ((dis["tda_correct"] == 1) & (dis["freetta_correct"] == 0)).sum()
            ) if len(dis) else 0,
            "freetta_wins": int(
                ((dis["freetta_correct"] == 1) & (dis["tda_correct"] == 0)).sum()
            ) if len(dis) else 0,
        })
    return pd.DataFrame(rows)


# ── Section 8: Failure Buckets ────────────────────────────────────────────────

BUCKET_CONDITIONS = {
    "only_freetta_correct": lambda g: (
        (g["clip_correct"] == 0) & (g["tda_correct"] == 0) & (g["freetta_correct"] == 1)
    ),
    "only_tda_correct": lambda g: (
        (g["clip_correct"] == 0) & (g["freetta_correct"] == 0) & (g["tda_correct"] == 1)
    ),
    "clip_correct_both_fail": lambda g: (
        (g["clip_correct"] == 1) & (g["tda_correct"] == 0) & (g["freetta_correct"] == 0)
    ),
    "all_fail": lambda g: (
        (g["clip_correct"] == 0) & (g["tda_correct"] == 0) & (g["freetta_correct"] == 0)
    ),
    "all_correct": lambda g: (
        (g["clip_correct"] == 1) & (g["tda_correct"] == 1) & (g["freetta_correct"] == 1)
    ),
}


def compute_failure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        for bucket, cond in BUCKET_CONDITIONS.items():
            mask = cond(g)
            rows.append({
                "dataset": ds,
                "bucket": bucket,
                "count": int(mask.sum()),
                "rate": float(mask.mean()),
                "mean_clip_entropy": float(g.loc[mask, "clip_entropy"].mean()) if mask.any() else float("nan"),
            })
    return pd.DataFrame(rows)


def export_failure_metadata(df: pd.DataFrame, output_dir: Path) -> None:
    for ds, g in df.groupby("dataset", observed=True):
        ds_dir = output_dir / ds / "failure_cases"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for bucket, cond in BUCKET_CONDITIONS.items():
            mask = cond(g)
            bucket_df = g[mask].sort_values("clip_entropy", ascending=False).head(10)
            if len(bucket_df):
                bucket_df.to_csv(ds_dir / f"{bucket}.csv", index=False)


# ── Difficulty-conditioned comparison ────────────────────────────────────────

def compute_difficulty_split(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", observed=True):
        ranks = g["clip_entropy"].rank(method="first")
        bins = pd.qcut(ranks, q=n_bins, labels=["easy", "medium", "hard"])
        for bin_name, bg in g.groupby(bins, observed=True):
            row = {"dataset": ds, "difficulty": bin_name, "n": len(bg)}
            for m in METHODS:
                row[f"{m}_acc"] = float(bg[f"{m}_correct"].mean())
            rows.append(row)
    return pd.DataFrame(rows)
