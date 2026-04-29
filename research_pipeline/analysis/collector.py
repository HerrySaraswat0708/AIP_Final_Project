from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_pipeline.methods.base import MethodBase, CLIPBaseline
from research_pipeline.methods.tda_variants import TDAAdapter, EntropyGatedTDA
from research_pipeline.methods.freetta_variants import FreeTTAAdapter, ConfGatedFreeTTA
from research_pipeline.methods.hybrid import HybridTDAFreeTTA

# Short names for all six methods
METHOD_NAMES = ("clip", "tda", "freetta", "conf_ftta", "ent_tda", "hybrid")


def _build_methods(data: dict) -> Dict[str, MethodBase]:
    txt = data["text_features"]
    ds = data["dataset"]
    return {
        "clip":      CLIPBaseline(txt),
        "tda":       TDAAdapter(txt, dataset=ds),
        "freetta":   FreeTTAAdapter(txt, dataset=ds),
        "conf_ftta": ConfGatedFreeTTA(txt, dataset=ds),
        "ent_tda":   EntropyGatedTDA(txt, dataset=ds),
        "hybrid":    HybridTDAFreeTTA(txt, dataset=ds),
    }


def collect(data: dict, verbose: bool = True) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Run all six methods over the dataset stream.

    Returns
    -------
    df         : per-sample DataFrame with predictions + diagnostics
    logit_arrs : dict  method_name -> (N, C) float32 numpy array of logit/probs
    """
    methods = _build_methods(data)
    img = data["image_features"]       # (N, D)
    lbl = data["labels"]               # (N,)
    order = data["order"]              # (N,) stream order
    raw_clip = data["clip_logits"]     # (N, C)
    ds = data["dataset"]
    N = data["num_samples"]

    rows: List[dict] = []
    logit_buffers: dict[str, list[np.ndarray]] = {m: [] for m in METHOD_NAMES}
    # Special: per-step FreeTTA class-mean drift  (N, C)
    mu_drift_rows: list[np.ndarray] = []

    clip_method = methods["clip"]

    with torch.no_grad():
        for stream_step, idx_t in enumerate(order):
            idx = int(idx_t.item())
            x = img[idx]
            y = int(lbl[idx].item())
            rc = raw_clip[idx]  # (C,) raw cosine sims

            # CLIP prediction used as reference for "changed" flag
            clip_trace = clip_method.predict_update(x, rc)
            clip_pred = clip_trace.pred

            row: dict = {
                "dataset": ds,
                "sample_idx": idx,
                "stream_step": stream_step,
                "label": y,
            }

            for mname, method in methods.items():
                trace = (
                    clip_trace
                    if mname == "clip"
                    else method.predict_update(x, rc)
                )
                correct = int(trace.pred == y)
                changed = int(trace.pred != clip_pred) if mname != "clip" else 0
                beneficial = int(
                    changed and (clip_pred != y) and (trace.pred == y)
                )
                harmful = int(
                    changed and (clip_pred == y) and (trace.pred != y)
                )

                row[f"{mname}_pred"] = trace.pred
                row[f"{mname}_correct"] = correct
                row[f"{mname}_confidence"] = trace.confidence
                row[f"{mname}_entropy"] = trace.entropy
                row[f"{mname}_changed"] = changed
                row[f"{mname}_beneficial"] = beneficial
                row[f"{mname}_harmful"] = harmful

                # Method-specific extras
                if mname == "tda":
                    row["tda_pos_cache_size"] = trace.extra.get("pos_cache_size", 0)
                    row["tda_neg_cache_size"] = trace.extra.get("neg_cache_size", 0)
                    row["tda_gate_open"] = trace.extra.get("gate_open", 0)
                elif mname == "freetta":
                    row["freetta_em_weight"] = trace.extra.get("em_weight", float("nan"))
                    row["freetta_mu_update_norm"] = trace.extra.get("mu_update_norm", 0.0)
                    row["freetta_mu_drift"] = trace.extra.get("mu_drift", 0.0)
                    row["freetta_prior_entropy"] = trace.extra.get("prior_entropy", 0.0)
                    row["freetta_sigma_trace"] = trace.extra.get("sigma_trace", 0.0)
                    mu_drift_rows.append(trace.extra.get("mu_drift_per_class", np.zeros(data["num_classes"])))
                elif mname == "conf_ftta":
                    row["conf_ftta_skip_rate"] = trace.extra.get("skip_rate", 0.0)
                    row["conf_ftta_mu_drift"] = trace.extra.get("mu_drift", 0.0)
                elif mname == "ent_tda":
                    row["ent_tda_pos_cache_size"] = trace.extra.get("pos_cache_size", 0)
                    row["ent_tda_adaptive_thresh"] = trace.extra.get("adaptive_low_thresh", 0.0)
                elif mname == "hybrid":
                    row["hybrid_tda_adj_norm"] = trace.extra.get("z_tda_adj_norm", 0.0)
                    row["hybrid_ftta_adj_norm"] = trace.extra.get("z_freetta_adj_norm", 0.0)
                    row["hybrid_mu_drift"] = trace.extra.get("ftta_mu_drift", 0.0)

                # Store logit/prob array
                logit_buffers[mname].append(
                    trace.probs.detach().cpu().numpy().astype(np.float32)
                )

            rows.append(row)

            if verbose and (stream_step + 1) % 500 == 0:
                print(
                    f"  [{ds}] step {stream_step + 1}/{N}  "
                    f"clip={row['clip_correct']} tda={row['tda_correct']} "
                    f"freetta={row['freetta_correct']}",
                    flush=True,
                )

    df = pd.DataFrame(rows)

    logit_arrs = {
        m: np.stack(bufs, axis=0)
        for m, bufs in logit_buffers.items()
    }
    # mu drift per class over the stream
    logit_arrs["mu_drift_by_class"] = (
        np.stack(mu_drift_rows, axis=0).astype(np.float32)
        if mu_drift_rows
        else np.zeros((N, data["num_classes"]), dtype=np.float32)
    )

    return df, logit_arrs
