#!/usr/bin/env python3
"""
Comprehensive experimental validation pipeline for CLIP TTA metrics.

Validates 10 metrics via controlled experiments using frozen CLIP features.
Generates per-sample CSVs, aggregate CSVs, 11 PNG plots, and a final report.

Usage:
    python experiments/comprehensive_validation.py \
        --features-dir data/processed \
        --output-dir   outputs/comprehensive_validation \
        --datasets dtd caltech eurosat pets imagenet \
        --seeds 42 123 456 \
        --device cpu \
        --max-samples 2000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_store import load_dataset_features
from models.TDA import TDA
from models.FreeTTA import FreeTTA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  Paper-aligned default hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
TDA_DEFAULTS: dict[str, dict] = {
    "caltech":  dict(alpha=5.0, beta=5.0,  neg_alpha=0.117, neg_beta=1.0,
                     low_entropy_thresh=0.2,  high_entropy_thresh=0.5,
                     pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "dtd":      dict(alpha=4.0, beta=4.5,  neg_alpha=0.05,  neg_beta=1.0,
                     low_entropy_thresh=0.05, high_entropy_thresh=0.4,
                     pos_shot_capacity=5, neg_shot_capacity=2, clip_scale=100.0),
    "eurosat":  dict(alpha=4.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0,
                     low_entropy_thresh=0.2,  high_entropy_thresh=0.5,
                     pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "pets":     dict(alpha=2.0, beta=7.0,  neg_alpha=0.117, neg_beta=1.0,
                     low_entropy_thresh=0.2,  high_entropy_thresh=0.5,
                     pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "imagenet": dict(alpha=1.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0,
                     low_entropy_thresh=0.2,  high_entropy_thresh=0.5,
                     pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
}
FREETTA_DEFAULTS: dict[str, dict] = {
    "caltech":  dict(alpha=0.02,  beta=3.0),
    "dtd":      dict(alpha=0.1,   beta=3.0),
    "eurosat":  dict(alpha=0.8,   beta=3.0),
    "pets":     dict(alpha=0.25,  beta=4.0),
    "imagenet": dict(alpha=0.05,  beta=4.0),
}

# ══════════════════════════════════════════════════════════════════════════════
#  Feature loading
# ══════════════════════════════════════════════════════════════════════════════

def load_feat(feat_dir: Path, dataset: str, device: str = "cpu"):
    d = load_dataset_features(feat_dir, dataset)
    img = F.normalize(torch.tensor(d["image_features"], dtype=torch.float32), dim=-1).to(device)
    txt = F.normalize(torch.tensor(d["text_features"],  dtype=torch.float32), dim=-1).to(device)
    lbl = torch.tensor(d["labels"], dtype=torch.long).to(device)
    return img, txt, lbl


def subsample(img, txt, lbl, max_n: int, seed: int = 0):
    """Random subsample to at most max_n samples, keeping all classes if possible."""
    N = img.shape[0]
    if N <= max_n:
        return img, txt, lbl
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, max_n, replace=False)
    idx_t = torch.from_numpy(idx).to(img.device)
    return img[idx_t], txt, lbl[idx_t]


def spc_subsample(img, lbl, spc: int, seed: int = 0):
    """Take at most `spc` samples per class, return indices."""
    C   = int(lbl.max().item()) + 1
    lbl_np = lbl.cpu().numpy()
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(C):
        mask = np.where(lbl_np == c)[0]
        if len(mask) == 0:
            continue
        chosen = rng.choice(mask, min(spc, len(mask)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    idx_t = torch.from_numpy(idx).to(img.device)
    return img[idx_t], lbl[idx_t]

# ══════════════════════════════════════════════════════════════════════════════
#  Stream orderings
# ══════════════════════════════════════════════════════════════════════════════

def order_natural(N: int, **_) -> np.ndarray:
    return np.arange(N)

def order_random(N: int, seed: int = 42, **_) -> np.ndarray:
    return np.random.RandomState(seed).permutation(N)

def order_class_blocked(lbl_np: np.ndarray, **_) -> np.ndarray:
    idx = []
    for c in np.unique(lbl_np):
        idx.extend(np.where(lbl_np == c)[0].tolist())
    return np.array(idx)

def order_cyclic(lbl_np: np.ndarray, **_) -> np.ndarray:
    classes = np.unique(lbl_np)
    by_class = {c: list(np.where(lbl_np == c)[0]) for c in classes}
    done, idx = set(), []
    i = 0
    while len(done) < len(classes):
        c = classes[i % len(classes)]
        if c not in done:
            lst = by_class[c]
            if lst:
                idx.append(lst.pop(0))
            else:
                done.add(c)
        i += 1
    return np.array(idx)

def order_adversarial(clip_ent: np.ndarray, **_) -> np.ndarray:
    """Hardest (highest entropy) samples first."""
    return np.argsort(-clip_ent)

STREAM_ORDERS = {
    "natural":       order_natural,
    "random":        order_random,
    "class_blocked": order_class_blocked,
    "cyclic":        order_cyclic,
    "adversarial":   order_adversarial,
}

# ══════════════════════════════════════════════════════════════════════════════
#  Per-sample detailed runner  (uses predict() to capture logits + cache state)
# ══════════════════════════════════════════════════════════════════════════════

def run_detailed(img: torch.Tensor, txt: torch.Tensor, lbl: torch.Tensor,
                 order: np.ndarray, tda_cfg: dict, ftta_cfg: dict) -> dict:
    """
    Process samples in `order`, recording per-sample statistics.

    Returns dict of numpy arrays, each length = len(order):
        true_label, clip_pred, tda_pred, ftta_pred
        clip_logits  (N, C)  scaled CLIP logits
        tda_logits   (N, C)  TDA fused logits
        ftta_logits  (N, C)  FreeTTA fused logits (un-softmaxed, estimated)
        clip_conf, tda_conf, ftta_conf  (N,)
        clip_ent,  tda_ent,  ftta_ent  (N,)  normalised [0,1]
        clip_correct, tda_correct, ftta_correct  (N,) bool
        tda_changed, ftta_changed  (N,) bool
        tda_pos_size, tda_neg_size  (N,) int  cache sizes after each step
        tda_neg_gate_open  (N,) bool
        ftta_weight  (N,)  EM weight = exp(-beta * norm_H)
        ftta_mu_drift  (N,)  mean centroid drift vs initial text features
    """
    N, C  = len(order), txt.shape[0]
    max_H = math.log(max(C, 2))
    dev   = str(img.device)

    tda_m  = TDA(txt, **tda_cfg, device=dev)
    ftta_m = FreeTTA(txt, **ftta_cfg, device=dev)

    clip_scale = tda_cfg.get("clip_scale", 100.0)
    # unscaled logits for FreeTTA.predict() interface (it scales internally)
    unscaled = (img @ txt.t())         # (total_N, C)
    scaled   = clip_scale * unscaled   # (total_N, C)

    # Pre-compute CLIP stats for all samples
    probs_all = torch.softmax(scaled, dim=-1)   # (total_N, C)
    H_all = (-(probs_all * torch.log(probs_all + 1e-8)).sum(-1) / max_H).clamp(0, 1)  # (total_N,)

    # Output arrays
    true_label   = lbl[order].cpu().numpy()
    clip_pred_arr = scaled[order].argmax(-1).cpu().numpy()

    tda_pred_arr  = np.empty(N, np.int64)
    ftta_pred_arr = np.empty(N, np.int64)
    tda_logits_arr  = np.empty((N, C), np.float32)
    ftta_logits_arr = np.empty((N, C), np.float32)
    tda_pos_sizes   = np.empty(N, np.int32)
    tda_neg_sizes   = np.empty(N, np.int32)
    tda_neg_gate    = np.empty(N, bool)
    ftta_weights    = np.empty(N, np.float32)
    ftta_mu_drifts  = np.empty(N, np.float32)

    for rank, idx in enumerate(order):
        x  = img[idx]
        cl = unscaled[idx]   # unscaled, for FreeTTA
        nh = float(H_all[idx].item())

        # TDA
        pred_t, _, fl_t = tda_m.predict(x)
        tda_pred_arr[rank]    = int(pred_t.item())
        tda_logits_arr[rank]  = fl_t.squeeze(0).cpu().numpy()
        tda_pos_sizes[rank]   = tda_m.pos_size
        tda_neg_sizes[rank]   = tda_m.neg_size
        tda_neg_gate[rank]    = (tda_m.low_entropy < nh < tda_m.high_entropy)

        # FreeTTA
        pred_f, fprobs = ftta_m.predict(x, cl)
        ftta_pred_arr[rank]    = int(pred_f.item())
        # store probs (monotone transform of logits – fine for confidence/entropy)
        ftta_logits_arr[rank]  = fprobs.squeeze(0).cpu().numpy()

        ftta_weights[rank]   = math.exp(-ftta_m.beta * nh)
        ftta_mu_drifts[rank] = float(
            (ftta_m.mu - ftta_m.mu0).norm(dim=-1).mean().item()
        )

    # Derived stats
    clip_conf = probs_all[order].max(-1).values.cpu().numpy()
    clip_ent  = H_all[order].cpu().numpy()

    tda_probs  = torch.softmax(torch.tensor(tda_logits_arr), dim=-1).numpy()
    tda_conf   = tda_probs.max(-1)
    tda_ent    = (-(tda_probs * np.log(tda_probs + 1e-8)).sum(-1) / max_H)

    ftta_probs = ftta_logits_arr  # already softmax
    ftta_conf  = ftta_probs.max(-1)
    ftta_ent   = (-(ftta_probs * np.log(ftta_probs + 1e-8)).sum(-1) / max_H)

    clip_correct = (clip_pred_arr == true_label)
    tda_correct  = (tda_pred_arr  == true_label)
    ftta_correct = (ftta_pred_arr == true_label)

    return dict(
        true_label=true_label, clip_pred=clip_pred_arr,
        tda_pred=tda_pred_arr, ftta_pred=ftta_pred_arr,
        clip_logits=scaled[order].cpu().numpy(),
        tda_logits=tda_logits_arr, ftta_logits=ftta_logits_arr,
        clip_conf=clip_conf, tda_conf=tda_conf, ftta_conf=ftta_conf,
        clip_ent=clip_ent,   tda_ent=tda_ent,   ftta_ent=ftta_ent,
        clip_correct=clip_correct, tda_correct=tda_correct, ftta_correct=ftta_correct,
        tda_changed=(tda_pred_arr != clip_pred_arr),
        ftta_changed=(ftta_pred_arr != clip_pred_arr),
        tda_pos_size=tda_pos_sizes, tda_neg_size=tda_neg_sizes,
        tda_neg_gate=tda_neg_gate,
        ftta_weight=ftta_weights, ftta_mu_drift=ftta_mu_drifts,
        order=order,
    )


def quick_run(img, txt, lbl, order, tda_cfg, ftta_cfg):
    """Fast batch run — returns (clip_acc, tda_acc, ftta_acc)."""
    img_o, lbl_o = img[order], lbl[order]
    cs = tda_cfg.get("clip_scale", 100.0)
    ca = float(((cs * (img_o @ txt.t())).argmax(-1) == lbl_o).float().mean())
    tm = TDA(txt, **tda_cfg, device=str(img.device))
    tp, _ = tm.run(img_o)
    ta = float((tp == lbl_o).float().mean())
    fm = FreeTTA(txt, **ftta_cfg, device=str(img.device))
    fp, _ = fm.run(img_o)
    fa = float((fp == lbl_o).float().mean())
    return ca, ta, fa


def flip_stats(clip_c: np.ndarray, method_c: np.ndarray):
    ben  = (~clip_c) & method_c
    harm = clip_c & (~method_c)
    total_flip = ben | harm
    bfp = float(ben.sum() / (total_flip.sum() + 1e-9))
    cr  = float(total_flip.sum() / len(clip_c))
    return int(ben.sum()), int(harm.sum()), bfp, cr

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Accuracy vs samples-per-class
# ══════════════════════════════════════════════════════════════════════════════

def exp1_accuracy_scaling(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg) -> pd.DataFrame:
    spc_vals = [5, 10, 25, 50, 100, 200, 500]
    rows = []
    for spc in spc_vals:
        sub_accs = {"clip": [], "tda": [], "ftta": []}
        for seed in seeds:
            si, sl = spc_subsample(img, lbl, spc, seed)
            if si.shape[0] < 5:
                continue
            order = order_random(si.shape[0], seed=seed)
            ca, ta, fa = quick_run(si, txt, sl, order, tda_cfg, ftta_cfg)
            sub_accs["clip"].append(ca); sub_accs["tda"].append(ta); sub_accs["ftta"].append(fa)
        if not sub_accs["clip"]:
            continue
        rows.append(dict(dataset=dataset, spc=spc,
                         clip_mean=np.mean(sub_accs["clip"]), clip_std=np.std(sub_accs["clip"]),
                         tda_mean =np.mean(sub_accs["tda"]),  tda_std =np.std(sub_accs["tda"]),
                         ftta_mean=np.mean(sub_accs["ftta"]), ftta_std=np.std(sub_accs["ftta"]),
                         n_samples=img.shape[0]))
    return pd.DataFrame(rows)


def plot_exp1(dfs: dict[str, pd.DataFrame], out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, (ds, df) in zip(axes[0], dfs.items()):
        for col, lbl, mk in [("clip","CLIP","o"), ("tda","TDA","s"), ("ftta","FreeTTA","^")]:
            ax.errorbar(df["spc"], df[f"{col}_mean"], yerr=df[f"{col}_std"],
                        label=lbl, marker=mk, capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("Samples / class")
        ax.set_ylabel("Accuracy"); ax.set_title(ds.upper())
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_samples_per_class.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Change Rate vs Accuracy / BFP
# ══════════════════════════════════════════════════════════════════════════════

def exp2_change_rate(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg) -> pd.DataFrame:
    N = img.shape[0]
    tda_alphas  = [0.1, 0.5, 1.0, 2.0, 5.0]
    ftta_alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    rows = []
    for method, alphas, base_cfg in [("TDA", tda_alphas, tda_cfg), ("FreeTTA", ftta_alphas, ftta_cfg)]:
        for alpha in alphas:
            cfg = dict(base_cfg); cfg["alpha"] = alpha
            accs, bfps, crs = [], [], []
            for seed in seeds:
                order = order_random(N, seed=seed)
                img_o, lbl_o = img[order], lbl[order]
                lbl_np = lbl_o.cpu().numpy()
                cs = tda_cfg.get("clip_scale", 100.0)
                cp = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
                if method == "TDA":
                    m = TDA(txt, **cfg, device=str(img.device))
                    p, _ = m.run(img_o)
                else:
                    m = FreeTTA(txt, **cfg, device=str(img.device))
                    p, _ = m.run(img_o)
                p = p.cpu().numpy()
                _, _, bfp, cr = flip_stats(cp == lbl_np, p == lbl_np)
                accs.append((p == lbl_np).mean())
                bfps.append(bfp); crs.append(cr)
            rows.append(dict(method=method, alpha=alpha, dataset=dataset,
                             acc=np.mean(accs), bfp=np.mean(bfps), change_rate=np.mean(crs)))
    return pd.DataFrame(rows)


def plot_exp2(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    markers = {"TDA": "s", "FreeTTA": "^"}
    colors  = plt.cm.tab10(np.linspace(0, 0.8, max(len(dfs), 1)))
    for (ax_i, ykey, ylabel) in [(0, "acc", "Accuracy"), (1, "bfp", "BFP")]:
        ax = axes[ax_i]
        for ci, (ds, df) in enumerate(dfs.items()):
            for method in ["TDA", "FreeTTA"]:
                sub = df[df["method"] == method]
                ax.scatter(sub["change_rate"], sub[ykey],
                           label=f"{ds}/{method}", marker=markers[method],
                           color=colors[ci], alpha=0.8,
                           edgecolors="k" if method == "TDA" else "none", linewidths=0.6)
        ax.set_xlabel("Change Rate"); ax.set_ylabel(ylabel)
        ax.set_title(f"Change Rate vs {ylabel}"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "change_rate_vs_accuracy.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — BFP vs entropy thresholds / FreeTTA beta
# ══════════════════════════════════════════════════════════════════════════════

def exp3_bfp_thresholds(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg) -> pd.DataFrame:
    N = img.shape[0]
    theta_lows  = [0.05, 0.1, 0.2, 0.3, 0.5]
    theta_highs = [0.2, 0.4, 0.6, 0.8]
    ftta_betas  = [0.3, 0.5, 0.7, 0.9, 1.5, 3.0, 5.0, 8.0]
    rows = []

    for tl in theta_lows:
        for th in theta_highs:
            if th <= tl:
                continue
            cfg = dict(tda_cfg, low_entropy_thresh=tl, high_entropy_thresh=th)
            bens, harms, bfps = [], [], []
            for seed in seeds:
                order = order_random(N, seed=seed)
                img_o, lbl_o = img[order], lbl[order]
                lbl_np = lbl_o.cpu().numpy()
                cs = cfg.get("clip_scale", 100.0)
                cp = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
                m = TDA(txt, **cfg, device=str(img.device))
                p, _ = m.run(img_o); p = p.cpu().numpy()
                b, h, bfp, _ = flip_stats(cp == lbl_np, p == lbl_np)
                bens.append(b); harms.append(h); bfps.append(bfp)
            rows.append(dict(method="TDA", theta_low=tl, theta_high=th,
                             param=f"lt={tl}/ht={th}", dataset=dataset,
                             beneficial=np.mean(bens), harmful=np.mean(harms),
                             bfp=np.mean(bfps), acc=0.0))

    for beta in ftta_betas:
        cfg = dict(ftta_cfg, beta=beta)
        bens, harms, bfps, accs = [], [], [], []
        for seed in seeds:
            order = order_random(N, seed=seed)
            img_o, lbl_o = img[order], lbl[order]
            lbl_np = lbl_o.cpu().numpy()
            cs = tda_cfg.get("clip_scale", 100.0)
            cp = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
            m = FreeTTA(txt, **cfg, device=str(img.device))
            p, _ = m.run(img_o); p = p.cpu().numpy()
            b, h, bfp, _ = flip_stats(cp == lbl_np, p == lbl_np)
            bens.append(b); harms.append(h); bfps.append(bfp)
            accs.append((p == lbl_np).mean())
        rows.append(dict(method="FreeTTA", beta=beta, theta_low=None, theta_high=None,
                         param=f"beta={beta}", dataset=dataset,
                         beneficial=np.mean(bens), harmful=np.mean(harms),
                         bfp=np.mean(bfps), acc=np.mean(accs)))
    return pd.DataFrame(rows)


def plot_exp3(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ds, df in dfs.items():
        tda_sub  = df[(df["method"] == "TDA") & (df["theta_low"] == 0.2)].copy()
        ftta_sub = df[df["method"] == "FreeTTA"].copy()
        if len(tda_sub):
            tda_sub = tda_sub.sort_values("theta_high")
            axes[0].plot(tda_sub["theta_high"], tda_sub["bfp"], marker="s", label=f"{ds}/TDA(lt=0.2)")
        if len(ftta_sub):
            ftta_sub = ftta_sub.sort_values("beta")
            axes[1].plot(ftta_sub["beta"], ftta_sub["bfp"], marker="^", label=f"{ds}/FreeTTA")
    axes[0].set_xlabel("High entropy threshold (TDA)"); axes[0].set_ylabel("BFP")
    axes[0].set_title("TDA: BFP vs entropy upper threshold"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    axes[1].set_xlabel("Beta (FreeTTA confidence gate)"); axes[1].set_ylabel("BFP")
    axes[1].set_title("FreeTTA: BFP vs beta"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bfp_vs_thresholds.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 4 — Entropy-confidence calibration (ECE + reliability diagram)
# ══════════════════════════════════════════════════════════════════════════════

def ece_score(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    val  = 0.0
    for i in range(n_bins):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if not m.any():
            continue
        val += m.sum() * abs(corrects[m].mean() - confs[m].mean())
    return val / max(len(confs), 1)


def exp4_calibration(data: dict, dataset: str) -> pd.DataFrame:
    rows = []
    for method, conf, ent, correct in [
        ("CLIP",    data["clip_conf"], data["clip_ent"], data["clip_correct"]),
        ("TDA",     data["tda_conf"],  data["tda_ent"],  data["tda_correct"]),
        ("FreeTTA", data["ftta_conf"], data["ftta_ent"], data["ftta_correct"]),
    ]:
        ec = ece_score(conf, correct.astype(float))
        rows.append(dict(dataset=dataset, method=method, ece=ec,
                         conf_correct=float(conf[correct].mean()) if correct.any() else 0.0,
                         conf_wrong  =float(conf[~correct].mean()) if (~correct).any() else 0.0,
                         ent_correct =float(ent[correct].mean())  if correct.any() else 0.0,
                         ent_wrong   =float(ent[~correct].mean()) if (~correct).any() else 0.0,
                         acc=float(correct.mean())))
    return pd.DataFrame(rows)


def plot_exp4(data: dict, dataset: str, out_dir: Path):
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    bc     = (bins[:-1] + bins[1:]) / 2

    # Reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (method, conf, correct) in zip(axes, [
        ("CLIP",    data["clip_conf"], data["clip_correct"]),
        ("TDA",     data["tda_conf"],  data["tda_correct"]),
        ("FreeTTA", data["ftta_conf"], data["ftta_correct"]),
    ]):
        acc_bins = np.array([correct[(conf >= bins[i]) & (conf < bins[i+1])].mean()
                             if ((conf >= bins[i]) & (conf < bins[i+1])).any() else 0.0
                             for i in range(n_bins)])
        ax.bar(bc, acc_bins, width=0.08, alpha=0.75, label="Accuracy")
        ax.plot([0, 1], [0, 1], "r--", label="Perfect")
        ax.set_title(f"{dataset.upper()} / {method}  ECE={ece_score(conf, correct.astype(float)):.3f}")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"reliability_diagrams_{dataset}.png", dpi=130)
    plt.close()

    # Confidence/entropy split by correctness
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for col, (method, conf, ent, correct) in enumerate([
        ("TDA",     data["tda_conf"],  data["tda_ent"],  data["tda_correct"]),
        ("FreeTTA", data["ftta_conf"], data["ftta_ent"], data["ftta_correct"]),
    ]):
        for row, (arr, xlabel) in enumerate([(conf, "Confidence"), (ent, "Entropy (normalised)")]):
            ax = axes2[row, col]
            ax.hist(arr[correct],  bins=25, alpha=0.6, density=True, label="Correct", color="steelblue")
            ax.hist(arr[~correct], bins=25, alpha=0.6, density=True, label="Wrong",   color="tomato")
            ax.set_title(f"{method} — {xlabel} ({dataset.upper()})")
            ax.set_xlabel(xlabel); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"entropy_confidence_correct_wrong_{dataset}.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 5 — Break-even point by stream order & window size
# ══════════════════════════════════════════════════════════════════════════════

def rolling_acc(correct: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(correct.astype(float), kernel, mode="valid")


def first_breakeven(roll_method: np.ndarray, roll_clip: np.ndarray,
                    consecutive: int = 10) -> Optional[int]:
    ahead = roll_method > roll_clip
    for j in range(len(ahead) - consecutive + 1):
        if ahead[j:j + consecutive].all():
            return j
    return None


def exp5_break_even(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg,
                    windows=(25, 50, 100, 250)) -> pd.DataFrame:
    N      = img.shape[0]
    lbl_np = lbl.cpu().numpy()
    cs     = tda_cfg.get("clip_scale", 100.0)

    clip_ent_np = ((-(torch.softmax(cs * (img @ txt.t()), dim=-1) *
                      torch.log(torch.softmax(cs * (img @ txt.t()), dim=-1) + 1e-8)
                      ).sum(-1)) / math.log(max(txt.shape[0], 2))).cpu().numpy()

    rows = []
    for oname, ofn in STREAM_ORDERS.items():
        if oname == "random":
            base_seeds = seeds
        else:
            base_seeds = [seeds[0]]

        for seed in base_seeds:
            if oname == "random":
                order = ofn(N, seed=seed)
            elif oname == "adversarial":
                order = ofn(clip_ent=clip_ent_np)
            elif oname in ("class_blocked", "cyclic"):
                order = ofn(lbl_np=lbl_np)
            else:
                order = ofn(N)

            img_o, lbl_o = img[order], lbl[order]
            lbl_o_np = lbl_o.cpu().numpy()

            clip_p  = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
            tm = TDA(txt, **tda_cfg, device=str(img.device))
            tda_p, _ = tm.run(img_o); tda_p = tda_p.cpu().numpy()
            fm = FreeTTA(txt, **ftta_cfg, device=str(img.device))
            ftta_p, _ = fm.run(img_o); ftta_p = ftta_p.cpu().numpy()

            clip_c = (clip_p  == lbl_o_np)
            tda_c  = (tda_p   == lbl_o_np)
            ftta_c = (ftta_p  == lbl_o_np)

            for win in windows:
                if N < win + 10:
                    continue
                rc = rolling_acc(clip_c, win)
                rt = rolling_acc(tda_c,  win)
                rf = rolling_acc(ftta_c, win)
                rows.append(dict(
                    dataset=dataset, stream=oname, seed=seed, window=win,
                    tda_breakeven  = first_breakeven(rt, rc),
                    ftta_breakeven = first_breakeven(rf, rc),
                    tda_final_acc  = float(rt[-1]),
                    ftta_final_acc = float(rf[-1]),
                    clip_final_acc = float(rc[-1]),
                ))
    return pd.DataFrame(rows)


def plot_exp5(dfs: dict, out_dir: Path, window: int = 100):
    streams = list(STREAM_ORDERS.keys())
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, (ds, df) in zip(axes[0], dfs.items()):
        sub = df[df["window"] == window].groupby("stream").agg(
            tda_be=("tda_breakeven",  "mean"),
            ftta_be=("ftta_breakeven", "mean"),
        ).reindex(streams)
        x = np.arange(len(streams))
        ax.bar(x - 0.2, sub["tda_be"].fillna(0),  0.35, label="TDA",     alpha=0.85)
        ax.bar(x + 0.2, sub["ftta_be"].fillna(0), 0.35, label="FreeTTA", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in streams], fontsize=7)
        ax.set_ylabel(f"Break-even sample (win={window})")
        ax.set_title(ds.upper()); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "break_even_by_stream_order.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 6 — Disagreement analysis + qualitative examples
# ══════════════════════════════════════════════════════════════════════════════

def exp6_disagreement(data: dict, dataset: str, out_dir: Path) -> pd.DataFrame:
    dis   = data["tda_pred"] != data["ftta_pred"]
    tc, fc, cc = data["tda_correct"], data["ftta_correct"], data["clip_correct"]

    row = dict(
        dataset=dataset,
        n_total=len(dis),
        n_disagree=int(dis.sum()),
        disagree_rate=float(dis.mean()),
        tda_wins   =int((tc & ~fc & dis).sum()),
        ftta_wins  =int((fc & ~tc & dis).sum()),
        both_wrong =int((~tc & ~fc & dis).sum()),
        tda_acc_on_disagree =float(tc[dis].mean()) if dis.any() else 0.0,
        ftta_acc_on_disagree=float(fc[dis].mean()) if dis.any() else 0.0,
        clip_acc_on_disagree=float(cc[dis].mean()) if dis.any() else 0.0,
    )

    # Save qualitative examples
    dis_idx = np.where(dis)[0][:20]  # up to 20 examples
    qual = []
    for i in dis_idx:
        qual.append(dict(
            rank=int(i), orig_idx=int(data["order"][i]),
            true_label=int(data["true_label"][i]),
            clip_pred=int(data["clip_pred"][i]),
            tda_pred =int(data["tda_pred"][i]),
            ftta_pred=int(data["ftta_pred"][i]),
            clip_conf=float(data["clip_conf"][i]),
            tda_conf =float(data["tda_conf"][i]),
            ftta_conf=float(data["ftta_conf"][i]),
            clip_ent =float(data["clip_ent"][i]),
            tda_pos_size =int(data["tda_pos_size"][i]),
            tda_neg_size =int(data["tda_neg_size"][i]),
            ftta_mu_drift=float(data["ftta_mu_drift"][i]),
            tda_correct =bool(data["tda_correct"][i]),
            ftta_correct=bool(data["ftta_correct"][i]),
            clip_correct=bool(data["clip_correct"][i]),
        ))
    pd.DataFrame(qual).to_csv(out_dir / "disagreement_examples.csv", index=False)
    return pd.DataFrame([row])


def plot_exp6(all_rows: list[pd.DataFrame], out_dir: Path):
    df = pd.concat(all_rows, ignore_index=True)
    ds_list = df["dataset"].tolist()
    x  = np.arange(len(ds_list))
    w  = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x - w, df["clip_acc_on_disagree"],  w, label="CLIP",    alpha=0.85)
    axes[0].bar(x,     df["tda_acc_on_disagree"],   w, label="TDA",     alpha=0.85)
    axes[0].bar(x + w, df["ftta_acc_on_disagree"],  w, label="FreeTTA", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ds_list)
    axes[0].set_ylabel("Accuracy on TDA≠FreeTTA samples")
    axes[0].set_title("Method accuracy on disagreement set")
    axes[0].legend(); axes[0].grid(alpha=.3, axis="y")

    axes[1].bar(ds_list, df["disagree_rate"], alpha=0.85)
    axes[1].set_ylabel("Disagreement rate"); axes[1].set_title("Fraction where TDA ≠ FreeTTA")
    axes[1].grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "disagreement_accuracy.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 7 — Failure buckets (parametric)
# ══════════════════════════════════════════════════════════════════════════════

BUCKETS = ["all_correct", "only_tda", "only_ftta",
           "clip_ok_tda_wrong", "clip_ok_ftta_wrong", "all_wrong"]

def bucket_counts(cc, tc, fc) -> dict:
    return dict(
        all_correct       = int((cc & tc & fc).sum()),
        only_tda          = int((~cc & tc & ~fc).sum()),
        only_ftta         = int((~cc & ~tc & fc).sum()),
        clip_ok_tda_wrong = int((cc & ~tc).sum()),
        clip_ok_ftta_wrong= int((cc & ~fc).sum()),
        all_wrong         = int((~cc & ~tc & ~fc).sum()),
    )


def exp7_failure_buckets(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg) -> pd.DataFrame:
    N, C = img.shape[0], txt.shape[0]
    sweep_K = [1, 3, 5, 10, 25, 50, 100]
    rows = []
    for K in sweep_K:
        cp_N = N / (C * K)
        cfg  = dict(tda_cfg, pos_shot_capacity=K, neg_shot_capacity=max(1, K // 2))
        agg  = defaultdict(list)
        for seed in seeds:
            order = order_random(N, seed=seed)
            img_o, lbl_o = img[order], lbl[order]
            lbl_np = lbl_o.cpu().numpy()
            cs = tda_cfg.get("clip_scale", 100.0)
            cp = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
            tm = TDA(txt, **cfg, device=str(img.device))
            tp, _ = tm.run(img_o); tp = tp.cpu().numpy()
            fm = FreeTTA(txt, **ftta_cfg, device=str(img.device))
            fp, _ = fm.run(img_o); fp = fp.cpu().numpy()
            b = bucket_counts(cp == lbl_np, tp == lbl_np, fp == lbl_np)
            for k, v in b.items():
                agg[k].append(v)
        row = dict(dataset=dataset, cache_K=K, cache_pressure=cp_N)
        for k in BUCKETS:
            row[k] = np.mean(agg[k])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_exp7(dfs: dict, out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(BUCKETS)))
    for ax, (ds, df) in zip(axes[0], dfs.items()):
        bot = np.zeros(len(df))
        for bk, col in zip(BUCKETS, colors):
            vals = df[bk].values
            ax.bar(df["cache_K"].astype(str), vals, bottom=bot,
                   label=bk.replace("_", "\n"), color=col, width=0.6)
            bot += vals
        ax.set_xlabel("TDA cache K"); ax.set_ylabel("Samples")
        ax.set_title(ds.upper()); ax.legend(fontsize=5, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_bucket_stacked_bars.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 8 — GAS: real datasets + synthetic cluster experiments
# ══════════════════════════════════════════════════════════════════════════════

def oracle_centroid_acc_fast(img_np: np.ndarray, lbl_np: np.ndarray) -> float:
    classes = np.unique(lbl_np)
    C       = len(classes)
    # Pre-compute class sums and counts
    cls_sum   = np.zeros((C, img_np.shape[1]))
    cls_count = np.zeros(C)
    c2i = {c: i for i, c in enumerate(classes)}
    for c in classes:
        mask = lbl_np == c
        cls_sum[c2i[c]]   = img_np[mask].sum(axis=0)
        cls_count[c2i[c]] = mask.sum()

    correct = 0
    for i in range(len(lbl_np)):
        ci = c2i[lbl_np[i]]
        # LOO centroid for own class
        sums = cls_sum.copy()
        cnts = cls_count.copy()
        sums[ci] -= img_np[i]
        cnts[ci]  = max(cnts[ci] - 1, 1e-9)
        centroids = sums / cnts[:, None]
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids /= np.where(norms > 1e-8, norms, 1.0)
        pred_i = classes[np.argmax(img_np[i] @ centroids.T)]
        correct += int(pred_i == lbl_np[i])
    return correct / len(lbl_np)


def oracle_1nn_acc(img_np: np.ndarray, lbl_np: np.ndarray) -> float:
    sims = img_np @ img_np.T
    np.fill_diagonal(sims, -1e9)
    preds = lbl_np[sims.argmax(axis=1)]
    return float((preds == lbl_np).mean())


def synthetic_gas(n_per_class: int = 100, dim: int = 64, seed: int = 42) -> dict:
    """GAS on four synthetic cluster geometries."""
    rng = np.random.RandomState(seed)
    C   = 10
    results = {}

    # 1. Spherical
    X, y = [], []
    for c in range(C):
        mean = rng.randn(dim)
        mean /= np.linalg.norm(mean)
        pts  = rng.randn(n_per_class, dim) * 0.3 + mean
        X.append(pts); y.extend([c] * n_per_class)
    X = np.vstack(X); y = np.array(y)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ca = oracle_centroid_acc_fast(X, y); na = oracle_1nn_acc(X, y)
    results["spherical"] = dict(centroid_acc=ca, nn_acc=na, gas=ca - na)

    # 2. Elongated (high variance in one direction)
    X, y = [], []
    for c in range(C):
        mean = rng.randn(dim); mean /= np.linalg.norm(mean)
        stretch = rng.randn(dim); stretch /= np.linalg.norm(stretch)
        noise = rng.randn(n_per_class, dim) * 0.15
        noise += np.outer(rng.randn(n_per_class) * 0.8, stretch)
        pts = noise + mean
        X.append(pts); y.extend([c] * n_per_class)
    X = np.vstack(X); y = np.array(y)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ca = oracle_centroid_acc_fast(X, y); na = oracle_1nn_acc(X, y)
    results["elongated"] = dict(centroid_acc=ca, nn_acc=na, gas=ca - na)

    # 3. Multi-modal (3 sub-clusters per class)
    X, y = [], []
    for c in range(C):
        sub_means = [rng.randn(dim) for _ in range(3)]
        sub_means = [m / np.linalg.norm(m) * 0.9 for m in sub_means]
        n_sub = n_per_class // 3
        for sm in sub_means:
            pts = rng.randn(n_sub, dim) * 0.15 + sm
            X.append(pts); y.extend([c] * n_sub)
    X = np.vstack(X); y = np.array(y)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ca = oracle_centroid_acc_fast(X, y); na = oracle_1nn_acc(X, y)
    results["multi_modal"] = dict(centroid_acc=ca, nn_acc=na, gas=ca - na)

    # 4. Overlapping fine-grained (classes very close)
    X, y = [], []
    base = rng.randn(dim); base /= np.linalg.norm(base)
    for c in range(C):
        perturbation = rng.randn(dim) * 0.05
        mean = base + perturbation; mean /= np.linalg.norm(mean)
        pts  = rng.randn(n_per_class, dim) * 0.25 + mean
        X.append(pts); y.extend([c] * n_per_class)
    X = np.vstack(X); y = np.array(y)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ca = oracle_centroid_acc_fast(X, y); na = oracle_1nn_acc(X, y)
    results["overlapping"] = dict(centroid_acc=ca, nn_acc=na, gas=ca - na)

    return results


def exp8_gas(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg,
             max_gas_n: int = 500) -> pd.DataFrame:
    N, C = img.shape[0], txt.shape[0]
    cs   = tda_cfg.get("clip_scale", 100.0)
    rows = []

    spc_vals = [5, 10, 25, 50, 100, 200, 500]
    for spc in spc_vals:
        gas_list, gap_list = [], []
        for seed in seeds:
            si, sl = spc_subsample(img, lbl, spc, seed)
            if si.shape[0] < C:
                continue
            # Limit for oracle computations
            idx = np.random.RandomState(seed).choice(si.shape[0],
                                                     min(si.shape[0], max_gas_n), replace=False)
            img_s = si[idx].cpu().numpy()
            lbl_s = sl[idx].cpu().numpy()
            ca   = oracle_centroid_acc_fast(img_s, lbl_s)
            na   = oracle_1nn_acc(img_s, lbl_s)
            gas  = ca - na

            order = order_random(si.shape[0], seed=seed)
            _, ta, fa = quick_run(si, txt, sl, order, tda_cfg, ftta_cfg)
            gap_list.append(fa - ta); gas_list.append(gas)

        if gas_list:
            rows.append(dict(dataset=dataset, spc=spc,
                             gas=np.mean(gas_list), ftta_minus_tda=np.mean(gap_list),
                             gas_std=np.std(gas_list)))

    # Also compute overall GAS for the full dataset
    idx_full = np.random.RandomState(0).choice(N, min(N, max_gas_n), replace=False)
    img_f  = img[idx_full].cpu().numpy(); lbl_f = lbl[idx_full].cpu().numpy()
    ca_f   = oracle_centroid_acc_fast(img_f, lbl_f)
    na_f   = oracle_1nn_acc(img_f, lbl_f)
    gas_f  = ca_f - na_f

    order_full = order_random(N, seed=seeds[0])
    _, ta_f, fa_f = quick_run(img, txt, lbl, order_full, tda_cfg, ftta_cfg)

    rows.append(dict(dataset=dataset, spc=-1,
                     gas=gas_f, ftta_minus_tda=fa_f - ta_f,
                     gas_std=0.0,
                     centroid_acc=ca_f, nn_acc=na_f))
    return pd.DataFrame(rows)


def plot_exp8(real_dfs: dict, syn_results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Real dataset GAS vs method gap (full dataset only)
    ax = axes[0]
    for ds, df in real_dfs.items():
        full_row = df[df["spc"] == -1]
        if full_row.empty:
            continue
        gas = float(full_row["gas"].iloc[0])
        gap = float(full_row["ftta_minus_tda"].iloc[0])
        ax.scatter(gas, gap, s=100, label=ds, zorder=5)
        ax.annotate(ds, (gas, gap), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("GAS = Centroid Acc − 1-NN Acc")
    ax.set_ylabel("FreeTTA − TDA Accuracy")
    ax.set_title("GAS vs Method Gap (real datasets)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # Synthetic
    ax2 = axes[1]
    names = list(syn_results.keys())
    gas_vals  = [syn_results[k]["gas"]          for k in names]
    cent_vals = [syn_results[k]["centroid_acc"]  for k in names]
    nn_vals   = [syn_results[k]["nn_acc"]        for k in names]
    x = np.arange(len(names))
    ax2.bar(x - 0.2, cent_vals, 0.35, label="Oracle Centroid", alpha=0.85)
    ax2.bar(x + 0.2, nn_vals,   0.35, label="Oracle 1-NN",     alpha=0.85)
    for i, g in enumerate(gas_vals):
        ax2.text(i, max(cent_vals[i], nn_vals[i]) + 0.01, f"GAS={g:+.3f}", ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([n.replace("_", "\n") for n in names])
    ax2.set_ylabel("Oracle Accuracy"); ax2.set_title("GAS on Synthetic Clusters")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "gas_vs_method_gap.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 9 — Cache Pressure vs TDA failure
# ══════════════════════════════════════════════════════════════════════════════

def exp9_cache_pressure(img, txt, lbl, dataset, seeds, tda_cfg, ftta_cfg) -> pd.DataFrame:
    N, C = img.shape[0], txt.shape[0]
    sweep_K = [1, 3, 5, 10, 25, 50, 100]
    rows = []
    for K in sweep_K:
        cp_val = N / (C * K)
        cfg    = dict(tda_cfg, pos_shot_capacity=K, neg_shot_capacity=max(1, K // 2))
        accs, harm_rates, gaps = [], [], []
        for seed in seeds:
            order  = order_random(N, seed=seed)
            img_o, lbl_o = img[order], lbl[order]
            lbl_np = lbl_o.cpu().numpy()
            cs = tda_cfg.get("clip_scale", 100.0)
            cp = (cs * (img_o @ txt.t())).argmax(-1).cpu().numpy()
            tm = TDA(txt, **cfg, device=str(img.device))
            tp, _ = tm.run(img_o); tp = tp.cpu().numpy()
            fm = FreeTTA(txt, **ftta_cfg, device=str(img.device))
            fp, _ = fm.run(img_o); fp = fp.cpu().numpy()
            _, harm, _, _ = flip_stats(cp == lbl_np, tp == lbl_np)
            accs.append((tp == lbl_np).mean())
            harm_rates.append(harm / N)
            gaps.append((fp == lbl_np).mean() - (tp == lbl_np).mean())
        rows.append(dict(dataset=dataset, cache_K=K, cache_pressure=cp_val,
                         tda_acc=np.mean(accs),
                         harmful_flip_rate=np.mean(harm_rates),
                         ftta_minus_tda=np.mean(gaps)))
    return pd.DataFrame(rows)


def plot_exp9(dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ds, df in dfs.items():
        kw = dict(marker="o", label=ds)
        axes[0].plot(df["cache_pressure"], df["tda_acc"],           **kw)
        axes[1].plot(df["cache_pressure"], df["harmful_flip_rate"], **kw)
        axes[2].plot(df["cache_pressure"], df["ftta_minus_tda"],    **kw)
    for ax, ylabel, title in zip(axes,
        ["TDA Accuracy", "Harmful Flip Rate", "FreeTTA − TDA Accuracy"],
        ["Cache Pressure vs TDA Accuracy",
         "Cache Pressure vs Harmful Flips",
         "Cache Pressure vs Method Gap"]):
        ax.set_xscale("log")
        ax.set_xlabel("Cache Pressure N/(C·K)")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cache_pressure_vs_tda_failure.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 10 — Mean EM Weight vs Update Quality
# ══════════════════════════════════════════════════════════════════════════════

def exp10_em_weights(img, txt, lbl, dataset, seeds, ftta_cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    N, C  = img.shape[0], txt.shape[0]
    max_H = math.log(max(C, 2))
    cs    = ftta_cfg.get("clip_scale", 100.0) if "clip_scale" in ftta_cfg else 100.0
    default_cs = TDA_DEFAULTS.get(dataset, TDA_DEFAULTS["dtd"]).get("clip_scale", 100.0)

    probs = torch.softmax(default_cs * (img @ txt.t()), dim=-1)
    H     = (-(probs * torch.log(probs + 1e-8)).sum(-1) / max_H).clamp(0, 1).cpu().numpy()
    alpha_t = 1.0 - H   # EM weight = 1 − normalised_entropy

    # Bucket by tertile
    q33, q67 = np.percentile(alpha_t, [33, 67])
    low  = alpha_t < q33
    mid  = (alpha_t >= q33) & (alpha_t < q67)
    high = alpha_t >= q67

    clip_p  = (default_cs * (img @ txt.t())).argmax(-1).cpu().numpy()
    clip_c  = (clip_p == lbl.cpu().numpy())
    bucket_rows = []
    for name, mask in [("low", low), ("mid", mid), ("high", high)]:
        bucket_rows.append(dict(
            dataset=dataset, bucket=name,
            count=int(mask.sum()),
            mean_alpha=float(alpha_t[mask].mean()) if mask.any() else 0.0,
            clip_acc  =float(clip_c[mask].mean())  if mask.any() else 0.0,
            frac      =float(mask.mean()),
        ))
    bucket_df = pd.DataFrame(bucket_rows)

    # Compare entropy-based vs fixed beta values
    fixed_betas = [0.25, 0.5, 1.0, 3.0]
    rows = []
    for beta in fixed_betas:
        cfg = dict(ftta_cfg, beta=beta)
        accs = []
        for seed in seeds:
            order = order_random(N, seed=seed)
            img_o, lbl_o = img[order], lbl[order]
            fm = FreeTTA(txt, **cfg, device=str(img.device))
            fp, _ = fm.run(img_o); accs.append((fp == lbl_o).float().mean().item())
        rows.append(dict(dataset=dataset, beta=beta, ftta_acc=np.mean(accs),
                         acc_std=np.std(accs)))

    # Also compute drift by bucket using the detailed run
    order0 = np.arange(N)
    data   = run_detailed(img, txt, lbl, order0, TDA_DEFAULTS.get(dataset, TDA_DEFAULTS["dtd"]),
                          ftta_cfg)
    for name, mask in [("low", low), ("mid", mid), ("high", high)]:
        if not mask.any():
            continue
        # correctness of update: future accuracy conditional on being in this bucket
        # Here we measure whether ftta correct correlates with alpha_t bucket
        rows.append(dict(dataset=dataset, beta=f"bucket_{name}",
                         ftta_acc=float(data["ftta_correct"][mask].mean()),
                         acc_std=0.0))

    acc_df = pd.DataFrame(rows)
    return acc_df, bucket_df


def plot_exp10(acc_dfs: dict, bucket_dfs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    numeric_betas = [0.25, 0.5, 1.0, 3.0]
    for ds, df in acc_dfs.items():
        sub = df[df["beta"].apply(lambda x: isinstance(x, (int, float)) and x in numeric_betas)]
        if sub.empty:
            continue
        axes[0].errorbar(sub["beta"], sub["ftta_acc"], yerr=sub["acc_std"],
                         marker="o", label=ds, capsize=3)
    axes[0].set_xlabel("Beta (EM weight gating)")
    axes[0].set_ylabel("FreeTTA Accuracy")
    axes[0].set_title("Fixed Beta vs FreeTTA Accuracy")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    combined = pd.concat(bucket_dfs.values(), ignore_index=True)
    all_ds   = combined["dataset"].unique()
    x = np.arange(3)
    bnames = ["low", "mid", "high"]
    w = 0.8 / max(len(all_ds), 1)
    for i, ds in enumerate(all_ds):
        sub = combined[combined["dataset"] == ds].set_index("bucket")
        vals = [float(sub.loc[b, "clip_acc"]) if b in sub.index else 0.0 for b in bnames]
        axes[1].bar(x + i * w - 0.4, vals, w, label=ds, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Low α\n(high entropy)", "Mid α", "High α\n(low entropy)"])
    axes[1].set_ylabel("CLIP Accuracy")
    axes[1].set_title("CLIP Accuracy by EM Weight Bucket\n(high α = confident = should drive better M-step)")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "em_weight_vs_update_quality.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Consolidated reliability + entropy-confidence plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined_entropy_confidence(all_data: dict, out_dir: Path):
    """One cross-dataset reliability diagram combining CLIP/TDA/FreeTTA."""
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    for ax, (ds, data) in zip(axes[0], all_data.items()):
        for method, conf, correct, color in [
            ("CLIP",    data["clip_conf"], data["clip_correct"], "gray"),
            ("TDA",     data["tda_conf"],  data["tda_correct"],  "steelblue"),
            ("FreeTTA", data["ftta_conf"], data["ftta_correct"], "tomato"),
        ]:
            n_bins = 10
            bins   = np.linspace(0, 1, n_bins + 1)
            bc     = (bins[:-1] + bins[1:]) / 2
            acc_b  = np.array([correct[(conf >= bins[i]) & (conf < bins[i+1])].mean()
                                if ((conf >= bins[i]) & (conf < bins[i+1])).any() else 0.0
                                for i in range(n_bins)])
            ax.plot(bc, acc_b, marker="o", color=color, label=method, lw=2, ms=4)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.set_title(ds.upper()); ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_diagrams.png", dpi=130)
    plt.close()


def plot_combined_ent_conf(all_data: dict, out_dir: Path):
    """Combined entropy-confidence plot across datasets."""
    n = len(all_data)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8), squeeze=False)
    for col, (ds, data) in enumerate(all_data.items()):
        for row, (arr, lbl_str) in enumerate([
            (data["clip_conf"], "Confidence"),
            (data["clip_ent"],  "Normalised Entropy"),
        ]):
            ax = axes[row, col]
            ax.hist(arr[data["tda_correct"]],   bins=25, alpha=0.5, density=True,
                    label="TDA correct",    color="steelblue")
            ax.hist(arr[~data["tda_correct"]],  bins=25, alpha=0.5, density=True,
                    label="TDA wrong",      color="royalblue", linestyle="--")
            ax.hist(arr[data["ftta_correct"]],  bins=25, alpha=0.4, density=True,
                    label="FreeTTA correct", color="tomato")
            ax.hist(arr[~data["ftta_correct"]], bins=25, alpha=0.4, density=True,
                    label="FreeTTA wrong",   color="salmon", linestyle="--")
            ax.set_title(f"{ds.upper()} — {lbl_str}")
            ax.legend(fontsize=6); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_confidence_correct_wrong.png", dpi=130)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  Final report
# ══════════════════════════════════════════════════════════════════════════════

METRIC_DOCS = {
    1: ("Accuracy vs Samples/Class",
        "How adaptation performance scales with dataset size.",
        "Methods that learn from examples (TDA cache, FreeTTA μ update) should improve with more samples/class.",
        "Vary N_spc ∈ {5,…,500}; run CLIP/TDA/FreeTTA over multiple seeds.",
        "accuracy_vs_spc.csv → accuracy_vs_samples_per_class.png"),
    2: ("Change Rate",
        "Fraction of CLIP predictions overridden by TDA or FreeTTA.",
        "Higher change rate is not monotonically better: at some point harmful flips dominate.",
        "Sweep alpha (adaptation strength) for both methods; measure change_rate, accuracy, BFP.",
        "change_rate_results.csv → change_rate_vs_accuracy.png"),
    3: ("Beneficial Flip Precision (BFP)",
        "Ratio of beneficial flips to total flips.  BFP=1 ⇒ every change was correct.",
        "Stricter gating raises BFP but reduces coverage; optimal threshold trades off both.",
        "Sweep TDA entropy thresholds and FreeTTA beta; measure BFP directly.",
        "bfp_threshold_results.csv → bfp_vs_thresholds.png"),
    4: ("Entropy & Confidence Calibration",
        "Whether confidence scores faithfully predict correctness (ECE, reliability).",
        "Adapted methods should show better confidence-accuracy alignment than zero-shot CLIP.",
        "Compute ECE and reliability diagrams for CLIP, TDA, FreeTTA.",
        "calibration_results.csv → reliability_diagrams.png / entropy_confidence_correct_wrong.png"),
    5: ("Break-Even Point",
        "Sample index at which rolling accuracy first exceeds CLIP continuously.",
        "FreeTTA (global EM) adapts earlier; TDA (local cache) benefits from class-local locality.",
        "5 stream orders × 4 window sizes; detect first 10-consecutive-window lead.",
        "break_even_results.csv → break_even_by_stream_order.png"),
    6: ("Disagreement Analysis",
        "Accuracy on samples where TDA and FreeTTA predict different classes.",
        "FreeTTA should win disagreements on high domain-shift sets; TDA on fine-grained sets.",
        "Find TDA≠FreeTTA samples; compute per-method accuracy; save 20 qualitative examples.",
        "disagreement_results.csv / disagreement_examples.csv → disagreement_accuracy.png"),
    7: ("Failure Buckets",
        "6-way partition: all_correct, only_tda, only_ftta, clip_ok_tda_wrong, clip_ok_ftta_wrong, all_wrong.",
        "High cache pressure (large N/CK) should inflate clip_ok_tda_wrong bucket.",
        "Sweep K ∈ {1,…,100}; track bucket fractions across seeds.",
        "failure_buckets.csv → failure_bucket_stacked_bars.png"),
    8: ("GAS — Geometry Alignment Score",
        "GAS = OracleCentroidAcc − Oracle1NNAcc. Positive ⇒ blob geometry; negative ⇒ clustered.",
        "Positive GAS predicts FreeTTA advantage (global centroid works); negative predicts TDA advantage.",
        "Compute oracles on real datasets (varied spc) + 4 synthetic geometries; correlate with method gap.",
        "gas_results.csv + synthetic_gas.json → gas_vs_method_gap.png"),
    9: ("Cache Pressure",
        "CachePressure = N / (C·K). High pressure ⇒ each class slot turns over rapidly.",
        "High cache pressure drives high harmful flip rate and predicts TDA failure.",
        "Sweep K ∈ {1,…,100}; measure TDA accuracy, harmful flips, and FreeTTA−TDA gap vs pressure.",
        "cache_pressure_results.csv → cache_pressure_vs_tda_failure.png"),
    10: ("Mean EM Weight",
        "α_t = 1 − H(x_t)/log(C). High α ⇒ confident sample drives larger M-step update.",
        "Samples with high α should be correctly classified by CLIP → good M-step signal.",
        "Bucket samples by α tertile; measure CLIP accuracy per bucket; sweep fixed beta vs entropy-based.",
        "em_weight_results.csv + em_weight_buckets.csv → em_weight_vs_update_quality.png"),
}


def write_report(out_dir: Path, datasets: list[str], summary: pd.DataFrame):
    lines = [
        "# Comprehensive Metric Validation Report",
        f"\n**Datasets:** {', '.join(d.upper() for d in datasets)}",
        "\n## Summary — Method Accuracy\n",
        summary.to_markdown(index=False),
        "\n---\n",
    ]
    for i in range(1, 11):
        name, measures, predicts, experiment, files = METRIC_DOCS[i]
        lines += [
            f"## {i}. {name}",
            f"**Measures:** {measures}",
            f"**Predicts:** {predicts}",
            f"**Experiment:** {experiment}",
            f"**Output files:** `{files}`",
            "",
        ]
    (out_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/processed")
    ap.add_argument("--output-dir",   default="outputs/comprehensive_validation")
    ap.add_argument("--datasets", nargs="+",
                    default=["dtd", "caltech", "eurosat", "pets", "imagenet"])
    ap.add_argument("--seeds",    nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--device",   default="cpu")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Subsample datasets to at most this many samples (0 = no limit)")
    args = ap.parse_args()

    feat_dir = ROOT / args.features_dir
    out_root = ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    device   = args.device
    datasets = args.datasets
    seeds    = args.seeds
    max_n    = args.max_samples

    # Per-experiment aggregate collectors
    e1_dfs, e2_dfs, e3_dfs, e4_all_data = {}, {}, {}, {}
    e5_dfs, e7_dfs, e8_dfs, e9_dfs      = {}, {}, {}, {}
    e10_acc_dfs, e10_bkt_dfs            = {}, {}
    e6_rows, summary_rows               = [], []

    # Run synthetic GAS once (independent of dataset)
    print("Computing synthetic GAS experiments...")
    syn_gas = synthetic_gas()
    (out_root / "synthetic_gas.json").write_text(json.dumps(syn_gas, indent=2))

    for ds in datasets:
        print(f"\n{'='*60}\n  {ds.upper()}\n{'='*60}")
        out_ds = out_root / ds
        out_ds.mkdir(parents=True, exist_ok=True)

        img, txt, lbl = load_feat(feat_dir, ds, device)
        if max_n > 0:
            img, txt, lbl = subsample(img, txt, lbl, max_n)

        tda_cfg  = TDA_DEFAULTS.get(ds, TDA_DEFAULTS["dtd"])
        ftta_cfg = FREETTA_DEFAULTS.get(ds, FREETTA_DEFAULTS["dtd"])

        # Baseline — natural order
        order0 = np.arange(len(lbl))
        ca, ta, fa = quick_run(img, txt, lbl, order0, tda_cfg, ftta_cfg)
        winner = "TDA" if ta > fa else ("FreeTTA" if fa > ta else "TIE")
        print(f"  CLIP={ca:.4f}  TDA={ta:.4f}  FreeTTA={fa:.4f}  winner={winner}")
        summary_rows.append(dict(dataset=ds, clip=ca, tda=ta, ftta=fa, winner=winner))

        # Detailed run for Exp 4 / 6 calibration
        print("  [4/6] Detailed run (predict loop)...")
        data = run_detailed(img, txt, lbl, order0, tda_cfg, ftta_cfg)
        pd.DataFrame({k: v for k, v in data.items()
                      if k not in ("clip_logits", "tda_logits", "ftta_logits", "order")
                     }).to_csv(out_ds / "per_sample_detailed.csv", index=False)

        e4_all_data[ds] = data
        cal_df = exp4_calibration(data, ds)
        cal_df.to_csv(out_ds / "calibration_results.csv", index=False)
        plot_exp4(data, ds, out_ds)

        dis_df = exp6_disagreement(data, ds, out_ds)
        dis_df.to_csv(out_ds / "disagreement_results.csv", index=False)
        e6_rows.append(dis_df)

        print("  [1] Accuracy scaling...")
        e1 = exp1_accuracy_scaling(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e1.to_csv(out_ds / "accuracy_vs_spc.csv", index=False)
        e1_dfs[ds] = e1

        print("  [2] Change rate...")
        e2 = exp2_change_rate(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e2.to_csv(out_ds / "change_rate_results.csv", index=False)
        e2_dfs[ds] = e2

        print("  [3] BFP thresholds...")
        e3 = exp3_bfp_thresholds(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e3.to_csv(out_ds / "bfp_threshold_results.csv", index=False)
        e3_dfs[ds] = e3

        print("  [5] Break-even...")
        e5 = exp5_break_even(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e5.to_csv(out_ds / "break_even_results.csv", index=False)
        e5_dfs[ds] = e5

        print("  [7] Failure buckets...")
        e7 = exp7_failure_buckets(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e7.to_csv(out_ds / "failure_buckets.csv", index=False)
        e7_dfs[ds] = e7

        print("  [8] GAS...")
        e8 = exp8_gas(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e8.to_csv(out_ds / "gas_results.csv", index=False)
        e8_dfs[ds] = e8

        print("  [9] Cache pressure...")
        e9 = exp9_cache_pressure(img, txt, lbl, ds, seeds, tda_cfg, ftta_cfg)
        e9.to_csv(out_ds / "cache_pressure_results.csv", index=False)
        e9_dfs[ds] = e9

        print("  [10] EM weights...")
        e10a, e10b = exp10_em_weights(img, txt, lbl, ds, seeds, ftta_cfg)
        e10a.to_csv(out_ds / "em_weight_results.csv",  index=False)
        e10b.to_csv(out_ds / "em_weight_buckets.csv",  index=False)
        e10_acc_dfs[ds] = e10a
        e10_bkt_dfs[ds] = e10b

    # ── Cross-dataset plots ──────────────────────────────────────────────────
    print("\nGenerating cross-dataset plots...")
    plot_exp1(e1_dfs, out_root)
    plot_exp2(e2_dfs, out_root)
    plot_exp3(e3_dfs, out_root)
    plot_combined_reliability  = plot_combined_entropy_confidence  # alias
    plot_combined_reliability(e4_all_data, out_root)
    plot_combined_ent_conf(e4_all_data, out_root)
    plot_exp5(e5_dfs, out_root)
    plot_exp6(e6_rows, out_root)
    plot_exp7(e7_dfs, out_root)
    plot_exp8(e8_dfs, syn_gas, out_root)
    plot_exp9(e9_dfs, out_root)
    plot_exp10(e10_acc_dfs, e10_bkt_dfs, out_root)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "method_summary.csv", index=False)

    print("\n" + "="*60)
    print("Method summary (natural order):")
    print(summary_df.to_string(index=False))

    # ── Report ───────────────────────────────────────────────────────────────
    write_report(out_root, datasets, summary_df)
    print(f"\nAll outputs → {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
