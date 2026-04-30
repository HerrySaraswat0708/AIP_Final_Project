"""
Extended Geometry Alignment Score (GAS) Analysis
================================================
Compares how CLIP, FreeTTA, and TDA reshape feature-space geometry.

GAS = Acc_centroid − Acc_1NN

  GAS > 0  →  centroid classifier beats 1-NN  →  global structure dominates
  GAS < 0  →  1-NN beats centroid             →  local structure dominates

Three spaces compared
  GAS_CLIP      : original CLIP image embedding space (two variants)
                  (a) text centroid  – what CLIP actually uses for prediction
                  (b) oracle centroid – mean of test images per GT class
  GAS_FreeTTA   : FreeTTA adapted centroids µ_c^(T) vs CLIP 1-NN baseline
  GAS_TDA       : TDA modified logit space (C-dim per sample)
"""
from __future__ import annotations
import sys, math, numpy as np, torch, torch.nn.functional as F
import pandas as pd, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.FreeTTA import FreeTTA
from models.TDA import TDA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA   = Path("data/processed")
OUT    = Path("outputs/deep_analysis")
DATASETS = ["caltech", "dtd", "eurosat", "pets"]
CLIP_SC  = 100.0
CHECKPOINTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

print(f"Device: {DEVICE}")

# ─── Data loading ─────────────────────────────────────────────────────────────

def load(ds: str):
    img = torch.tensor(np.load(DATA / f"{ds}_image_features.npy"),
                       dtype=torch.float32, device=DEVICE)
    txt = torch.tensor(np.load(DATA / f"{ds}_text_features.npy"),
                       dtype=torch.float32, device=DEVICE)
    lbl = torch.tensor(np.load(DATA / f"{ds}_labels.npy"),
                       dtype=torch.long, device=DEVICE)
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    return img, txt, lbl

# ─── Core GAS primitives ───────────────────────────────────────────────────────

@torch.no_grad()
def oracle_centroids(feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean of image features per GT class, L2-normalised."""
    C = int(labels.max().item()) + 1
    D = feats.shape[1]
    mu = torch.zeros(C, D, device=feats.device)
    cnt = torch.zeros(C, device=feats.device)
    mu.scatter_add_(0, labels.unsqueeze(1).expand_as(feats), feats)
    cnt.scatter_add_(0, labels, torch.ones(len(labels), device=feats.device))
    return F.normalize(mu / (cnt.unsqueeze(1) + 1e-8), dim=-1)

@torch.no_grad()
def centroid_acc(feats: torch.Tensor, labels: torch.Tensor,
                 centroids: torch.Tensor) -> float:
    """Classify each sample by nearest centroid (cosine similarity)."""
    preds = (F.normalize(feats, dim=-1) @ F.normalize(centroids, dim=-1).T).argmax(1)
    return (preds == labels).float().mean().item() * 100

@torch.no_grad()
def nn1_loo_acc(feats: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Leave-one-out 1-NN accuracy in feats space.
    Works in float32; for N>5000 computes in row-blocks to stay under 2GB.
    """
    feats = F.normalize(feats, dim=-1)
    N = feats.shape[0]
    preds = torch.empty(N, dtype=torch.long, device=feats.device)
    BLOCK = 512   # rows per block
    for i in range(0, N, BLOCK):
        s  = feats[i:i+BLOCK] @ feats.T          # (B, N)
        # mask out self
        for b in range(s.shape[0]):
            s[b, i + b] = -float("inf")
        preds[i:i+BLOCK] = labels[s.argmax(1)]
    return (preds == labels).float().mean().item() * 100

def gas(cent_a: float, nn_a: float) -> float:
    return cent_a - nn_a

# ─── TDA logit-space helpers ───────────────────────────────────────────────────

@torch.no_grad()
def tda_logits_from_state(img: torch.Tensor, txt: torch.Tensor,
                           model: TDA) -> torch.Tensor:
    """
    Recompute TDA fused logits for ALL samples using the current (frozen) cache.
    Returns (N, C) tensor.
    """
    clip  = CLIP_SC * (img @ txt.T)                                    # (N, C)
    pcap  = model.pos_cap

    # Positive affinity
    p_aff = torch.einsum("nd,csd->ncs", img, model.pos_feat)           # (N, C, pcap)
    slot  = (torch.arange(pcap, device=DEVICE) < model.pos_counts.unsqueeze(1)).float()
    p_w   = torch.exp(-(model.beta - model.beta * p_aff)) * slot       # (N, C, pcap)
    pos_t = p_w.sum(-1)                                                 # (N, C)

    s_tda = clip + model.alpha * pos_t

    if model.neg_size > 0:
        ncap  = model.neg_cap
        n_aff = torch.einsum("nd,csd->ncs", img, model.neg_feat)
        nslot = (torch.arange(ncap, device=DEVICE) < model.neg_counts.unsqueeze(1)).float()
        n_w   = torch.exp(-(model.neg_beta - model.neg_beta * n_aff)) * nslot
        neg_t = torch.einsum("ncs,csk->nk", n_w, model.neg_pmaps)
        s_tda = s_tda - model.neg_alpha * neg_t

    return s_tda   # (N, C)

# ─── Snapshot runners ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_freetta_snapshots(img: torch.Tensor, txt: torch.Tensor,
                           cfg: dict, checkpoints: list) -> dict:
    """
    Run FreeTTA sample-by-sample.  Return {frac: mu_c} snapshots at each checkpoint.
    """
    model = FreeTTA(txt, device=DEVICE, clip_scale=CLIP_SC, **cfg)
    N = img.shape[0]
    clip_all = img @ txt.T   # (N, C) — unscaled dot products

    snaps = {}   # frac -> mu tensor copy
    cp_set = sorted(checkpoints)
    cp_idx = 0

    for i in range(N):
        x  = img[i]
        cl = clip_all[i]

        cp_probs = F.softmax(cl * CLIP_SC, dim=0)
        h   = -(cp_probs * torch.log(cp_probs + 1e-8)).sum()
        nh  = (h / model.max_entropy).clamp(0.0, 1.0)
        w   = torch.exp(-model.beta * nh)

        # M-step
        delta  = w * cp_probs
        Ny_new = model.Ny + delta
        model.mu = (model.Ny.unsqueeze(1) * model.mu
                    + delta.unsqueeze(1) * x.unsqueeze(0)) / (Ny_new.unsqueeze(1) + 1e-8)
        if model.normalize_mu:
            model.mu = F.normalize(model.mu, dim=-1)
        model.Ny = Ny_new

        # Snapshot if we've crossed the next checkpoint threshold
        while cp_idx < len(cp_set) and (i + 1) / N >= cp_set[cp_idx] - 1e-9:
            snaps[cp_set[cp_idx]] = model.mu.clone()
            cp_idx += 1

    return snaps

@torch.no_grad()
def run_tda_snapshots(img: torch.Tensor, txt: torch.Tensor,
                       cfg: dict, checkpoints: list) -> tuple:
    """
    Run TDA sample-by-sample.  Return list of (frac, TDA_model_copy) snapshots.
    """
    model = TDA(txt, device=DEVICE, clip_scale=CLIP_SC, **cfg)
    N = img.shape[0]
    clip_all = CLIP_SC * (img @ txt.T)

    snaps = {}   # frac -> shallow TDA snapshot (just cache tensors)
    cp_set = sorted(checkpoints)
    cp_idx = 0

    for i in range(N):
        x  = img[i]
        cl = clip_all[i]

        probs = F.softmax(cl, dim=0)
        pred  = int(cl.argmax().item())
        h     = -(probs * torch.log(probs + 1e-12)).sum().item()
        nh    = min(h / model.max_entropy, 1.0)
        loss  = h

        model._update_slot(False, pred, x.detach(), loss)
        if model.low_entropy < nh < model.high_entropy:
            model._update_slot(True, pred, x.detach(), loss, pmap=probs)

        while cp_idx < len(cp_set) and (i + 1) / N >= cp_set[cp_idx] - 1e-9:
            # Store cache tensors (cheap clone)
            snaps[cp_set[cp_idx]] = dict(
                pos_feat   = model.pos_feat.clone(),
                pos_counts = model.pos_counts.clone(),
                neg_feat   = model.neg_feat.clone(),
                neg_counts = model.neg_counts.clone(),
                neg_pmaps  = model.neg_pmaps.clone(),
            )
            cp_idx += 1

    return snaps, model

# ─── Main analysis loop ────────────────────────────────────────────────────────

rows_static  = []
rows_time    = []

for ds in DATASETS:
    img, txt, lbl = load(ds)
    N, C = img.shape[0], txt.shape[0]
    print(f"\n{'='*60}")
    print(f"  {ds.upper()}  N={N}  C={C}")
    print(f"{'='*60}")

    ft_cfg  = FreeTTA.DATASET_DEFAULTS[ds]
    tda_cfg = {k: v for k, v in TDA.DATASET_DEFAULTS[ds].items()
               if k not in ("clip_scale",)}  # clip_scale passed separately

    # ── 1-NN baseline in CLIP feature space (shared denominator) ─────────────
    print("  Computing CLIP 1-NN (LOO)...")
    nn_clip = nn1_loo_acc(img, lbl)
    print(f"    1-NN CLIP = {nn_clip:.2f}%")

    # ── CLIP text-centroid (what CLIP actually does) ──────────────────────────
    clip_text_acc  = centroid_acc(img, lbl, txt)
    gas_clip_text  = gas(clip_text_acc, nn_clip)
    print(f"    Centroid (text features) = {clip_text_acc:.2f}%  "
          f"GAS_text = {gas_clip_text:+.2f}")

    # ── CLIP oracle centroid (mean of test images per class) ──────────────────
    mu_oracle      = oracle_centroids(img, lbl)
    clip_oracle_acc = centroid_acc(img, lbl, mu_oracle)
    gas_clip_oracle = gas(clip_oracle_acc, nn_clip)
    print(f"    Centroid (oracle img)    = {clip_oracle_acc:.2f}%  "
          f"GAS_oracle = {gas_clip_oracle:+.2f}")

    # ── FreeTTA: run with checkpoints ─────────────────────────────────────────
    print("  Running FreeTTA with snapshots...")
    ft_snaps = run_freetta_snapshots(img, txt, ft_cfg, CHECKPOINTS)
    mu_final = ft_snaps[1.0]

    ft_cent_acc  = centroid_acc(img, lbl, mu_final)
    gas_freetta  = gas(ft_cent_acc, nn_clip)
    # Also run full FreeTTA for overall accuracy
    ft_model = FreeTTA(txt, device=DEVICE, clip_scale=CLIP_SC, **ft_cfg)
    ft_preds, clip_preds = ft_model.run(img)
    ft_acc   = (ft_preds.cpu() == lbl.cpu()).float().mean().item() * 100
    clip_acc = (clip_preds.cpu() == lbl.cpu()).float().mean().item() * 100

    print(f"    FreeTTA centroid acc = {ft_cent_acc:.2f}%  "
          f"GAS_FreeTTA = {gas_freetta:+.2f}  "
          f"overall={ft_acc:.2f}%  clip={clip_acc:.2f}%")

    # ── TDA: run with checkpoints ──────────────────────────────────────────────
    print("  Running TDA with snapshots...")
    tda_snaps, tda_final = run_tda_snapshots(img, txt, tda_cfg, CHECKPOINTS)

    # Logit space with final cache
    s_tda = tda_logits_from_state(img, txt, tda_final)          # (N, C)
    s_norm = F.normalize(s_tda, dim=-1)

    tda_cent_logit = oracle_centroids(s_norm, lbl)              # centroid in logit space
    tda_cent_acc   = centroid_acc(s_norm, lbl, tda_cent_logit)
    tda_nn_acc     = nn1_loo_acc(s_norm, lbl)                   # 1-NN in logit space
    gas_tda        = gas(tda_cent_acc, tda_nn_acc)

    # TDA overall accuracy (from argmax of s_tda)
    tda_acc = (s_tda.argmax(1).cpu() == lbl.cpu()).float().mean().item() * 100

    print(f"    TDA logit centroid = {tda_cent_acc:.2f}%  "
          f"TDA logit 1-NN = {tda_nn_acc:.2f}%  "
          f"GAS_TDA = {gas_tda:+.2f}  overall={tda_acc:.2f}%")

    # ── Static results table ──────────────────────────────────────────────────
    for method, ca, na, g, acc in [
        ("CLIP_text",    clip_text_acc,  nn_clip,    gas_clip_text,  clip_acc),
        ("CLIP_oracle",  clip_oracle_acc, nn_clip,   gas_clip_oracle, clip_acc),
        ("FreeTTA",      ft_cent_acc,    nn_clip,    gas_freetta,    ft_acc),
        ("TDA_logit",    tda_cent_acc,   tda_nn_acc, gas_tda,        tda_acc),
    ]:
        rows_static.append(dict(
            dataset=ds, method=method,
            centroid_acc=round(ca, 2), nn1_acc=round(na, 2),
            gas=round(g, 2), overall_acc=round(acc, 2),
            clip_acc=round(clip_acc, 2), acc_gain=round(acc - clip_acc, 2),
        ))

    # ── Time-evolution of GAS ──────────────────────────────────────────────────
    print("  Computing GAS time-evolution...")
    for frac in CHECKPOINTS:
        t = int(round(frac * N))

        # FreeTTA: use μ at checkpoint, classify all N samples
        mu_t      = ft_snaps[frac]
        ft_ca_t   = centroid_acc(img, lbl, mu_t)
        gas_ft_t  = gas(ft_ca_t, nn_clip)

        # TDA: recompute logits for all N using cache at checkpoint
        snap = tda_snaps[frac]
        # Temporarily patch model with checkpoint cache
        _pf, _pc = tda_final.pos_feat, tda_final.pos_counts
        _nf, _nc, _np = tda_final.neg_feat, tda_final.neg_counts, tda_final.neg_pmaps
        tda_final.pos_feat   = snap["pos_feat"]
        tda_final.pos_counts = snap["pos_counts"]
        tda_final.neg_feat   = snap["neg_feat"]
        tda_final.neg_counts = snap["neg_counts"]
        tda_final.neg_pmaps  = snap["neg_pmaps"]

        s_t       = tda_logits_from_state(img, txt, tda_final)
        s_t_norm  = F.normalize(s_t, dim=-1)
        tda_cen_t = oracle_centroids(s_t_norm, lbl)
        tda_ca_t  = centroid_acc(s_t_norm, lbl, tda_cen_t)
        tda_na_t  = nn1_loo_acc(s_t_norm, lbl)
        gas_tda_t = gas(tda_ca_t, tda_na_t)

        # Restore
        tda_final.pos_feat, tda_final.pos_counts = _pf, _pc
        tda_final.neg_feat, tda_final.neg_counts, tda_final.neg_pmaps = _nf, _nc, _np

        rows_time.append(dict(
            dataset=ds, frac=frac, samples_seen=t,
            gas_clip_text=round(gas_clip_text, 2),
            gas_clip_oracle=round(gas_clip_oracle, 2),
            gas_freetta=round(gas_ft_t, 2),
            gas_tda=round(gas_tda_t, 2),
            ft_centroid_acc=round(ft_ca_t, 2),
            tda_centroid_acc=round(tda_ca_t, 2),
            tda_nn_acc=round(tda_na_t, 2),
        ))
        print(f"    t={t:5d} ({frac:.0%})  "
              f"GAS_FT={gas_ft_t:+.2f}  GAS_TDA={gas_tda_t:+.2f}")

# ─── Save CSVs ────────────────────────────────────────────────────────────────
df_static = pd.DataFrame(rows_static)
df_time   = pd.DataFrame(rows_time)
df_static.to_csv(OUT / "gas_extended_static.csv",   index=False)
df_time.to_csv(  OUT / "gas_extended_time.csv",     index=False)
print(f"\nCSVs saved.")

# ─── Print summary table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  GEOMETRY ALIGNMENT SCORE — SUMMARY")
print("="*80)
print(f"{'Dataset':10s} {'Method':15s} {'Centroid%':>10s} {'1-NN%':>8s} "
      f"{'GAS':>8s} {'OvAcc%':>8s} {'AccGain':>8s}")
print("-"*70)
for _, r in df_static.iterrows():
    print(f"{r.dataset:10s} {r.method:15s} {r.centroid_acc:10.2f} {r.nn1_acc:8.2f} "
          f"{r.gas:+8.2f} {r.overall_acc:8.2f} {r.acc_gain:+8.2f}")

# ─── Plotting ─────────────────────────────────────────────────────────────────
COLORS = {
    "CLIP_text":   "#78909C",
    "CLIP_oracle": "#37474F",
    "FreeTTA":     "#E53935",
    "TDA_logit":   "#1E88E5",
}
DS_COLORS = {"caltech": "#2196F3", "dtd": "#FF5722",
             "eurosat": "#4CAF50", "pets": "#9C27B0"}

# ── Plot 1: GAS comparison bar chart ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
methods = ["CLIP_text", "CLIP_oracle", "FreeTTA", "TDA_logit"]
labels  = ["CLIP\n(text centroid)", "CLIP\n(oracle centroid)",
           "FreeTTA\n(adapted µ)", "TDA\n(logit space)"]
x = np.arange(len(DATASETS))
w = 0.20
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

for i, (meth, lab) in enumerate(zip(methods, labels)):
    vals = [df_static[(df_static.dataset == ds) & (df_static.method == meth)]["gas"].values[0]
            for ds in DATASETS]
    bars = ax.bar(x + offsets[i], vals, w, label=lab, color=COLORS[meth],
                  alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v >= 0 else -0.3),
                f"{v:+.1f}", ha="center", va=va, fontsize=7.5)

ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
ax.set_xticks(x)
ax.set_xticklabels([d.upper() for d in DATASETS], fontsize=11)
ax.set_ylabel("GAS = Acc_centroid − Acc_1NN  (pp)", fontsize=11)
ax.set_title("Geometry Alignment Score: CLIP vs FreeTTA vs TDA", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "gas_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gas_comparison_bar.png")

# ── Plot 2: GAS time-evolution per dataset ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
fig.suptitle("GAS Evolution Over Test Stream: FreeTTA vs TDA", fontsize=13, fontweight="bold")
axes = axes.flatten()

for idx, ds in enumerate(DATASETS):
    ax  = axes[idx]
    sub = df_time[df_time.dataset == ds]
    pct = sub.frac.values * 100

    ax.axhline(sub.gas_clip_text.iloc[0],   color=COLORS["CLIP_text"],
               linestyle=":", linewidth=1.5, label="CLIP text centroid (static)")
    ax.axhline(sub.gas_clip_oracle.iloc[0], color=COLORS["CLIP_oracle"],
               linestyle="--", linewidth=1.5, label="CLIP oracle centroid (static)")
    ax.plot(pct, sub.gas_freetta, marker="o", color=COLORS["FreeTTA"],
            linewidth=2.5, markersize=5, label="GAS_FreeTTA(t)")
    ax.plot(pct, sub.gas_tda,     marker="s", color=COLORS["TDA_logit"],
            linewidth=2.5, markersize=5, linestyle="--", label="GAS_TDA(t)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.fill_between(pct, 0, sub.gas_freetta.values,
                    where=sub.gas_freetta.values > 0,
                    alpha=0.08, color=COLORS["FreeTTA"])
    ax.set_title(f"{ds.upper()}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Stream progress (%)")
    ax.set_ylabel("GAS (pp)")
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "gas_time_evolution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gas_time_evolution.png")

# ── Plot 3: GAS_FreeTTA vs accuracy gain scatter ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("GAS vs Accuracy Gain: FreeTTA and TDA", fontsize=13, fontweight="bold")

ft_rows  = df_static[df_static.method == "FreeTTA"]
tda_rows = df_static[df_static.method == "TDA_logit"]

for ax, rows, meth, col in [
    (axes[0], ft_rows,  "FreeTTA", COLORS["FreeTTA"]),
    (axes[1], tda_rows, "TDA",     COLORS["TDA_logit"]),
]:
    for _, r in rows.iterrows():
        c = DS_COLORS[r.dataset]
        ax.scatter(r.gas, r.acc_gain, s=140, color=c, edgecolor="white",
                   linewidth=1.2, zorder=5)
        ax.annotate(r.dataset.upper(), (r.gas, r.acc_gain),
                    xytext=(5, 4), textcoords="offset points", fontsize=9, color=c)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GAS (Centroid − 1-NN,  pp)", fontsize=10)
    ax.set_ylabel("Accuracy Gain vs CLIP  (pp)", fontsize=10)
    ax.set_title(f"{meth}: GAS vs Accuracy Gain", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Correlation annotation
for ax, rows in [(axes[0], ft_rows), (axes[1], tda_rows)]:
    if len(rows) > 2:
        r = rows["gas"].corr(rows["acc_gain"])
        ax.text(0.05, 0.93, f"ρ = {r:.2f}", transform=ax.transAxes,
                fontsize=10, color="black")

plt.tight_layout()
plt.savefig(OUT / "gas_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gas_vs_accuracy.png")

# ── Plot 4: Component breakdown (centroid acc + 1-NN acc + GAS) ───────────────
fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 5))
fig.suptitle("GAS Component Breakdown per Dataset", fontsize=13, fontweight="bold")

for idx, ds in enumerate(DATASETS):
    ax  = axes[idx]
    sub = df_static[df_static.dataset == ds]

    method_order = ["CLIP_text", "CLIP_oracle", "FreeTTA", "TDA_logit"]
    m_labels     = ["CLIP\ntext", "CLIP\noracle", "FreeTTA", "TDA\nlogit"]
    x_pos        = np.arange(len(method_order))

    ca  = [sub[sub.method == m]["centroid_acc"].values[0] for m in method_order]
    na  = [sub[sub.method == m]["nn1_acc"].values[0]      for m in method_order]
    ga  = [sub[sub.method == m]["gas"].values[0]          for m in method_order]

    ax.bar(x_pos - 0.2, ca, 0.35, label="Centroid Acc", color="#4CAF50", alpha=0.8)
    ax.bar(x_pos + 0.2, na, 0.35, label="1-NN Acc",     color="#FF9800", alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(x_pos, ga, "D-", color="#E53935", linewidth=2, markersize=8, label="GAS")
    ax2.axhline(0, color="#E53935", linewidth=0.7, linestyle="--", alpha=0.5)
    ax2.set_ylabel("GAS (pp)", color="#E53935", fontsize=9)
    ax2.tick_params(axis="y", colors="#E53935")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_labels, fontsize=8)
    ax.set_title(ds.upper(), fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, min(ca + na) - 10), max(ca + na) + 5)
    ax.grid(axis="y", alpha=0.2)

    if idx == 0:
        ax.legend(loc="lower left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "gas_component_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: gas_component_breakdown.png")

print(f"\nAll outputs in: {OUT}/")
print("CSVs: gas_extended_static.csv, gas_extended_time.csv")
print("PNGs: gas_comparison_bar.png, gas_time_evolution.png,")
print("      gas_vs_accuracy.png, gas_component_breakdown.png")
