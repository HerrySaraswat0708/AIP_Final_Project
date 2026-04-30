"""
Experiment 1: Vary negative-cache entropy thresholds — activate the dead gate
Experiment 2: Vary K in TDA with majority-vote instead of affinity-sum
"""
from __future__ import annotations
import sys, math, numpy as np, torch, torch.nn.functional as F, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.TDA import TDA

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA     = Path("data/processed")
OUT      = Path("outputs/deep_analysis")
DATASETS = ["caltech", "dtd", "eurosat", "pets"]   # imagenet text feats mismatch
CLIP_SC  = 100.0

# ── paper defaults ───────────────────────────────────────────────────────────
PAPER_TDA = {
    "caltech":  dict(alpha=5.0, beta=5.0,  neg_alpha=0.117, pos_shot_capacity=3, neg_shot_capacity=2),
    "dtd":      dict(alpha=2.0, beta=3.0,  neg_alpha=0.117, pos_shot_capacity=3, neg_shot_capacity=2),
    "eurosat":  dict(alpha=4.0, beta=8.0,  neg_alpha=0.117, pos_shot_capacity=3, neg_shot_capacity=2),
    "pets":     dict(alpha=2.0, beta=7.0,  neg_alpha=0.117, pos_shot_capacity=3, neg_shot_capacity=2),
}

def load(ds):
    img  = torch.tensor(np.load(DATA / f"{ds}_image_features.npy"), dtype=torch.float32, device=DEVICE)
    txt  = torch.tensor(np.load(DATA / f"{ds}_text_features.npy"),  dtype=torch.float32, device=DEVICE)
    lbl  = torch.tensor(np.load(DATA / f"{ds}_labels.npy"),         dtype=torch.long)
    img  = F.normalize(img, dim=-1)
    txt  = F.normalize(txt, dim=-1)
    return img, txt, lbl

def run_tda(img, txt, lbl, **kwargs):
    model = TDA(txt, device=DEVICE, clip_scale=CLIP_SC, **kwargs)
    preds, _ = model.run(img)
    return (preds.cpu() == lbl).float().mean().item() * 100, model

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Negative-cache threshold sweep
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  EXPERIMENT 1: Negative-Cache Threshold Sweep")
print("="*70)

# Threshold configs: (lo, hi, label)
THRESH_CONFIGS = [
    (0.2,  0.50, "Baseline [0.2-0.50] (paper, 0% fire)"),
    (0.0,  1.01, "All-open  [0.0-1.01] (100% fire)"),
    (0.5,  1.01, "High-H    [0.5-1.01] (100% fire)"),
    (0.8,  1.01, "Very-high [0.8-1.01] (100% fire)"),
]

# neg_alpha sweep (with threshold all-open so gate fires):
NEG_ALPHA_CONFIGS = [0.0, 0.05, 0.117, 0.25, 0.5, 1.0]

exp1_rows = []

for ds in DATASETS:
    img, txt, lbl = load(ds)
    base_cfg = PAPER_TDA[ds]
    clip_acc = (CLIP_SC * (img @ txt.T)).argmax(1).cpu().eq(lbl).float().mean().item() * 100

    # Part A: vary thresholds (keep neg_alpha=0.117)
    for lo, hi, label in THRESH_CONFIGS:
        acc, m = run_tda(img, txt, lbl,
                         low_entropy_thresh=lo, high_entropy_thresh=hi,
                         neg_alpha=base_cfg["neg_alpha"], **{k:v for k,v in base_cfg.items() if k!="neg_alpha"})
        neg_fired = m.neg_size
        row = dict(dataset=ds, experiment="thresh", config=label,
                   low_thresh=lo, high_thresh=hi,
                   neg_alpha=base_cfg["neg_alpha"],
                   neg_cache_used=neg_fired, acc=round(acc,2),
                   vs_clip=round(acc-clip_acc,2))
        exp1_rows.append(row)
        print(f"  {ds:8s} | {label:40s} | neg_used={neg_fired:5d} | acc={acc:.2f}% ({acc-clip_acc:+.2f})")

    # Part B: vary neg_alpha (gate fully open: [0.0, 1.01])
    print()
    for na in NEG_ALPHA_CONFIGS:
        acc, m = run_tda(img, txt, lbl,
                         low_entropy_thresh=0.0, high_entropy_thresh=1.01,
                         neg_alpha=na,
                         **{k:v for k,v in base_cfg.items() if k!="neg_alpha"})
        row = dict(dataset=ds, experiment="neg_alpha", config=f"neg_alpha={na}",
                   low_thresh=0.0, high_thresh=1.01,
                   neg_alpha=na, neg_cache_used=m.neg_size,
                   acc=round(acc,2), vs_clip=round(acc-clip_acc,2))
        exp1_rows.append(row)
        print(f"  {ds:8s} | neg_alpha={na:.3f} (gate fully open)            "
              f"| neg_used={m.neg_size:5d} | acc={acc:.2f}% ({acc-clip_acc:+.2f})")
    print()

e1df = pd.DataFrame(exp1_rows)
e1df.to_csv(OUT / "exp1_negative_cache_sweep.csv", index=False)
print(f"\nExp1 saved → {OUT}/exp1_negative_cache_sweep.csv")


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — K majority-vote
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  EXPERIMENT 2: K-Majority Vote vs TDA Affinity-Sum")
print("="*70)

K_VALUES = [1, 3, 5, 7, 10, 15, 20]

@torch.no_grad()
def run_majority_vote(img: torch.Tensor, txt: torch.Tensor, lbl: torch.Tensor,
                      K: int, pos_cap: int, ds: str) -> tuple[float, float]:
    """
    TDA variant: store exemplars in positive cache identically to TDA,
    but replace the affinity-sum prediction with K-NN majority vote.
    Also report the standard TDA (affinity-sum) accuracy for comparison.
    """
    cfg = PAPER_TDA[ds]
    C, D = txt.shape[0], txt.shape[1]
    N = img.shape[0]
    max_ent = math.log(max(C, 2))

    # Cache storage (same logic as TDA positive cache)
    pos_feat   = torch.zeros(C, pos_cap, D, device=DEVICE)
    pos_loss   = torch.full((C, pos_cap), float("inf"), device=DEVICE)
    pos_counts = torch.zeros(C, dtype=torch.long, device=DEVICE)

    clip_logits_all = CLIP_SC * (img @ txt.T)  # (N, C)
    alpha, beta = cfg["alpha"], cfg["beta"]

    mv_correct  = 0
    tda_correct = 0

    for i in range(N):
        x = img[i]                          # (D,)
        cl = clip_logits_all[i]             # (C,)
        clip_probs = torch.softmax(cl, dim=0)
        clip_pred  = int(cl.argmax().item())
        h = -(clip_probs * torch.log(clip_probs + 1e-12)).sum().item()
        norm_h = min(h / max_ent, 1.0)
        ent_loss = h

        # Update positive cache (same as TDA)
        cnt = int(pos_counts[clip_pred].item())
        if cnt < pos_cap:
            pos_feat[clip_pred, cnt] = x
            pos_loss[clip_pred, cnt] = ent_loss
            pos_counts[clip_pred]   += 1
        else:
            worst_val, worst_idx = pos_loss[clip_pred].max(0)
            if ent_loss < float(worst_val.item()):
                pos_feat[clip_pred, int(worst_idx.item())] = x
                pos_loss[clip_pred, int(worst_idx.item())] = ent_loss

        # ── Standard TDA affinity-sum prediction ─────────────────────────
        aff  = torch.einsum("d,csd->cs", x, pos_feat)          # (C, cap)
        slot = torch.arange(pos_cap, device=DEVICE).unsqueeze(0) < pos_counts.unsqueeze(1)
        w    = torch.exp(-(beta - beta * aff)) * slot.float()   # (C, cap)
        tda_logit = cl + alpha * w.sum(-1)                      # (C,)
        tda_pred  = int(tda_logit.argmax().item())

        # ── K-NN majority-vote prediction ─────────────────────────────────
        # Gather all cached exemplars
        total_cached = int(pos_counts.sum().item())
        if total_cached == 0:
            mv_pred = clip_pred
        else:
            # Build flat list of (feat, class_label)
            feat_list, label_list = [], []
            for c in range(C):
                n = int(pos_counts[c].item())
                if n > 0:
                    feat_list.append(pos_feat[c, :n])       # (n, D)
                    label_list.append(torch.full((n,), c, dtype=torch.long, device=DEVICE))
            all_feats  = torch.cat(feat_list,  dim=0)       # (total, D)
            all_labels = torch.cat(label_list, dim=0)       # (total,)

            sims   = x @ all_feats.T                        # (total,)
            k_act  = min(K, total_cached)
            top_k  = sims.topk(k_act).indices
            votes  = all_labels[top_k]
            # majority: mode
            mv_pred = int(torch.zeros(C, device=DEVICE)
                          .scatter_add_(0, votes, torch.ones(k_act, device=DEVICE))
                          .argmax().item())

        gt = int(lbl[i].item())
        mv_correct  += (mv_pred  == gt)
        tda_correct += (tda_pred == gt)

    mv_acc  = mv_correct  / N * 100
    tda_acc = tda_correct / N * 100
    return mv_acc, tda_acc


exp2_rows = []

for ds in DATASETS:
    img, txt, lbl = load(ds)
    cfg  = PAPER_TDA[ds]
    clip_acc = (CLIP_SC * (img @ txt.T)).argmax(1).cpu().eq(lbl).float().mean().item() * 100
    print(f"\n{ds.upper()}  (CLIP={clip_acc:.2f}%)")

    for K in K_VALUES:
        mv_acc, tda_acc = run_majority_vote(img, txt, lbl, K=K,
                                             pos_cap=cfg["pos_shot_capacity"], ds=ds)
        row = dict(dataset=ds, K=K,
                   clip_acc=round(clip_acc,2),
                   tda_affinity_acc=round(tda_acc,2),
                   majority_vote_acc=round(mv_acc,2),
                   mv_vs_clip=round(mv_acc-clip_acc,2),
                   tda_vs_clip=round(tda_acc-clip_acc,2),
                   mv_vs_tda=round(mv_acc-tda_acc,2))
        exp2_rows.append(row)
        print(f"  K={K:2d} | TDA-affinity={tda_acc:.2f}% ({tda_acc-clip_acc:+.2f}) | "
              f"Majority-vote={mv_acc:.2f}% ({mv_acc-clip_acc:+.2f}) | "
              f"MV-TDA={mv_acc-tda_acc:+.2f}%")

e2df = pd.DataFrame(exp2_rows)
e2df.to_csv(OUT / "exp2_majority_vote.csv", index=False)
print(f"\nExp2 saved → {OUT}/exp2_majority_vote.csv")
