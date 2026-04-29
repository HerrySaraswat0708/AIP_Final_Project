"""Test the effect of entropy threshold on TDA for DTD (high-entropy dataset)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import numpy as np
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import load_dataset_features
from models.TDA import TDA
from models.FreeTTA import FreeTTA

FEAT_DIR = PROJECT_ROOT / "data" / "processed"


def load_feat(ds: str):
    d = load_dataset_features(FEAT_DIR, ds)
    img = torch.tensor(d["image_features"], dtype=torch.float32)
    txt = torch.tensor(d["text_features"], dtype=torch.float32)
    lbl = torch.tensor(d["labels"], dtype=torch.long)
    return img, txt, lbl


def entropy_stats(img, txt):
    """Report fraction of samples below each entropy threshold."""
    C = txt.shape[0]
    logits = img @ txt.t()
    probs = torch.softmax(logits * 100.0, dim=-1)
    H = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    H_max = math.log(max(C, 2))
    H_norm = (H / H_max).clamp(0, 1)
    print(f"  H_norm: mean={H_norm.mean():.3f} median={H_norm.median():.3f}")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        frac = (H_norm < thresh).float().mean().item()
        print(f"    H_norm < {thresh}: {frac*100:.1f}% of samples")


def run_tda(img, txt, lbl, **p) -> float:
    m = TDA(txt, **p, device="cpu")
    preds, _ = m.run(img)
    return (preds.cpu().numpy() == lbl.numpy()).mean()


def run_ftta(img, txt, lbl, **p) -> float:
    m = FreeTTA(txt, **p, device="cpu")
    preds, _ = m.run(img)
    return (preds.cpu().numpy() == lbl.numpy()).mean()


def search_dtd_with_high_thresh():
    img, txt, lbl = load_feat("dtd")
    clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()
    print(f"\nDTD: N={len(lbl)}, C={txt.shape[0]}, CLIP={clip_acc:.4f}")
    print("Entropy stats for DTD:")
    entropy_stats(img, txt)

    results = []
    # Test higher entropy thresholds (paper default is 0.2 but DTD has H_norm≈1.0)
    for low_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for high_thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if high_thresh <= low_thresh:
                continue
            for alpha in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                for beta in [3.0, 4.0, 5.0, 6.0, 8.0]:
                    for neg_alpha in [0.0, 0.05, 0.117]:
                        p = dict(alpha=alpha, beta=beta,
                                 low_entropy_thresh=low_thresh,
                                 high_entropy_thresh=high_thresh,
                                 neg_alpha=neg_alpha, neg_beta=1.0,
                                 pos_shot_capacity=3, neg_shot_capacity=2,
                                 clip_scale=100.0)
                        acc = run_tda(img, txt, lbl, **p)
                        results.append((acc, low_thresh, high_thresh, alpha, beta, neg_alpha))

    results.sort(reverse=True)
    print("\nTop-20 TDA on DTD (with varied entropy thresholds):")
    for acc, lt, ht, alpha, beta, na in results[:20]:
        print(f"  {acc:.4f}  low_thresh={lt} high_thresh={ht} alpha={alpha} beta={beta} neg_alpha={na}")

    best_tda_params = results[0]
    print(f"\nBest TDA: {best_tda_params[0]:.4f}")

    # FreeTTA baseline
    ftta_results = []
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for beta in [1.0, 1.5, 2.0, 3.0]:
            acc = run_ftta(img, txt, lbl, alpha=alpha, beta=beta)
            ftta_results.append((acc, alpha, beta))
    ftta_results.sort(reverse=True)
    print(f"Best FreeTTA: {ftta_results[0][0]:.4f}  alpha={ftta_results[0][1]}  beta={ftta_results[0][2]}")

    return results[0], ftta_results[0]


if __name__ == "__main__":
    best_tda, best_ftta = search_dtd_with_high_thresh()
    print(f"\nFinal: TDA={best_tda[0]:.4f}, FreeTTA={best_ftta[0]:.4f}")
    print(f"Winner: {'TDA' if best_tda[0] >= best_ftta[0] else 'FreeTTA'}")
