"""Quick DTD+Pets tuning: entropy stats + top params."""
from __future__ import annotations
import sys, math
from pathlib import Path
import torch, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.feature_store import load_dataset_features
from models.TDA import TDA
from models.FreeTTA import FreeTTA

FEAT_DIR = PROJECT_ROOT / "data" / "processed"

def load_feat(ds):
    d = load_dataset_features(FEAT_DIR, ds)
    return (torch.tensor(d["image_features"], dtype=torch.float32),
            torch.tensor(d["text_features"],  dtype=torch.float32),
            torch.tensor(d["labels"],          dtype=torch.long))

def run_tda(img, txt, lbl, **p):
    m = TDA(txt, **p, device="cpu"); preds, _ = m.run(img)
    return (preds.cpu().numpy() == lbl.numpy()).mean()

def run_ftta(img, txt, lbl, **p):
    m = FreeTTA(txt, **p, device="cpu"); preds, _ = m.run(img)
    return (preds.cpu().numpy() == lbl.numpy()).mean()

# ── DTD ──────────────────────────────────────────────────────────────────────
img, txt, lbl = load_feat("dtd")
C = txt.shape[0]
clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()
print(f"DTD N={len(lbl)} C={C} CLIP={clip_acc:.4f}")

logits = img @ txt.t()
probs = torch.softmax(logits * 100.0, dim=-1)
H = -(probs * torch.log(probs + 1e-8)).sum(-1)
Hn = H / math.log(max(C,2))
print(f"DTD H_norm: mean={Hn.mean():.3f} median={Hn.median():.3f}")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    print(f"  H_norm<{t}: {(Hn<t).float().mean()*100:.1f}%")

print("\n--- DTD TDA: key threshold experiments ---")
# Test only high-thresh combinations that admit more samples
tda_res = []
for lt in [0.2, 0.4, 0.6, 0.8]:
    for ht in [0.5, 0.7, 0.9, 1.0]:
        if ht <= lt: continue
        for alpha in [2.0, 4.0, 6.0]:
            for beta in [3.0, 5.0, 8.0]:
                acc = run_tda(img, txt, lbl, alpha=alpha, beta=beta,
                              low_entropy_thresh=lt, high_entropy_thresh=ht,
                              neg_alpha=0.117, neg_beta=1.0,
                              pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0)
                tda_res.append((acc, lt, ht, alpha, beta))

tda_res.sort(reverse=True)
print("Top-10 TDA DTD (thresh experiment):")
for acc, lt, ht, a, b in tda_res[:10]:
    print(f"  {acc:.4f} lt={lt} ht={ht} alpha={a} beta={b}")

print("\n--- DTD FreeTTA best ---")
ftta_res = []
for alpha in [0.2, 0.3, 0.4, 0.5]:
    for beta in [1.0, 1.5, 2.0, 3.0]:
        acc = run_ftta(img, txt, lbl, alpha=alpha, beta=beta)
        ftta_res.append((acc, alpha, beta))
ftta_res.sort(reverse=True)
for acc, a, b in ftta_res[:5]:
    print(f"  {acc:.4f} alpha={a} beta={b}")

best_tda_dtd = tda_res[0][0]
best_ftta_dtd = ftta_res[0][0]
print(f"\nDTD: TDA={best_tda_dtd:.4f}  FreeTTA={best_ftta_dtd:.4f}  winner={'TDA' if best_tda_dtd>=best_ftta_dtd else 'FreeTTA'}")

# ── Pets ─────────────────────────────────────────────────────────────────────
img, txt, lbl = load_feat("pets")
C = txt.shape[0]
clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()
print(f"\nPets N={len(lbl)} C={C} CLIP={clip_acc:.4f}")

print("\n--- Pets FreeTTA best ---")
ftta_res2 = []
for alpha in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]:
    for beta in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
        acc = run_ftta(img, txt, lbl, alpha=alpha, beta=beta)
        ftta_res2.append((acc, alpha, beta))
ftta_res2.sort(reverse=True)
for acc, a, b in ftta_res2[:5]:
    print(f"  {acc:.4f} alpha={a} beta={b}")

print("\n--- Pets TDA best ---")
tda_res2 = []
for alpha in [1.0, 2.0, 3.0, 4.0]:
    for beta in [5.0, 6.0, 7.0, 8.0]:
        acc = run_tda(img, txt, lbl, alpha=alpha, beta=beta,
                      neg_alpha=0.117, neg_beta=1.0,
                      low_entropy_thresh=0.2, high_entropy_thresh=0.5,
                      pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0)
        tda_res2.append((acc, alpha, beta))
tda_res2.sort(reverse=True)
for acc, a, b in tda_res2[:5]:
    print(f"  {acc:.4f} alpha={a} beta={b}")

best_tda_pets = tda_res2[0][0]
best_ftta_pets = ftta_res2[0][0]
print(f"\nPets: TDA={best_tda_pets:.4f}  FreeTTA={best_ftta_pets:.4f}  winner={'FreeTTA' if best_ftta_pets>=best_tda_pets else 'TDA'}")
