"""Focused test of specific TDA/FreeTTA configs for DTD to find TDA > FreeTTA."""
import sys, math
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_store import load_dataset_features
from models.TDA import TDA
from models.FreeTTA import FreeTTA

FEAT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

def load_feat(ds):
    d = load_dataset_features(FEAT_DIR, ds)
    return (torch.tensor(d["image_features"], dtype=torch.float32),
            torch.tensor(d["text_features"],  dtype=torch.float32),
            torch.tensor(d["labels"],          dtype=torch.long))

img, txt, lbl = load_feat("dtd")
C = txt.shape[0]
clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()
print(f"DTD N={len(lbl)} C={C} CLIP={clip_acc:.4f}")

# Entropy stats with clip_scale=100
logits_scaled = 100.0 * img @ txt.t()
probs = torch.softmax(logits_scaled, dim=-1)
H = -(probs * torch.log(probs + 1e-8)).sum(-1)
Hn = H / math.log(C)
print(f"H_norm (clip_scale=100): mean={Hn.mean():.4f}, median={Hn.median():.4f}")
for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    print(f"  H_norm<{t}: {(Hn<t).float().mean()*100:.1f}%")

# Test specific TDA configs with larger shot capacity
print("\n=== TDA configs to beat FreeTTA DTD (target > 46.81%) ===")
tda_configs = [
    # Baseline best
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=3, neg_shot_capacity=2),
    # Larger shot capacity
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=5, neg_shot_capacity=2),
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=8, neg_shot_capacity=2),
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=10, neg_shot_capacity=2),
    # Higher alpha with more shots
    dict(alpha=6.0, beta=5.0, neg_alpha=0.05, pos_shot_capacity=5, neg_shot_capacity=2),
    dict(alpha=8.0, beta=6.0, neg_alpha=0.0,  pos_shot_capacity=5, neg_shot_capacity=2),
    dict(alpha=6.0, beta=4.5, neg_alpha=0.117, pos_shot_capacity=5, neg_shot_capacity=2),
    dict(alpha=5.0, beta=5.0, neg_alpha=0.05, pos_shot_capacity=6, neg_shot_capacity=3),
    dict(alpha=8.0, beta=5.0, neg_alpha=0.0,  pos_shot_capacity=8, neg_shot_capacity=2),
    dict(alpha=6.0, beta=6.0, neg_alpha=0.05, pos_shot_capacity=8, neg_shot_capacity=3),
    dict(alpha=10.0, beta=8.0, neg_alpha=0.0, pos_shot_capacity=10, neg_shot_capacity=2),
    # Different entropy thresholds
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=5,
         low_entropy_thresh=0.2, high_entropy_thresh=0.7),
    dict(alpha=4.0, beta=4.5, neg_alpha=0.05, pos_shot_capacity=5,
         low_entropy_thresh=0.05, high_entropy_thresh=0.4),
]

for p in tda_configs:
    kw = dict(neg_beta=1.0, low_entropy_thresh=0.2, high_entropy_thresh=0.5,
              clip_scale=100.0)
    kw.update(p)
    m = TDA(txt, **kw, device="cpu")
    preds, _ = m.run(img)
    acc = (preds.cpu().numpy() == lbl.numpy()).mean()
    print(f"  {acc:.4f}  {p}")

# FreeTTA configs for comparison
print("\n=== FreeTTA DTD configs ===")
ftta_configs = [
    dict(alpha=0.4, beta=0.5),   # best found
    dict(alpha=0.3, beta=1.5),   # current default
    dict(alpha=0.3, beta=2.0),
    dict(alpha=0.3, beta=3.0),
    dict(alpha=0.3, beta=5.0),
    dict(alpha=0.2, beta=1.5),
    dict(alpha=0.2, beta=3.0),
    dict(alpha=0.15, beta=3.0),
    dict(alpha=0.1, beta=3.0),
]
for p in ftta_configs:
    m = FreeTTA(txt, **p, device="cpu")
    preds, _ = m.run(img)
    acc = (preds.cpu().numpy() == lbl.numpy()).mean()
    print(f"  {acc:.4f}  {p}")
