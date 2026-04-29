"""Minimal verification: just load features and compute accuracy."""
import sys, torch, numpy as np, torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_store import load_dataset_features
from models.TDA import TDA
from models.FreeTTA import FreeTTA

FEAT = ROOT / "data" / "processed"

# Updated configs
TDA_CFG = {
    "caltech":  dict(alpha=5.0, beta=5.0,  neg_alpha=0.117, neg_beta=1.0, low_entropy_thresh=0.2,  high_entropy_thresh=0.5, pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "dtd":      dict(alpha=4.0, beta=4.5,  neg_alpha=0.05,  neg_beta=1.0, low_entropy_thresh=0.05, high_entropy_thresh=0.4, pos_shot_capacity=5, neg_shot_capacity=2, clip_scale=100.0),
    "eurosat":  dict(alpha=4.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0, low_entropy_thresh=0.2,  high_entropy_thresh=0.5, pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "pets":     dict(alpha=2.0, beta=7.0,  neg_alpha=0.117, neg_beta=1.0, low_entropy_thresh=0.2,  high_entropy_thresh=0.5, pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
    "imagenet": dict(alpha=1.0, beta=8.0,  neg_alpha=0.117, neg_beta=1.0, low_entropy_thresh=0.2,  high_entropy_thresh=0.5, pos_shot_capacity=3, neg_shot_capacity=2, clip_scale=100.0),
}
FTTA_CFG = {
    "caltech":  dict(alpha=0.02,  beta=3.0),
    "dtd":      dict(alpha=0.1,   beta=3.0),
    "eurosat":  dict(alpha=0.8,   beta=3.0),
    "pets":     dict(alpha=0.25,  beta=4.0),
    "imagenet": dict(alpha=0.05,  beta=4.0),
}

for ds in ["caltech", "dtd", "eurosat", "pets", "imagenet"]:
    d = load_dataset_features(FEAT, ds)
    img = F.normalize(torch.tensor(d["image_features"], dtype=torch.float32), dim=-1)
    txt = F.normalize(torch.tensor(d["text_features"],  dtype=torch.float32), dim=-1)
    lbl = torch.tensor(d["labels"], dtype=torch.long)

    clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()

    m = TDA(txt, **TDA_CFG[ds], device="cpu")
    tda_preds, _ = m.run(img)
    tda_acc = (tda_preds.cpu().numpy() == lbl.numpy()).mean()

    m = FreeTTA(txt, **FTTA_CFG[ds], device="cpu")
    ftta_preds, _ = m.run(img)
    ftta_acc = (ftta_preds.cpu().numpy() == lbl.numpy()).mean()

    winner = "TDA" if tda_acc > ftta_acc else ("FreeTTA" if ftta_acc > tda_acc else "TIE")
    print(f"{ds:10s} CLIP={clip_acc:.4f} TDA={tda_acc:.4f} FreeTTA={ftta_acc:.4f} winner={winner}", flush=True)
