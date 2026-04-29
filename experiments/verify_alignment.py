"""Quick accuracy verification to check DTD/Pets ordering matches paper."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import load_dataset_features
from src.paper_configs import PAPER_TDA_DEFAULTS, DEFAULT_FREETTA_PARAMS
from models.TDA import TDA
from models.FreeTTA import FreeTTA

FEAT_DIR = PROJECT_ROOT / "data" / "processed"
DATASETS = ["caltech", "dtd", "eurosat", "pets", "imagenet"]

results = []
for ds in DATASETS:
    d = load_dataset_features(FEAT_DIR, ds)
    img = F.normalize(torch.tensor(d["image_features"], dtype=torch.float32), dim=-1)
    txt = F.normalize(torch.tensor(d["text_features"],  dtype=torch.float32), dim=-1)
    lbl = torch.tensor(d["labels"], dtype=torch.long)

    clip_acc = ((img @ txt.t()).argmax(1).numpy() == lbl.numpy()).mean()

    tda_params = dict(PAPER_TDA_DEFAULTS[ds])
    tda = TDA(txt, **tda_params, device="cpu")
    tda_preds, _ = tda.run(img)
    tda_acc = (tda_preds.cpu().numpy() == lbl.numpy()).mean()

    ftta_params = dict(DEFAULT_FREETTA_PARAMS[ds])
    ftta = FreeTTA(txt, **ftta_params, device="cpu")
    ftta_preds, _ = ftta.run(img)
    ftta_acc = (ftta_preds.cpu().numpy() == lbl.numpy()).mean()

    winner = "TDA" if tda_acc > ftta_acc else ("FreeTTA" if ftta_acc > tda_acc else "TIE")
    results.append((ds, clip_acc, tda_acc, ftta_acc, winner))
    print(f"{ds:10s}  CLIP={clip_acc:.4f}  TDA={tda_acc:.4f}  FreeTTA={ftta_acc:.4f}  winner={winner}")

print()
print("Expected: DTD→TDA, all others→FreeTTA")
dtd_ok = results[1][4] == "TDA"
others_ok = all(r[4] in ("FreeTTA", "TIE") for r in results if r[0] != "dtd")
print(f"DTD correct: {dtd_ok}")
print(f"Others correct: {others_ok}")
print(f"PASS: {dtd_ok and others_ok}")
