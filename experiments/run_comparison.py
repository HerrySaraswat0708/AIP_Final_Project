"""
Compare FreeTTA vs TDA vs CLIP baseline on 5 datasets.

Usage:
    python experiments/run_comparison.py
    python experiments/run_comparison.py --datasets caltech,dtd,eurosat
    python experiments/run_comparison.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.feature_store import load_dataset_features

# Paper-reported numbers for reference
PAPER = {
    "caltech":  dict(clip=93.36, freetta=94.63, tda=94.24),
    "dtd":      dict(clip=43.87, freetta=46.96, tda=47.40),
    "eurosat":  dict(clip=48.43, freetta=62.93, tda=58.00),
    "pets":     dict(clip=88.25, freetta=90.11, tda=88.63),
    "imagenet": dict(clip=62.05, freetta=64.92, tda=64.67),
}

ALL_DATASETS = ("caltech", "dtd", "eurosat", "pets", "imagenet")


def load_features(features_dir: Path, dataset: str, device: torch.device):
    payload = load_dataset_features(features_dir, dataset)
    img = torch.from_numpy(payload["image_features"]).float()
    txt = torch.from_numpy(payload["text_features"]).float()
    lbl = torch.from_numpy(payload["labels"]).long()
    img = F.normalize(img, dim=-1).to(device)
    txt = F.normalize(txt, dim=-1).to(device)
    lbl = lbl.to(device)
    return img, txt, lbl


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return float((preds == labels).float().mean().item()) * 100.0


def run_dataset(dataset: str, img: torch.Tensor, txt: torch.Tensor,
                lbl: torch.Tensor, device: torch.device) -> dict:
    N = lbl.shape[0]
    results = {"dataset": dataset, "n_samples": N}

    # ── FreeTTA ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model_f = FreeTTA.for_dataset(dataset, txt, device=str(device))
    preds_f, clip_preds = model_f.run(img)
    results["time_freetta"] = time.perf_counter() - t0
    results["freetta_acc"]  = accuracy(preds_f, lbl)
    results["clip_acc"]     = accuracy(clip_preds, lbl)

    # ── TDA ───────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model_t = TDA.for_dataset(dataset, txt, device=str(device))
    preds_t, _ = model_t.run(img)
    results["time_tda"] = time.perf_counter() - t0
    results["tda_acc"]  = accuracy(preds_t, lbl)

    return results


def print_table(all_results: list[dict]) -> None:
    sep = "─" * 100
    print(f"\n{sep}")
    print(f"{'Dataset':<12} {'N':>6}  {'CLIP':>8}  {'FreeTTA':>10}  {'TDA':>8}  "
          f"{'ΔCLIP→F':>9}  {'ΔCLIP→T':>9}  {'Winner':>8}  {'Time_F':>7}  {'Time_T':>7}")
    print(sep)

    freetta_wins = 0
    tda_wins = 0

    for r in all_results:
        ds    = r["dataset"]
        clip  = r["clip_acc"]
        f_acc = r["freetta_acc"]
        t_acc = r["tda_acc"]
        delta_f = f_acc - clip
        delta_t = t_acc - clip
        winner  = "FreeTTA" if f_acc > t_acc else "TDA    "
        if f_acc > t_acc:
            freetta_wins += 1
        else:
            tda_wins += 1

        print(f"{ds:<12} {r['n_samples']:>6}  {clip:>7.2f}%  {f_acc:>9.2f}%  "
              f"{t_acc:>7.2f}%  {delta_f:>+8.2f}%  {delta_t:>+8.2f}%  "
              f"{winner:>8}  {r['time_freetta']:>6.1f}s  {r['time_tda']:>6.1f}s")

    print(sep)
    # Averages
    avg_clip = sum(r["clip_acc"]    for r in all_results) / len(all_results)
    avg_f    = sum(r["freetta_acc"] for r in all_results) / len(all_results)
    avg_t    = sum(r["tda_acc"]     for r in all_results) / len(all_results)
    print(f"{'AVERAGE':<12} {'':>6}  {avg_clip:>7.2f}%  {avg_f:>9.2f}%  {avg_t:>7.2f}%  "
          f"{avg_f-avg_clip:>+8.2f}%  {avg_t-avg_clip:>+8.2f}%  "
          f"{'FreeTTA' if avg_f > avg_t else 'TDA    ':>8}")
    print(sep)

    print(f"\nDataset wins  →  FreeTTA: {freetta_wins}/{len(all_results)}   "
          f"TDA: {tda_wins}/{len(all_results)}")

    # Paper comparison
    paper_ds = [r["dataset"] for r in all_results if r["dataset"] in PAPER]
    if paper_ds:
        print(f"\n{'─'*60}")
        print(f"{'Dataset':<12}  {'FreeTTA (ours)':>15}  {'FreeTTA (paper)':>16}  "
              f"{'TDA (ours)':>12}  {'TDA (paper)':>12}")
        print(f"{'─'*60}")
        for r in all_results:
            ds = r["dataset"]
            if ds not in PAPER:
                continue
            p = PAPER[ds]
            print(f"{ds:<12}  {r['freetta_acc']:>14.2f}%  {p['freetta']:>15.2f}%  "
                  f"{r['tda_acc']:>11.2f}%  {p['tda']:>11.2f}%")
        print(f"{'─'*60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",     default=",".join(ALL_DATASETS))
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--device",       default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output",       default="outputs/comparison_results.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    datasets   = [d.strip() for d in args.datasets.split(",") if d.strip()]
    feats_dir  = ROOT / args.features_dir

    print(f"Device : {device}")
    print(f"Datasets: {datasets}")
    print(f"Features: {feats_dir}\n")

    all_results = []
    for ds in datasets:
        print(f"Loading {ds} ...", end=" ", flush=True)
        img, txt, lbl = load_features(feats_dir, ds, device)
        print(f"{lbl.shape[0]} samples, {txt.shape[0]} classes")

        r = run_dataset(ds, img, txt, lbl, device)
        all_results.append(r)
        print(f"  CLIP={r['clip_acc']:.2f}%  FreeTTA={r['freetta_acc']:.2f}%  "
              f"TDA={r['tda_acc']:.2f}%  "
              f"({'FreeTTA wins' if r['freetta_acc'] > r['tda_acc'] else 'TDA wins'})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_table(all_results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "results": all_results}, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
