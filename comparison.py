from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

from models.FreeTTA import FreeTTA
from models.TDA import TDA
from src.feature_store import list_available_datasets, load_dataset_features


# Targets read from your table image.
PAPER_TARGETS = {
    "vit_b16": {
        "caltech": 94.63,
        "dtd": 46.96,
        "eurosat": 62.93,
        "pets": 90.11,
    },
    "rn50": {
        "caltech": 90.12,
        "dtd": 44.21,
        "eurosat": 43.64,
        "pets": 86.44,
    },
}


PAPER_TDA_DEFAULTS = {
    "dtd": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 2.0,
        "beta": 3.0,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.05,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "caltech": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 0.75,
        "beta": 1.5,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.0,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "eurosat": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 1.45,
        "beta": 3.2,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.0,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": True,
        "fallback_margin": 0.0,
    },
    "pets": {
        "cache_size": 1000,
        "shot_capacity": 3,
        "k": 0,
        "alpha": 5.9,
        "beta": 8.9,
        "low_entropy_thresh": 0.2,
        "high_entropy_thresh": 0.5,
        "neg_alpha": 0.32,
        "neg_beta": 1.0,
        "neg_mask_lower": 0.03,
        "neg_mask_upper": 1.0,
        "clip_scale": 100.0,
        "fallback_to_clip": False,
        "fallback_margin": 0.0,
    },
}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_overrides(path: Path | None, method: str) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    best_per_dataset = payload.get("best_per_dataset", {})
    overrides: Dict[str, Dict[str, float]] = {}

    for dataset_key, row in best_per_dataset.items():
        key = str(dataset_key).lower()
        if method == "freetta":
            alpha = row.get("alpha")
            beta = row.get("beta")
            if alpha is not None and beta is not None:
                overrides[key] = {"alpha": float(alpha), "beta": float(beta)}
        elif method == "tda":
            current = {}
            for f in [
                "cache_size",
                "shot_capacity",
                "k",
                "alpha",
                "beta",
                "low_entropy_thresh",
                "high_entropy_thresh",
                "neg_alpha",
                "neg_beta",
                "neg_mask_lower",
                "neg_mask_upper",
                "clip_scale",
                "fallback_to_clip",
                "fallback_margin",
            ]:
                if f in row and row[f] is not None:
                    current[f] = row[f]
            if current:
                overrides[key] = current

    return overrides


def evaluate_dataset(
    dataset: str,
    features_dir: Path,
    device: torch.device,
    max_samples: int | None,
    tda_params: Dict[str, float],
    freetta_alpha: float,
    freetta_beta: float,
) -> Dict[str, float]:
    payload = load_dataset_features(features_dir, dataset)

    image_features = payload["image_features"]
    text_features = payload["text_features"]
    labels = payload["labels"]

    if max_samples is not None:
        image_features = image_features[:max_samples]
        labels = labels[:max_samples]

    image_t = torch.as_tensor(image_features, dtype=torch.float32, device=device)
    image_t = F.normalize(image_t, dim=-1)

    text_t = torch.as_tensor(text_features, dtype=torch.float32, device=device)
    text_t = F.normalize(text_t, dim=-1)

    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)

    total = int(labels_t.numel())

    tda = TDA(text_features=text_t, device=device, **tda_params)
    freetta = FreeTTA(
        text_features=text_t,
        alpha=freetta_alpha,
        beta=freetta_beta,
        device=device,
    )

    clip_correct = 0
    tda_correct = 0
    freetta_correct = 0

    t0 = time.perf_counter()
    with torch.inference_mode():
        clip_logits_all = image_t @ text_t.T
        clip_pred_all = torch.argmax(clip_logits_all, dim=-1)
        clip_correct = int((clip_pred_all == labels_t).sum().item())
    clip_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.inference_mode():
        for i in range(total):
            pred_tda, _, _ = tda.predict(image_t[i])
            tda_correct += int((pred_tda == labels_t[i]).item())
    tda_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    with torch.inference_mode():
        for i in range(total):
            x = image_t[i]
            clip_logits = 100.0 * (x @ text_t.T)
            pred_freetta, _ = freetta.predict(x, clip_logits)
            freetta_correct += int((pred_freetta.squeeze(0) == labels_t[i]).item())
    freetta_seconds = time.perf_counter() - t2

    return {
        "samples": float(total),
        "clip_acc": float(clip_correct / max(total, 1)),
        "tda_acc": float(tda_correct / max(total, 1)),
        "freetta_acc": float(freetta_correct / max(total, 1)),
        "clip_ms_per_sample": float((clip_seconds / max(total, 1)) * 1000.0),
        "tda_ms_per_sample": float((tda_seconds / max(total, 1)) * 1000.0),
        "freetta_ms_per_sample": float((freetta_seconds / max(total, 1)) * 1000.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple TDA vs FreeTTA comparison")
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--freetta-alpha", type=float, default=0.05)
    parser.add_argument("--freetta-beta", type=float, default=7.0)

    parser.add_argument(
        "--freetta-params-json",
        type=Path,
        default=Path("outputs/tuning/best_freetta_params.json"),
    )
    parser.add_argument(
        "--tda-params-json",
        type=Path,
        default=Path("outputs/tuning/best_tda_params.json"),
    )
    parser.add_argument("--disable-per-dataset-params", action="store_true")
    parser.add_argument(
        "--paper-backbone",
        choices=["vit_b16", "rn50", "off"],
        default="vit_b16",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {args.features_dir}")

    device = resolve_device(args.device)
    available = list_available_datasets(args.features_dir)
    if not available:
        raise RuntimeError(f"No dataset feature triplets found in {args.features_dir}")

    datasets = [d.lower() for d in (args.datasets if args.datasets else available)]

    tda_overrides = {}
    freetta_overrides = {}
    if not args.disable_per_dataset_params:
        tda_overrides = load_best_overrides(args.tda_params_json, method="tda")
        freetta_overrides = load_best_overrides(args.freetta_params_json, method="freetta")

    print(f"Device: {device}")
    print(f"Available datasets: {available}")
    print(f"Running datasets: {datasets}")

    all_rows = []
    for ds in datasets:
        tda_params = dict(PAPER_TDA_DEFAULTS.get(ds, PAPER_TDA_DEFAULTS["dtd"]))
        tda_params.update(tda_overrides.get(ds, {}))

        freetta_cfg = {
            "alpha": args.freetta_alpha,
            "beta": args.freetta_beta,
        }
        freetta_cfg.update(freetta_overrides.get(ds, {}))

        row = evaluate_dataset(
            dataset=ds,
            features_dir=args.features_dir,
            device=device,
            max_samples=args.max_samples,
            tda_params=tda_params,
            freetta_alpha=float(freetta_cfg["alpha"]),
            freetta_beta=float(freetta_cfg["beta"]),
        )
        all_rows.append((ds, row))

        print(
            f"[{ds}] samples={int(row['samples'])} | "
            f"CLIP={row['clip_acc']*100:.2f}% ({row['clip_ms_per_sample']:.3f} ms/sample) | "
            f"TDA={row['tda_acc']*100:.2f}% ({row['tda_ms_per_sample']:.3f} ms/sample) | "
            f"FreeTTA={row['freetta_acc']*100:.2f}% ({row['freetta_ms_per_sample']:.3f} ms/sample)"
        )

    if all_rows:
        print("\nSummary")
        targets = PAPER_TARGETS.get(args.paper_backbone, {}) if args.paper_backbone != "off" else {}
        for ds, row in all_rows:
            tda_gap_clip = (row["tda_acc"] - row["clip_acc"]) * 100.0
            freetta_gap_tda = (row["freetta_acc"] - row["tda_acc"]) * 100.0
            line = (
                f"- {ds}: CLIP={row['clip_acc']*100:.2f}% | "
                f"TDA={row['tda_acc']*100:.2f}% (vs CLIP {tda_gap_clip:+.2f} pts) | "
                f"FreeTTA={row['freetta_acc']*100:.2f}% (vs TDA {freetta_gap_tda:+.2f} pts)"
            )
            target = targets.get(ds.lower())
            if target is not None:
                line += f" | target({args.paper_backbone})={target:.2f}% | TDA_gap={row['tda_acc']*100.0 - target:+.2f} pts"
            print(line)


if __name__ == "__main__":
    main()
