from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.evaluate_tda import evaluate_clip_loaded, evaluate_loaded, load_tda_dataset


BEST_TDA_PARAMS = {
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
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TDA on all datasets with best parameters")
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="*", default=["dtd", "caltech", "eurosat", "pets"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=Path("outputs/tuning/tda_target_matched_params.json"),
        help="Optional JSON containing a top-level `params` mapping (dataset -> param dict).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/tuning/best_tda_run_results.json"),
    )
    return parser.parse_args()


def load_params(args: argparse.Namespace) -> dict:
    if args.params_json.exists():
        payload = json.loads(args.params_json.read_text(encoding="utf-8"))
        params = payload.get("params")
        if isinstance(params, dict) and params:
            return {str(k).lower(): v for k, v in params.items()}
    return BEST_TDA_PARAMS


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    params_map = load_params(args)

    rows = []
    for ds in [d.lower() for d in args.datasets]:
        if ds not in params_map:
            print(f"[Skip] No params configured for dataset='{ds}'")
            continue

        payload = load_tda_dataset(
            dataset=ds,
            device=device,
            max_samples=args.max_samples,
            features_dir=str(args.features_dir),
        )

        clip_acc = evaluate_clip_loaded(payload)
        tda_acc = evaluate_loaded(
            payload=payload,
            device=device,
            **params_map[ds],
        )

        row = {
            "dataset": ds,
            "samples": int(payload["num_samples"]),
            "clip_acc": float(clip_acc),
            "tda_acc": float(tda_acc),
            "gain_vs_clip": float(tda_acc - clip_acc),
            "params": params_map[ds],
        }
        rows.append(row)

        print(
            f"[{ds}] samples={row['samples']} "
            f"CLIP={row['clip_acc']*100:.2f}% "
            f"TDA={row['tda_acc']*100:.2f}% "
            f"gain={row['gain_vs_clip']*100:+.2f} pts"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "features_dir": str(args.features_dir),
        "max_samples": args.max_samples,
        "results": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
