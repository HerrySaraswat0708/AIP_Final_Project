from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_tda import evaluate_loaded, load_tda_dataset
from src.paper_configs import PAPER_TDA_DEFAULTS


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search TDA hyperparameters")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--alphas", default="")
    parser.add_argument("--betas", default="")
    parser.add_argument("--neg-alphas", default="0.0,0.05,0.117,0.2,0.3")
    parser.add_argument("--stream-seeds", default="1")
    parser.add_argument("--shuffle-stream", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = load_tda_dataset(dataset=args.dataset, device=device, features_dir=args.features_dir)
    dataset_key = str(payload["dataset"]).lower()
    base = dict(PAPER_TDA_DEFAULTS[dataset_key])

    alphas = parse_float_list(args.alphas) if args.alphas else [float(base["alpha"])]
    betas = parse_float_list(args.betas) if args.betas else [float(base["beta"])]
    neg_alphas = parse_float_list(args.neg_alphas)
    stream_seeds = parse_int_list(args.stream_seeds)

    rows = []
    best = None
    for alpha, beta, neg_alpha, stream_seed in itertools.product(alphas, betas, neg_alphas, stream_seeds):
        params = dict(base)
        params["alpha"] = float(alpha)
        params["beta"] = float(beta)
        params["neg_alpha"] = float(neg_alpha)
        acc = float(
            evaluate_loaded(
                        payload=payload,
                        device=device,
                        shuffle_stream=bool(args.shuffle_stream),
                        stream_seed=stream_seed,
                        **params,
                    )
                )
        row = {
            "dataset": dataset_key,
            "alpha": float(alpha),
            "beta": float(beta),
            "neg_alpha": float(neg_alpha),
            "stream_seed": int(stream_seed),
            "accuracy": acc,
        }
        rows.append(row)
        if best is None or acc > best["accuracy"]:
            best = row
        print(json.dumps(row), flush=True)

    result = {"dataset": dataset_key, "base": base, "best": best, "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
