from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.evaluate_freetta import evaluate_loaded, load_freetta_dataset


BEST_CONFIGS = {
    "dtd": {"alpha": 0.2, "beta": 2.0},
    "caltech": {"alpha": 0.1, "beta": 1.0},
    "eurosat": {"alpha": 0.3, "beta": 4.5},
    "pets": {"alpha": 0.1, "beta": 0.1},
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}

    for dataset, cfg in BEST_CONFIGS.items():
        payload = load_freetta_dataset(
            dataset=dataset,
            device=device,
            max_samples=None,
            features_dir="data/processed",
        )

        print(
            f"\n[Run] {dataset} "
            f"alpha={cfg['alpha']:.4f} beta={cfg['beta']:.2f} "
            f"samples={payload['num_samples']}"
        )
        acc = evaluate_loaded(
            payload=payload,
            alpha=float(cfg["alpha"]),
            beta=float(cfg["beta"]),
            device=device,
            shuffle_stream=True,
            stream_seed=1,
        )

        rows[dataset] = {
            "alpha": float(cfg["alpha"]),
            "beta": float(cfg["beta"]),
            "freetta_accuracy": float(acc),
        }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "freetta_best_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
