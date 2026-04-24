import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.evaluate_freetta import evaluate_loaded, load_freetta_dataset
from src.paper_configs import DEFAULT_FREETTA_PARAMS
from src.paper_setup import EXPECTED_TEST_SPLIT_SIZES


BEST_CONFIGS = {key: dict(value) for key, value in DEFAULT_FREETTA_PARAMS.items()}


def _validate_payload(dataset: str, payload) -> None:
    expected = EXPECTED_TEST_SPLIT_SIZES.get(str(dataset).lower())
    actual = int(payload["num_samples"])
    if expected is not None and actual != expected:
        raise ValueError(
            f"Dataset '{dataset}' has {actual} samples in data/processed, expected {expected} for the official split."
        )


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

        _validate_payload(dataset, payload)

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
            shuffle_stream=False,
        )

        rows[dataset] = {
            "alpha": float(cfg["alpha"]),
            "beta": float(cfg["beta"]),
            "freetta_accuracy": float(acc),
        }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "/tuning/best_freetta_run_results.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
