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
from src.feature_store import list_available_datasets
from src.paper_configs import DEFAULT_DATASETS, DEFAULT_FREETTA_PARAMS


BEST_CONFIGS = DEFAULT_FREETTA_PARAMS


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    available = set(list_available_datasets(Path("data/processed")))

    for dataset in DEFAULT_DATASETS:
        if dataset not in available:
            print(f"[Skip] Missing features for dataset={dataset}")
            continue

        cfg = BEST_CONFIGS.get(dataset)
        if cfg is None:
            print(f"[Skip] No FreeTTA config found for dataset={dataset}")
            continue
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
