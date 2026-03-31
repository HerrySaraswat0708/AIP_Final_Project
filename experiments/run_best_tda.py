import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from experiments.evaluate_tda import evaluate_clip_loaded, evaluate_loaded, load_tda_dataset


DEFAULT_DATASETS = ["dtd", "caltech", "eurosat", "pets"]
PARAMS_JSON = Path("outputs/tda_target_matched_params.json")
OUTPUT_JSON = Path("outputs/best_tda_run_results.json")


def load_best_configs(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No valid `params` mapping found in {path}")
    return {str(dataset).lower(): dict(cfg) for dataset, cfg in params.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_configs = load_best_configs(PARAMS_JSON)
    rows = []  # type: List[Dict]

    for dataset in DEFAULT_DATASETS:
        if dataset not in best_configs:
            print(f"[Skip] No TDA config found for dataset={dataset}")
            continue

        cfg = best_configs[dataset]
        payload = load_tda_dataset(
            dataset=dataset,
            device=device,
            max_samples=None,
            features_dir="data/processed",
        )

        print(
            f"\n[Run] {dataset} "
            f"cache={cfg['cache_size']} alpha={cfg['alpha']:.4f} beta={cfg['beta']:.2f} "
            f"samples={payload['num_samples']}"
        )

        clip_acc = evaluate_clip_loaded(payload)
        tda_acc = evaluate_loaded(
            payload=payload,
            device=device,
            **cfg,
        )

        rows.append(
            {
                "dataset": dataset,
                "samples": int(payload["num_samples"]),
                "clip_acc": float(clip_acc),
                "tda_acc": float(tda_acc),
                "gain_vs_clip": float(tda_acc - clip_acc),
                "params": cfg,
            }
        )

        print(
            f"[Result] {dataset} "
            f"CLIP={clip_acc * 100:.2f}% "
            f"TDA={tda_acc * 100:.2f}% "
            f"gain={((tda_acc - clip_acc) * 100):+.2f} pts"
        )

    report = {
        "device": str(device),
        "features_dir": "data/processed",
        "results": rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Saved] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
