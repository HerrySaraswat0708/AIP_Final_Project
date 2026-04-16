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
from src.paper_configs import PAPER_TDA_DEFAULTS


DEFAULT_DATASETS = list(PAPER_TDA_DEFAULTS.keys())
PARAMS_JSON_CANDIDATES = [
    Path("outputs/tuning/tda_target_matched_params.json"),
    Path("outputs/tuning/best_tda_params.json"),
    Path("outputs/tuning/best_tda_run_results.json"),
]
OUTPUT_JSON = Path("outputs/tuning/best_tda_run_results.json")


def resolve_params_json() -> Path | None:
    for candidate in PARAMS_JSON_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_best_configs(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    if isinstance(params, dict) and params:
        return {str(dataset).lower(): dict(cfg) for dataset, cfg in params.items()}

    best_per_dataset = payload.get("best_per_dataset")
    if isinstance(best_per_dataset, dict) and best_per_dataset:
        configs = {}
        for dataset, row in best_per_dataset.items():
            if not isinstance(row, dict):
                continue
            cfg = {
                key: value
                for key, value in row.items()
                if key
                in {
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
                    "pos_shot_capacity",
                    "neg_shot_capacity",
                    "clip_scale",
                    "fallback_to_clip",
                    "fallback_margin",
                }
            }
            if cfg:
                configs[str(dataset).lower()] = cfg
        if configs:
            return configs

    results = payload.get("results")
    if isinstance(results, list) and results:
        configs = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            dataset = row.get("dataset")
            cfg = row.get("params")
            if dataset is not None and isinstance(cfg, dict) and cfg:
                configs[str(dataset).lower()] = dict(cfg)
        if configs:
            return configs

    raise ValueError(f"No valid TDA config mapping found in {path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_configs = {str(dataset).lower(): dict(cfg) for dataset, cfg in PAPER_TDA_DEFAULTS.items()}
    use_tuned = os.getenv("AIP_USE_TUNED_TDA", "").strip().lower() in {"1", "true", "yes"}
    params_path = resolve_params_json() if use_tuned else None
    if params_path is not None:
        best_configs.update(load_best_configs(params_path))
        print(f"[Config] Using tuned params from {params_path}")
    else:
        print("[Config] Using checked-in paper defaults.")
    rows = []  # type: List[Dict]

    for dataset in DEFAULT_DATASETS:
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
