from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.evaluate_freetta import evaluate_loaded, load_freetta_dataset
from src.feature_store import list_available_datasets
from src.paper_configs import DEFAULT_DATASETS, DEFAULT_FREETTA_PARAMS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_by_accuracy(rows):
    return max(rows, key=lambda item: float(item["accuracy"]))


features_dir = Path("data/processed")
available = set(list_available_datasets(features_dir))
datasets = [dataset for dataset in DEFAULT_DATASETS if dataset in available]

# FreeTTA is sensitive to alpha; beta controls entropy-based update weight.
alpha_list = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
beta_list = [0.1, 0.5, 1.0, 2.0, 4.5]

# Strong defaults from corrected online-EM runs in this repo.
recommended_defaults = DEFAULT_FREETTA_PARAMS


best_per_dataset = {}
all_rows = []

use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

for dataset in datasets:
    key = dataset.lower()
    payload = load_freetta_dataset(
        dataset=dataset,
        device=device,
        max_samples=None,
        features_dir=str(features_dir),
    )

    print(f"[Loaded] {dataset} samples={payload['num_samples']} device={device}")

    dataset_rows = []
    for alpha in alpha_list:
        for beta in beta_list:
            acc = float(
                evaluate_loaded(
                    payload=payload,
                    alpha=alpha,
                    beta=beta,
                    device=device,
                    shuffle_stream=True,
                    stream_seed=1,
                )
            )
            row = {
                "dataset": key,
                "alpha": float(alpha),
                "beta": float(beta),
                "accuracy": acc,
            }
            dataset_rows.append(row)
            all_rows.append(row)
            print(f"{dataset:>8} alpha={alpha:.4f} beta={beta:.2f} acc={acc:.6f}")

    best_row = best_by_accuracy(dataset_rows)
    best_row["dataset"] = key
    best_per_dataset[key] = best_row
    print(f"[Best/{dataset}] {best_row}")

    # Also print recommended default behavior for quick reproducibility.
    if key in recommended_defaults:
        cfg = recommended_defaults[key]
        rec_acc = float(
            evaluate_loaded(
                payload=payload,
                alpha=float(cfg["alpha"]),
                beta=float(cfg["beta"]),
                device=device,
                shuffle_stream=True,
                stream_seed=1,
            )
        )
        print(
            f"[Recommended/{dataset}] alpha={cfg['alpha']:.4f} "
            f"beta={cfg['beta']:.2f} acc={rec_acc:.6f}"
        )

    del payload
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print("\n--- Tuning Complete ---")
for ds, results in best_per_dataset.items():
    print(
        f"Best for {ds}: {results['accuracy']:.4f} "
        f"(Alpha: {results['alpha']}, Beta: {results['beta']})"
    )

output_path = Path("outputs")
output_path.mkdir(exist_ok=True)

payload = {
    "search_space": {"alpha_list": alpha_list, "beta_list": beta_list},
    "recommended_defaults": recommended_defaults,
    "best_per_dataset": best_per_dataset,
    "all_rows": all_rows,
}
with open(output_path / "tta_results.json", "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=4)

print(f"\n[Saved] Results to {output_path / 'tta_results.json'}")
