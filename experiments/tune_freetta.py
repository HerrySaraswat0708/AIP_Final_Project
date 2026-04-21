from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_freetta import evaluate_loaded, load_freetta_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def best_by_accuracy(rows):
    return max(rows, key=lambda item: float(item["accuracy"]))


datasets = ["DTD", "caltech", "eurosat", "pets", "imagenet"]

# FreeTTA is sensitive to alpha; beta controls entropy-based update weight.
alpha_list = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
beta_list = [0.1, 0.5, 1.0, 2.0, 4.5]

# Paper default: a shared alpha/beta pair is used across datasets.
recommended_defaults = {
    "dtd": {"alpha": 0.2, "beta": 4.5},
    "caltech": {"alpha": 0.2, "beta": 4.5},
    "eurosat": {"alpha": 0.2, "beta": 4.5},
    "pets": {"alpha": 0.2, "beta": 4.5},
    "imagenet": {"alpha": 0.2, "beta": 4.5},
}


def _parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _default_datasets() -> list[str]:
    return list(datasets)


def _format_minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 1.0:
        return f"{seconds:.1f}s"
    if minutes < 120.0:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def _estimate_runtime(payload, alpha: float, beta: float, device) -> tuple[float, float]:
    probe_samples = min(int(payload["num_samples"]), 64)
    probe_payload = {
        "image_features": payload["image_features"][:probe_samples],
        "text_features": payload["text_features"],
        "labels": payload["labels"][:probe_samples],
        "num_samples": probe_samples,
    }
    t0 = time.perf_counter()
    evaluate_loaded(
        payload=probe_payload,
        alpha=alpha,
        beta=beta,
        device=device,
        shuffle_stream=True,
        stream_seed=1,
    )
    elapsed = time.perf_counter() - t0
    sec_per_sample = elapsed / max(probe_samples, 1)
    estimated_seconds = sec_per_sample * float(payload["num_samples"])
    return sec_per_sample, estimated_seconds


def main() -> None:
    selected_datasets = _parse_str_list(os.getenv("AIP_TUNE_DATASETS"), _default_datasets())
    selected_alpha_list = _parse_float_list(os.getenv("AIP_TUNE_ALPHA_LIST"), alpha_list)
    selected_beta_list = _parse_float_list(os.getenv("AIP_TUNE_BETA_LIST"), beta_list)
    max_samples_env = os.getenv("AIP_TUNE_MAX_SAMPLES")
    max_samples = int(max_samples_env) if max_samples_env else None
    trial_count = len(selected_alpha_list) * len(selected_beta_list)

    print(
        f"[Startup] device={device} datasets={selected_datasets} "
        f"trials_per_dataset={trial_count} max_samples={max_samples}",
        flush=True,
    )

    best_per_dataset = {}
    all_rows = []

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for dataset in selected_datasets:
        key = dataset.lower()
        print(f"[Loading] dataset={dataset}", flush=True)
        payload = load_freetta_dataset(
            dataset=dataset,
            device=device,
            max_samples=max_samples,
            features_dir="data/processed",
        )

        print(f"[Loaded] {dataset} samples={payload['num_samples']} device={device}", flush=True)
        sec_per_sample, estimated_single_pass = _estimate_runtime(
            payload=payload,
            alpha=float(selected_alpha_list[0]),
            beta=float(selected_beta_list[0]),
            device=device,
        )
        print(
            f"[Estimate] {dataset}: ~{sec_per_sample*1000.0:.3f} ms/sample, "
            f"~{_format_minutes(estimated_single_pass)} per trial, "
            f"~{_format_minutes(estimated_single_pass * trial_count)} for {trial_count} trials",
            flush=True,
        )

        dataset_rows = []
        for alpha in selected_alpha_list:
            for beta in selected_beta_list:
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
                    "dataset": dataset,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "accuracy": acc,
                }
                dataset_rows.append(row)
                all_rows.append(row)
                print(f"{dataset:>8} alpha={alpha:.4f} beta={beta:.2f} acc={acc:.6f}", flush=True)

        best_row = best_by_accuracy(dataset_rows)
        best_per_dataset[key] = best_row
        print(f"[Best/{dataset}] {best_row}", flush=True)

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
                f"beta={cfg['beta']:.2f} acc={rec_acc:.6f}",
                flush=True,
            )

        del payload
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print("\n--- Tuning Complete ---", flush=True)
    for ds, results in best_per_dataset.items():
        print(
            f"Best for {ds}: {results['accuracy']:.4f} "
            f"(Alpha: {results['alpha']}, Beta: {results['beta']})",
            flush=True,
        )

    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)

    payload = {
        "search_space": {"alpha_list": selected_alpha_list, "beta_list": selected_beta_list},
        "recommended_defaults": recommended_defaults,
        "best_per_dataset": best_per_dataset,
        "all_rows": all_rows,
    }
    with open(output_path / "tta_results.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    print(f"\n[Saved] Results to {output_path / 'tta_results.json'}", flush=True)


if __name__ == "__main__":
    main()
