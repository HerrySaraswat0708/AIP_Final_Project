import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.evaluate_tda import evaluate_clip_loaded, evaluate_loaded, load_tda_dataset
from src.feature_store import list_available_datasets
from src.paper_configs import DEFAULT_DATASETS, PAPER_TDA_DEFAULTS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAPER_DEFAULTS = {
    key: {"alpha": float(value["alpha"]), "beta": float(value["beta"])}
    for key, value in PAPER_TDA_DEFAULTS.items()
}


def best_by_accuracy(rows):
    return max(rows, key=lambda item: float(item["accuracy"]))


def canonical(ds: str) -> str:
    return ds.strip().lower()


# ----------------------------
# SEARCH SPACE (paper-aligned)
# ----------------------------
features_dir = Path("data/processed")
available = set(list_available_datasets(features_dir))
datasets = [dataset for dataset in DEFAULT_DATASETS if dataset in available]
cache_size_list = [1000]  # retained for backward compatibility
shot_capacity_list = [3]
k_list = [0]
alpha_scales = [0.75, 1.0, 1.5, 2.0, 6.0]
beta_scales = [1.0, 1.5, 3.0, 9.0]
low_entropy_thresh_list = [0.2]
high_entropy_thresh_list = [0.5]
neg_alpha_list = [0.0, 0.05, 0.117, 0.3]
neg_beta_list = [1.0]
neg_mask_lower_list = [0.03]
neg_mask_upper_list = [1.0]
clip_scale_list = [100.0]
max_samples = None

use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

best_per_dataset = {}
all_results = []

start_all = time.perf_counter()
for dataset_name in datasets:
    payload = load_tda_dataset(
        dataset=dataset_name,
        device=device,
        max_samples=max_samples,
        features_dir=str(features_dir),
    )
    dataset_key = canonical(payload["dataset"])

    clip_acc = evaluate_clip_loaded(payload)
    print(f"\n[Loaded] {dataset_key} samples={payload['num_samples']} device={device} clip_acc={clip_acc:.6f}")

    base = PAPER_DEFAULTS.get(dataset_key, {"alpha": 2.0, "beta": 5.0})
    base_cfg = PAPER_TDA_DEFAULTS.get(dataset_key, PAPER_TDA_DEFAULTS["imagenet"])
    fallback_to_clip = bool(base_cfg.get("fallback_to_clip", True))
    fallback_margin = float(base_cfg.get("fallback_margin", 0.0))

    dataset_rows = []
    for c_size in cache_size_list:
        for shot_capacity in shot_capacity_list:
            for k in k_list:
                for a_scale in alpha_scales:
                    alpha = float(base["alpha"] * a_scale)
                    for b_scale in beta_scales:
                        beta = float(base["beta"] * b_scale)
                        for low_entropy_thresh in low_entropy_thresh_list:
                            for high_entropy_thresh in high_entropy_thresh_list:
                                if low_entropy_thresh >= high_entropy_thresh:
                                    continue
                                for neg_alpha in neg_alpha_list:
                                    for neg_beta in neg_beta_list:
                                        for neg_mask_lower in neg_mask_lower_list:
                                            for neg_mask_upper in neg_mask_upper_list:
                                                if neg_mask_lower >= neg_mask_upper:
                                                    continue
                                                for clip_scale in clip_scale_list:
                                                    t0 = time.perf_counter()
                                                    acc = float(
                                                        evaluate_loaded(
                                                            payload=payload,
                                                            cache_size=c_size,
                                                            k=k,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            low_entropy_thresh=low_entropy_thresh,
                                                            high_entropy_thresh=high_entropy_thresh,
                                                            neg_alpha=neg_alpha,
                                                            neg_beta=neg_beta,
                                                            neg_mask_lower=neg_mask_lower,
                                                            neg_mask_upper=neg_mask_upper,
                                                            shot_capacity=shot_capacity,
                                                            clip_scale=clip_scale,
                                                            fallback_to_clip=fallback_to_clip,
                                                            fallback_margin=fallback_margin,
                                                            device=device,
                                                        )
                                                    )
                                                    elapsed = time.perf_counter() - t0

                                                    row = {
                                                        "dataset": dataset_key,
                                                        "cache_size": int(c_size),
                                                        "shot_capacity": int(shot_capacity),
                                                        "k": int(k),
                                                        "alpha": float(alpha),
                                                        "beta": float(beta),
                                                        "low_entropy_thresh": float(low_entropy_thresh),
                                                        "high_entropy_thresh": float(high_entropy_thresh),
                                                        "neg_alpha": float(neg_alpha),
                                                        "neg_beta": float(neg_beta),
                                                        "neg_mask_lower": float(neg_mask_lower),
                                                        "neg_mask_upper": float(neg_mask_upper),
                                                        "clip_scale": float(clip_scale),
                                                        "fallback_to_clip": bool(fallback_to_clip),
                                                        "fallback_margin": float(fallback_margin),
                                                        "accuracy": acc,
                                                        "clip_accuracy": float(clip_acc),
                                                        "gain_vs_clip": float(acc - clip_acc),
                                                        "elapsed_s": float(elapsed),
                                                    }
                                                    dataset_rows.append(row)
                                                    all_results.append(row)

                                                    print(
                                                        f"{dataset_key:>8} shot={shot_capacity} k={k} "
                                                        f"alpha={alpha:.3f} beta={beta:.3f} "
                                                        f"neg_alpha={neg_alpha:.3f} "
                                                        f"acc={acc:.6f} gain={acc - clip_acc:+.6f}"
                                                    )

    best = best_by_accuracy(dataset_rows)
    best_per_dataset[dataset_key] = best

    print(
        f"[Best] {dataset_key}: acc={best['accuracy']:.6f} "
        f"clip={best['clip_accuracy']:.6f} gain={best['gain_vs_clip']:+.6f} "
        f"alpha={best['alpha']:.3f} beta={best['beta']:.3f} shot={best['shot_capacity']}"
    )

    del payload
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


total_elapsed = time.perf_counter() - start_all

output_dir = Path("outputs/tuning")
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "best_tda_params.json"

report = {
    "method": "TDA",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "device": str(device),
    "max_samples": max_samples,
    "features_dir": str(features_dir),
    "search_space": {
        "datasets": datasets,
        "cache_size_list": cache_size_list,
        "shot_capacity_list": shot_capacity_list,
        "k_list": k_list,
        "alpha_scales": alpha_scales,
        "beta_scales": beta_scales,
        "low_entropy_thresh_list": low_entropy_thresh_list,
        "high_entropy_thresh_list": high_entropy_thresh_list,
        "neg_alpha_list": neg_alpha_list,
        "neg_beta_list": neg_beta_list,
        "neg_mask_lower_list": neg_mask_lower_list,
        "neg_mask_upper_list": neg_mask_upper_list,
        "clip_scale_list": clip_scale_list,
        "fallback_to_clip": fallback_to_clip,
        "fallback_margin": fallback_margin,
    },
    "best_overall": best_by_accuracy(list(best_per_dataset.values())),
    "best_per_dataset": best_per_dataset,
    "total_elapsed_s": float(total_elapsed),
    "all_results": all_results,
}

out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\n[Saved] {out_path}")
