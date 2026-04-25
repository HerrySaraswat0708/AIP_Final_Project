import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TDA import TDA
from src.feature_store import load_dataset_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAPER_DEFAULTS = {
    "dtd": {"alpha": 2.0, "beta": 3.0},
    "caltech": {"alpha": 5.0, "beta": 5.0},
    "eurosat": {"alpha": 4.0, "beta": 8.0},
    "pets": {"alpha": 2.0, "beta": 7.0},
    "imagenet": {"alpha": 1.0, "beta": 8.0},
}


def best_by_accuracy(rows):
    return max(rows, key=lambda item: float(item["accuracy"]))


def canonical(ds: str) -> str:
    return ds.strip().lower()


# ----------------------------
# SEARCH SPACE (paper-aligned)
# ----------------------------
datasets = ["imagenet", "dtd", "caltech", "eurosat", "pets"]
cache_size_list = [1000]  # retained for backward compatibility
shot_capacity_list = [3]
k_list = [0]
alpha_scales = [2.0, 0.75, 1.5, 6.0]
beta_scales = [3.0, 1.5, 9]
low_entropy_thresh_list = [0.2]
high_entropy_thresh_list = [0.5]
neg_alpha_list = [0.0, 0.05, 0.3]
neg_beta_list = [1.0]
neg_mask_lower_list = [0.03]
neg_mask_upper_list = [1.0]
clip_scale_list = [100.0]
fallback_to_clip = True
fallback_margin = 0.0
max_samples = None

use_cuda = device.type == "cuda"
if use_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def load_tda_dataset(dataset, device, max_samples=None, features_dir="data/processed"):
    raw = load_dataset_features(Path(features_dir), dataset)
    image_features = raw["image_features"]
    text_features = raw["text_features"]
    labels = raw["labels"]

    if max_samples is not None and max_samples > 0:
        image_features = image_features[:max_samples]
        labels = labels[:max_samples]

    return {
        "dataset": raw["dataset_key"],
        "image_features": torch.from_numpy(image_features).float().to(device),
        "text_features": torch.from_numpy(text_features).float().to(device),
        "labels": torch.from_numpy(labels).long().to(device),
        "num_samples": int(len(labels)),
    }


def evaluate_loaded(
    payload,
    cache_size,
    k,
    alpha,
    beta,
    low_entropy_thresh,
    high_entropy_thresh,
    device,
    neg_alpha=0.117,
    neg_beta=1.0,
    neg_mask_lower=0.03,
    neg_mask_upper=1.0,
    shot_capacity=3,
    pos_shot_capacity=None,
    neg_shot_capacity=None,
    clip_scale=100.0,
    fallback_to_clip=False,
    fallback_margin=0.0,
    shuffle_stream=True,
    stream_seed=1,
):
    image_features = payload["image_features"]
    labels = payload["labels"]
    text_features = payload["text_features"]
    total = int(payload["num_samples"])

    model = TDA(
        text_features=text_features,
        cache_size=cache_size,
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
        pos_shot_capacity=pos_shot_capacity,
        neg_shot_capacity=neg_shot_capacity,
        clip_scale=clip_scale,
        fallback_to_clip=fallback_to_clip,
        fallback_margin=fallback_margin,
        device=device,
    )

    correct_count = torch.tensor(0, device=device)
    if shuffle_stream:
        generator = torch.Generator()
        generator.manual_seed(int(stream_seed))
        order = torch.randperm(total, generator=generator).to(labels.device)
    else:
        order = torch.arange(total, device=labels.device)

    with torch.inference_mode():
        for idx in order:
            pred, _, _ = model.predict(image_features[idx])
            correct_count += (pred.squeeze(0) == labels[idx]).to(correct_count.dtype)

    return float(correct_count.item() / max(total, 1))


def evaluate_clip_loaded(payload) -> float:
    image_features = F.normalize(payload["image_features"], dim=-1)
    text_features = F.normalize(payload["text_features"], dim=-1)
    labels = payload["labels"]

    logits = image_features @ text_features.t()
    pred = torch.argmax(logits, dim=-1)
    return float((pred == labels).float().mean().item())


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _count_trials(
    alpha_values: list[float],
    beta_values: list[float],
    neg_alpha_values: list[float],
) -> int:
    return (
        len(cache_size_list)
        * len(shot_capacity_list)
        * len(k_list)
        * len(alpha_values)
        * len(beta_values)
        * len(low_entropy_thresh_list)
        * len(high_entropy_thresh_list)
        * len(neg_alpha_values)
        * len(neg_beta_list)
        * len(neg_mask_lower_list)
        * len(neg_mask_upper_list)
        * len(clip_scale_list)
    )


def _format_minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 1.0:
        return f"{seconds:.1f}s"
    if minutes < 120.0:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def _estimate_runtime(
    payload,
    alpha: float,
    beta: float,
    neg_alpha: float,
    device,
) -> tuple[float, float]:
    probe_samples = min(int(payload["num_samples"]), 128)
    probe_payload = {
        "dataset": payload["dataset"],
        "image_features": payload["image_features"][:probe_samples],
        "text_features": payload["text_features"],
        "labels": payload["labels"][:probe_samples],
        "num_samples": probe_samples,
    }
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    evaluate_loaded(
        payload=probe_payload,
        cache_size=cache_size_list[0],
        k=k_list[0],
        alpha=alpha,
        beta=beta,
        low_entropy_thresh=low_entropy_thresh_list[0],
        high_entropy_thresh=high_entropy_thresh_list[0],
        neg_alpha=neg_alpha,
        neg_beta=neg_beta_list[0],
        neg_mask_lower=neg_mask_lower_list[0],
        neg_mask_upper=neg_mask_upper_list[0],
        shot_capacity=shot_capacity_list[0],
        clip_scale=clip_scale_list[0],
        fallback_to_clip=fallback_to_clip,
        fallback_margin=fallback_margin,
        device=device,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - t0
    sec_per_sample = elapsed / max(probe_samples, 1)
    estimated_seconds = sec_per_sample * float(payload["num_samples"])
    return sec_per_sample, estimated_seconds

def main() -> None:
    selected_datasets = _parse_str_list(os.getenv("AIP_TUNE_DATASETS"), _default_datasets())
    selected_alpha_scales = _parse_float_list(os.getenv("AIP_TUNE_ALPHA_SCALES"), alpha_scales)
    selected_beta_scales = _parse_float_list(os.getenv("AIP_TUNE_BETA_SCALES"), beta_scales)
    selected_neg_alpha_list = _parse_float_list(os.getenv("AIP_TUNE_NEG_ALPHA_LIST"), neg_alpha_list)
    max_samples_env = os.getenv("AIP_TUNE_MAX_SAMPLES")
    selected_max_samples = int(max_samples_env) if max_samples_env else max_samples
    trial_count = _count_trials(selected_alpha_scales, selected_beta_scales, selected_neg_alpha_list)

    print(
        f"[Startup] device={device} datasets={selected_datasets} "
        f"trials_per_dataset={trial_count} max_samples={selected_max_samples}",
        flush=True,
    )

    best_per_dataset = {}
    all_results = []

    start_all = time.perf_counter()
    for dataset_name in selected_datasets:
        print(f"[Loading] dataset={dataset_name}", flush=True)
        payload = load_tda_dataset(
            dataset=dataset_name,
            device=device,
            max_samples=selected_max_samples,
            features_dir="data/processed",
        )
        dataset_key = canonical(payload["dataset"])

        clip_acc = evaluate_clip_loaded(payload)
        print(
            f"\n[Loaded] {dataset_key} samples={payload['num_samples']} device={device} clip_acc={clip_acc:.6f}",
            flush=True,
        )
        sec_per_sample, estimated_single_pass = _estimate_runtime(
            payload=payload,
            alpha=float(PAPER_DEFAULTS.get(dataset_key, {"alpha": 2.0})["alpha"] * selected_alpha_scales[0]),
            beta=float(PAPER_DEFAULTS.get(dataset_key, {"beta": 5.0})["beta"] * selected_beta_scales[0]),
            neg_alpha=float(selected_neg_alpha_list[0]),
            device=device,
        )
        print(
            f"[Estimate] {dataset_key}: ~{sec_per_sample*1000.0:.3f} ms/sample, "
            f"~{_format_minutes(estimated_single_pass)} per trial, "
            f"~{_format_minutes(estimated_single_pass * trial_count)} for {trial_count} trials",
            flush=True,
        )

        base = PAPER_DEFAULTS.get(dataset_key, {"alpha": 2.0, "beta": 5.0})

        dataset_rows = []
        for c_size in cache_size_list:
            for shot_capacity in shot_capacity_list:
                for k in k_list:
                    for a_scale in selected_alpha_scales:
                        alpha = float(base["alpha"] * a_scale)
                        for b_scale in selected_beta_scales:
                            beta = float(base["beta"] * b_scale)
                            for low_entropy_thresh in low_entropy_thresh_list:
                                for high_entropy_thresh in high_entropy_thresh_list:
                                    if low_entropy_thresh >= high_entropy_thresh:
                                        continue
                                    for neg_alpha in selected_neg_alpha_list:
                                        for neg_beta in neg_beta_list:
                                            for neg_mask_lower in neg_mask_lower_list:
                                                for neg_mask_upper in neg_mask_upper_list:
                                                    if neg_mask_lower >= neg_mask_upper:
                                                        continue
                                                    for clip_scale in clip_scale_list:
                                                        _sync_if_cuda(device)
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
                                                        _sync_if_cuda(device)
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
                                                            f"acc={acc:.6f} gain={acc - clip_acc:+.6f}",
                                                            flush=True,
                                                        )

        best = best_by_accuracy(dataset_rows)
        best_per_dataset[dataset_key] = best

        print(
            f"[Best] {dataset_key}: acc={best['accuracy']:.6f} "
            f"clip={best['clip_accuracy']:.6f} gain={best['gain_vs_clip']:+.6f} "
            f"alpha={best['alpha']:.3f} beta={best['beta']:.3f} shot={best['shot_capacity']}",
            flush=True,
        )

        del payload
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - start_all

    output_dir = Path("outputs") / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "best_tda_run_results.json"

    report = {
        "method": "TDA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "max_samples": selected_max_samples,
        "features_dir": "data/processed",
        "search_space": {
            "datasets": selected_datasets,
            "cache_size_list": cache_size_list,
            "shot_capacity_list": shot_capacity_list,
            "k_list": k_list,
            "alpha_scales": selected_alpha_scales,
            "beta_scales": selected_beta_scales,
            "low_entropy_thresh_list": low_entropy_thresh_list,
            "high_entropy_thresh_list": high_entropy_thresh_list,
            "neg_alpha_list": selected_neg_alpha_list,
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
    print(f"\n[Saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
