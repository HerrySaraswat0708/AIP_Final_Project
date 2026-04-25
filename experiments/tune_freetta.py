from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.FreeTTA import FreeTTA
from src.feature_store import load_dataset_features


DEFAULT_DATASETS = ("dtd", "caltech", "eurosat", "pets", "imagenet")
# Focused search over the 2 meaningful hyperparams after the formula fix.
# Legacy params (clip_scale, entropy_scale, normalize_mu, etc.) are accepted but ignored.
DEFAULT_ALPHA_LIST = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
DEFAULT_BETA_LIST  = (1.0, 2.0, 3.0, 3.5, 4.0, 5.0)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    return [item.strip().lower() in {"1", "true", "yes", "y"} for item in raw.split(",") if item.strip()]


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_payload(dataset: str, device: torch.device, features_dir: str, max_samples: int | None) -> dict:
    payload = load_dataset_features(Path(features_dir), dataset)
    image_features = torch.from_numpy(payload["image_features"]).float().to(device)
    text_features = torch.from_numpy(payload["text_features"]).float().to(device)
    labels = torch.from_numpy(payload["labels"]).long().to(device)

    if max_samples is not None and max_samples > 0:
        image_features = image_features[:max_samples]
        labels = labels[:max_samples]

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return {
        "dataset": str(payload["dataset_key"]).lower(),
        "image_features": image_features,
        "text_features": text_features,
        "labels": labels,
        "num_samples": int(labels.shape[0]),
    }


def evaluate_loaded(
    payload: dict,
    *,
    alpha: float,
    beta: float,
    clip_scale: float,
    entropy_scale: float,
    normalize_mu: bool,
    use_paper_cov_update: bool,
    use_reg_inverse: bool,
    normalize_entropy: bool,
    use_fused_posterior: bool,
    device: torch.device,
    shuffle_stream: bool,
    stream_seed: int,
) -> tuple[float, float]:
    model = FreeTTA(
        text_features=payload["text_features"],
        alpha=alpha,
        beta=beta,
        clip_scale=clip_scale,
        entropy_scale=entropy_scale,
        normalize_mu=normalize_mu,
        use_paper_cov_update=use_paper_cov_update,
        use_reg_inverse=use_reg_inverse,
        normalize_entropy=normalize_entropy,
        use_fused_posterior=use_fused_posterior,
        device=device,
    )

    image_features = payload["image_features"]
    text_features = payload["text_features"]
    labels = payload["labels"]
    total = int(payload["num_samples"])

    clip_correct = torch.tensor(0, device=device)
    freetta_correct = torch.tensor(0, device=device)

    if shuffle_stream:
        generator = torch.Generator()
        generator.manual_seed(int(stream_seed))
        order = torch.randperm(total, generator=generator).to(labels.device)
    else:
        order = torch.arange(total, device=labels.device)

    with torch.inference_mode():
        for idx in order:
            x = image_features[idx]
            y = labels[idx]

            clip_logits = x @ text_features.t()
            clip_pred = torch.argmax(clip_logits, dim=-1)
            clip_correct += (clip_pred == y).to(clip_correct.dtype)

            pred, _ = model.predict(x, clip_logits)
            freetta_correct += (pred.squeeze(0) == y).to(freetta_correct.dtype)

    clip_acc = float(clip_correct.item() / max(total, 1))
    freetta_acc = float(freetta_correct.item() / max(total, 1))
    return clip_acc, freetta_acc


def best_by_accuracy(rows: list[dict]) -> dict:
    return max(rows, key=lambda item: float(item["accuracy"]))


def estimate_runtime(payload: dict, trial_kwargs: dict, device: torch.device) -> tuple[float, float]:
    probe_samples = min(int(payload["num_samples"]), 64)
    probe_payload = {
        "dataset": payload["dataset"],
        "image_features": payload["image_features"][:probe_samples],
        "text_features": payload["text_features"],
        "labels": payload["labels"][:probe_samples],
        "num_samples": probe_samples,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    evaluate_loaded(probe_payload, device=device, **trial_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    sec_per_sample = elapsed / max(probe_samples, 1)
    return sec_per_sample, sec_per_sample * float(payload["num_samples"])


def format_minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 1.0:
        return f"{seconds:.1f}s"
    if minutes < 120.0:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune FreeTTA on frozen CLIP features")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHA_LIST))
    parser.add_argument("--betas", default=",".join(str(x) for x in DEFAULT_BETA_LIST))
    # Legacy params – kept for CLI compatibility but ignored by FreeTTA
    parser.add_argument("--clip-scales", default="100.0")
    parser.add_argument("--entropy-scales", default="100.0")
    parser.add_argument("--normalize-mu", default="true")
    parser.add_argument("--paper-cov", default="false")
    parser.add_argument("--reg-inverse", default="false")
    parser.add_argument("--normalize-entropy", default="false")
    parser.add_argument("--fused-posterior", default="false")
    parser.add_argument("--shuffle-stream", action="store_true")
    parser.add_argument("--stream-seed", type=int, default=1)
    parser.add_argument("--output", default="outputs/tuning/best_freetta_run_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    clip_scales = parse_float_list(args.clip_scales)
    entropy_scales = parse_float_list(args.entropy_scales)
    normalize_mu_list = parse_bool_list(args.normalize_mu)
    paper_cov_list = parse_bool_list(args.paper_cov)
    reg_inverse_list = parse_bool_list(args.reg_inverse)
    normalize_entropy_list = parse_bool_list(args.normalize_entropy)
    fused_posterior_list = parse_bool_list(args.fused_posterior)

    total_trials = (
        len(alphas)
        * len(betas)
        * len(clip_scales)
        * len(entropy_scales)
        * len(normalize_mu_list)
        * len(paper_cov_list)
        * len(reg_inverse_list)
        * len(normalize_entropy_list)
        * len(fused_posterior_list)
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(
        f"[Startup] device={device} datasets={datasets} "
        f"trials_per_dataset={total_trials} max_samples={args.max_samples} "
        f"shuffle_stream={args.shuffle_stream} seed={args.stream_seed}",
        flush=True,
    )

    all_rows: list[dict] = []
    best_per_dataset: dict[str, dict] = {}

    for dataset in datasets:
        payload = load_payload(dataset, device=device, features_dir=args.features_dir, max_samples=args.max_samples)
        print(f"[Loaded] {payload['dataset']} samples={payload['num_samples']}", flush=True)

        probe_kwargs = {
            "alpha": float(alphas[0]),
            "beta": float(betas[0]),
            "clip_scale": float(clip_scales[0]),
            "entropy_scale": float(entropy_scales[0]),
            "normalize_mu": bool(normalize_mu_list[0]),
            "use_paper_cov_update": bool(paper_cov_list[0]),
            "use_reg_inverse": bool(reg_inverse_list[0]),
            "normalize_entropy": bool(normalize_entropy_list[0]),
            "use_fused_posterior": bool(fused_posterior_list[0]),
            "shuffle_stream": bool(args.shuffle_stream),
            "stream_seed": int(args.stream_seed),
        }
        sec_per_sample, est = estimate_runtime(payload, probe_kwargs, device)
        print(
            f"[Estimate] {payload['dataset']}: ~{sec_per_sample*1000.0:.3f} ms/sample, "
            f"~{format_minutes(est)} per trial, "
            f"~{format_minutes(est * total_trials)} total",
            flush=True,
        )

        dataset_rows: list[dict] = []
        for alpha in alphas:
            for beta in betas:
                for clip_scale in clip_scales:
                    for entropy_scale in entropy_scales:
                        for normalize_mu in normalize_mu_list:
                            for use_paper_cov_update in paper_cov_list:
                                for use_reg_inverse in reg_inverse_list:
                                    for normalize_entropy in normalize_entropy_list:
                                        for use_fused_posterior in fused_posterior_list:
                                            clip_acc, acc = evaluate_loaded(
                                                payload,
                                                alpha=float(alpha),
                                                beta=float(beta),
                                                clip_scale=float(clip_scale),
                                                entropy_scale=float(entropy_scale),
                                                normalize_mu=bool(normalize_mu),
                                                use_paper_cov_update=bool(use_paper_cov_update),
                                                use_reg_inverse=bool(use_reg_inverse),
                                                normalize_entropy=bool(normalize_entropy),
                                                use_fused_posterior=bool(use_fused_posterior),
                                                device=device,
                                                shuffle_stream=bool(args.shuffle_stream),
                                                stream_seed=int(args.stream_seed),
                                            )
                                            row = {
                                                "dataset": payload["dataset"],
                                                "alpha": float(alpha),
                                                "beta": float(beta),
                                                "clip_scale": float(clip_scale),
                                                "entropy_scale": float(entropy_scale),
                                                "normalize_mu": bool(normalize_mu),
                                                "use_paper_cov_update": bool(use_paper_cov_update),
                                                "use_reg_inverse": bool(use_reg_inverse),
                                                "normalize_entropy": bool(normalize_entropy),
                                                "use_fused_posterior": bool(use_fused_posterior),
                                                "shuffle_stream": bool(args.shuffle_stream),
                                                "stream_seed": int(args.stream_seed),
                                                "clip_accuracy": float(clip_acc),
                                                "accuracy": float(acc),
                                            }
                                            dataset_rows.append(row)
                                            all_rows.append(row)
                                            print(json.dumps(row), flush=True)

        best = best_by_accuracy(dataset_rows)
        best_per_dataset[payload["dataset"]] = dict(best)
        print(f"[Best/{payload['dataset']}] {best}", flush=True)

        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "device": str(device),
        "features_dir": args.features_dir,
        "max_samples": args.max_samples,
        "search_space": {
            "alphas": alphas,
            "betas": betas,
            "clip_scales": clip_scales,
            "entropy_scales": entropy_scales,
            "normalize_mu": normalize_mu_list,
            "paper_cov": paper_cov_list,
            "reg_inverse": reg_inverse_list,
            "normalize_entropy": normalize_entropy_list,
            "fused_posterior": fused_posterior_list,
        },
        "shuffle_stream": bool(args.shuffle_stream),
        "stream_seed": int(args.stream_seed),
        "best_per_dataset": best_per_dataset,
        "all_rows": all_rows,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[Saved] {output_path}", flush=True)


if __name__ == "__main__":
    main()
