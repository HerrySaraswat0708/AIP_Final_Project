from pathlib import Path
import torch
import torch.nn.functional as F

from models.TDA import TDA
from src.feature_store import load_dataset_features


INFERENCE_MODE = getattr(torch, "inference_mode", torch.no_grad)


def load_tda_dataset(dataset, device, max_samples=None, features_dir="data/processed"):
    root = Path(features_dir)
    raw = load_dataset_features(root, dataset)

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
    clip_scale=100.0,
    fallback_to_clip=True,
    fallback_margin=0.0,
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
        clip_scale=clip_scale,
        fallback_to_clip=fallback_to_clip,
        fallback_margin=fallback_margin,
        device=device,
    )

    correct_count = torch.tensor(0, device=device)

    with INFERENCE_MODE():
        for i in range(total):
            pred, _, _ = model.predict(image_features[i])
            correct_count += int(pred.item() == labels[i].item())

    return float(correct_count.item() / max(total, 1))


def evaluate_clip_loaded(payload) -> float:
    image_features = F.normalize(payload["image_features"], dim=-1)
    text_features = F.normalize(payload["text_features"], dim=-1)
    labels = payload["labels"]

    logits = image_features @ text_features.t()
    pred = torch.argmax(logits, dim=-1)
    return float((pred == labels).float().mean().item())
