from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from models.FreeTTA import FreeTTA
from src.feature_store import load_dataset_features


def load_freetta_dataset(
    dataset: str,
    device: torch.device,
    max_samples: int | None = None,
    features_dir: str | Path = "data/processed",
) -> Dict[str, torch.Tensor | int]:
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
        "image_features": image_features,
        "text_features": text_features,
        "labels": labels,
        "num_samples": int(labels.shape[0]),
    }


def evaluate_loaded(
    payload: Dict[str, torch.Tensor | int],
    alpha: float = 0.008,
    beta: float = 1.0,
    device: torch.device | str = "cuda",
    shuffle_stream: bool = True,
    stream_seed: int = 1,
) -> float:
    if isinstance(device, str):
        device = torch.device(device)

    image_features = payload["image_features"]
    text_features = payload["text_features"]
    labels = payload["labels"]
    total = int(payload["num_samples"])

    model = FreeTTA(text_features=text_features, alpha=alpha, beta=beta, device=device)

    clip_correct = torch.tensor(0, device=device)
    freetta_correct = torch.tensor(0, device=device)

    if shuffle_stream:
        generator = torch.Generator()
        generator.manual_seed(int(stream_seed))
        order = torch.randperm(total, generator=generator).to(labels.device)
    else:
        order = torch.arange(total, device=labels.device)

    with torch.inference_mode():
        for i in order.tolist():
            x = image_features[i]
            y = labels[i]

            # Eq. (13) in FreeTTA uses cosine-similarity logits (no extra x100 scaling).
            clip_logits = x @ text_features.t()
            clip_pred = torch.argmax(clip_logits, dim=-1)
            clip_correct += (clip_pred == y)

            pred, _ = model.predict(x, clip_logits)
            freetta_correct += (pred.squeeze(0) == y)

    clip_acc = float(clip_correct.item() / max(total, 1))
    freetta_acc = float(freetta_correct.item() / max(total, 1))
    print(f"[CLIP] {clip_acc:.6f}")
    print(f"[FreeTTA] {freetta_acc:.6f}")
    return freetta_acc


def evaluate(
    dataset: str,
    alpha: float = 0.008,
    beta: float = 1.0,
    device: torch.device | str = "cuda",
    max_samples: int | None = None,
    features_dir: str | Path = "data/processed",
    shuffle_stream: bool = True,
    stream_seed: int = 1,
) -> float:
    if isinstance(device, str):
        device = torch.device(device)
    payload = load_freetta_dataset(
        dataset=dataset,
        device=device,
        max_samples=max_samples,
        features_dir=features_dir,
    )
    return evaluate_loaded(
        payload=payload,
        alpha=alpha,
        beta=beta,
        device=device,
        shuffle_stream=shuffle_stream,
        stream_seed=stream_seed,
    )
