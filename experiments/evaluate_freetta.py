from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from models.FreeTTA import FreeTTA
from src.feature_store import load_dataset_features


INFERENCE_MODE = getattr(torch, "inference_mode", torch.no_grad)


def load_freetta_dataset(
    dataset,
    device,
    max_samples=None,
    features_dir="data/processed",
):
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
    payload,
    alpha=0.008,
    beta=1.0,
    device="cuda",
    shuffle_stream=True,
    stream_seed=1,
):
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

    with INFERENCE_MODE():
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
    dataset,
    alpha=0.008,
    beta=1.0,
    device="cuda",
    max_samples=None,
    features_dir="data/processed",
    shuffle_stream=True,
    stream_seed=1,
):
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
