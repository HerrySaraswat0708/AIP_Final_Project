from pathlib import Path

import torch
import torch.nn.functional as F

from models.EdgeFreeTTA import EdgeFreeTTA
from src.feature_store import load_dataset_features


def load_edgefreetta_dataset(dataset, device, max_samples=None, features_dir="data/processed"):
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
    rank=8,
    fusion_alpha=0.5,
    learning_rate=1e-2,
    beta=4.5,
    min_confidence=0.65,
    align_weight=0.5,
    residual_weight=0.05,
    weight_decay=1e-4,
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

    model = EdgeFreeTTA(
        text_features=text_features,
        rank=rank,
        fusion_alpha=fusion_alpha,
        learning_rate=learning_rate,
        beta=beta,
        min_confidence=min_confidence,
        align_weight=align_weight,
        residual_weight=residual_weight,
        weight_decay=weight_decay,
        device=device,
    )

    clip_correct = torch.tensor(0, device=device)
    edge_correct = torch.tensor(0, device=device)

    if shuffle_stream:
        generator = torch.Generator()
        generator.manual_seed(int(stream_seed))
        order = torch.randperm(total, generator=generator).to(labels.device)
    else:
        order = torch.arange(total, device=labels.device)

    for idx in order:
        x = image_features[idx]
        y = labels[idx]

        clip_logits = x @ text_features.t()
        clip_pred = torch.argmax(clip_logits, dim=-1)
        clip_correct += (clip_pred == y).to(clip_correct.dtype)

        pred, _, _ = model.predict_and_adapt(x, clip_logits)
        edge_correct += (pred.squeeze(0) == y).to(edge_correct.dtype)

    clip_acc = float(clip_correct.item() / max(total, 1))
    edge_acc = float(edge_correct.item() / max(total, 1))
    print(f"[CLIP] {clip_acc:.6f}")
    print(f"[EdgeFreeTTA] {edge_acc:.6f}")
    return edge_acc


def evaluate(
    dataset,
    rank=8,
    fusion_alpha=0.5,
    learning_rate=1e-2,
    beta=4.5,
    min_confidence=0.65,
    align_weight=0.5,
    residual_weight=0.05,
    weight_decay=1e-4,
    device="cuda",
    max_samples=None,
    features_dir="data/processed",
    shuffle_stream=True,
    stream_seed=1,
):
    if isinstance(device, str):
        device = torch.device(device)
    payload = load_edgefreetta_dataset(
        dataset=dataset,
        device=device,
        max_samples=max_samples,
        features_dir=features_dir,
    )
    return evaluate_loaded(
        payload=payload,
        rank=rank,
        fusion_alpha=fusion_alpha,
        learning_rate=learning_rate,
        beta=beta,
        min_confidence=min_confidence,
        align_weight=align_weight,
        residual_weight=residual_weight,
        weight_decay=weight_decay,
        device=device,
        shuffle_stream=shuffle_stream,
        stream_seed=stream_seed,
    )
