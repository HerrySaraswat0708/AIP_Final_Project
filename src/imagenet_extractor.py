from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_compat import get_clip_module, get_extraction_runtime
from src.imagenet_loader import ensure_imagenetv2, load_imagenet


IMAGENET_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def extract_imagenet() -> None:
    device, batch_size, num_workers, pin_memory = get_extraction_runtime()

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Batch size:", batch_size)
    print("DataLoader workers:", num_workers)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    image_dir = ensure_imagenetv2()
    print(f"Using ImageNetV2 matched-frequency data at: {image_dir}")

    clip = get_clip_module()
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    loader, class_names = load_imagenet(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        preprocess=preprocess,
    )
    print(f"ImageNet sample count: {len(loader.dataset)}")

    image_features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.detach())
            labels.append(targets)

    image_features = torch.cat(image_features).cpu().numpy()
    labels = torch.cat(labels).numpy()

    with torch.no_grad():
        text_feature_list = []
        for template in IMAGENET_TEMPLATES:
            prompts = [template.format(name) for name in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "imagenet_image_features.npy", image_features)
    np.save(output_dir / "imagenet_text_features.npy", text_features.cpu().numpy())
    np.save(output_dir / "imagenet_labels.npy", labels)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_imagenet()
