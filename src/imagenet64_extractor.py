from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_compat import get_clip_module
from src.imagenet_loader import IMAGENET_TEMPLATES
from src.imagenet64_loader import load_imagenet64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP features for downsampled ImageNet64 validation.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/Imagenet64_val"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-image-features",
        action="store_true",
        help="Reuse existing imagenet64_image_features.npy and imagenet64_labels.npy, and only refresh text features.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_imagenet64() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image_features_path = args.output_dir / "imagenet64_image_features.npy"
    labels_path = args.output_dir / "imagenet64_labels.npy"
    text_features_path = args.output_dir / "imagenet64_text_features.npy"

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    clip = get_clip_module()
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    loader, class_names = load_imagenet64(
        transform=preprocess,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Loaded ImageNet64 validation split with {len(loader.dataset)} samples and {len(class_names)} classes")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_image_features:
        if not image_features_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                "--skip-image-features was requested, but existing image features or labels were not found in "
                f"{args.output_dir}"
            )
        print(f"Reusing existing image features from {image_features_path}")
        print(f"Reusing existing labels from {labels_path}")
        image_features_np = np.load(image_features_path)
        labels_np = np.load(labels_path)
    else:
        image_features = []
        labels = []
        with torch.no_grad():
            for images, targets in tqdm(loader):
                images = images.to(device, non_blocking=True)
                features = model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                image_features.append(features.detach())
                labels.append(targets)

        image_features_np = torch.cat(image_features, dim=0).cpu().numpy()
        labels_np = torch.cat(labels, dim=0).numpy()

    with torch.no_grad():
        text_feature_list = []
        for template in IMAGENET_TEMPLATES:
            prompts = [template.format(class_name) for class_name in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    np.save(image_features_path, image_features_np)
    np.save(text_features_path, text_features.float().cpu().numpy())
    np.save(labels_path, labels_np)
    print("ImageNet64 feature extraction completed!")


if __name__ == "__main__":
    extract_imagenet64()
