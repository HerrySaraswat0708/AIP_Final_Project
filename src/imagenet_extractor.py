from __future__ import annotations

import argparse
from pathlib import Path

import clip
import numpy as np
import torch
from tqdm import tqdm

from src.imagenet_loader import load_imagenet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP features for ImageNet validation set.")
    parser.add_argument("--root", type=str, default="data/raw/IMAGENET")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)

    loader, class_names = load_imagenet(root=args.root, batch_size=args.batch_size)

    image_features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Encoding ImageNet images"):
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features.append(feats.cpu().numpy())
            labels.append(targets.numpy())

    prompts = [f"a photo of a {name}" for name in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features = np.concatenate(image_features, axis=0)
    labels = np.concatenate(labels, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "imagenet_image_features.npy", image_features)
    np.save(args.output_dir / "imagenet_text_features.npy", text_features.cpu().numpy())
    np.save(args.output_dir / "imagenet_labels.npy", labels)
    print("Saved ImageNet CLIP features to", args.output_dir)


if __name__ == "__main__":
    main()
