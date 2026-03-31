from __future__ import annotations

import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dtd_loader import load_dtd


class SplitImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def _find_dtd_root() -> Path | None:
    candidates = [
        Path("data/raw/DTD_fresh/dtd"),
        Path("data/raw/DTD/dtd"),
        Path("data/raw/DTD/dtd/dtd"),
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    return None


def _read_label_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _auto_generate_dtd_split(split_path: Path, dtd_root: Path) -> bool:
    labels_dir = dtd_root / "labels"
    train_rel = _read_label_file(labels_dir / "train1.txt")
    val_rel = _read_label_file(labels_dir / "val1.txt")
    test_rel = _read_label_file(labels_dir / "test1.txt")

    if not train_rel or not test_rel:
        return False

    all_rel = train_rel + val_rel + test_rel
    class_tokens = sorted({rel.split("/")[0] for rel in all_rel if "/" in rel})
    if not class_tokens:
        return False

    label_map = {cls: i for i, cls in enumerate(class_tokens)}

    def _to_items(rel_paths: list[str]) -> list[list[object]]:
        items: list[list[object]] = []
        for rel in rel_paths:
            rel_norm = rel.replace("\\", "/")
            if "/" not in rel_norm:
                continue
            class_token = rel_norm.split("/")[0]
            if class_token not in label_map:
                continue
            class_name = class_token.replace("_", " ").lower()
            items.append([rel_norm, int(label_map[class_token]), class_name])
        return items

    payload = {
        "train": _to_items(train_rel),
        "val": _to_items(val_rel),
        "test": _to_items(test_rel),
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Info] Generated DTD split file at: {split_path}")
    return True


def _load_dtd_split(preprocess):
    split_candidates = [
        Path("data/splits/split_zhou_DescribableTextures.json"),
        Path("data/splits/split_zhou_DescribableTextures_tda.json"),
    ]

    dtd_root = _find_dtd_root()
    if dtd_root is None:
        return None

    split_path = None
    for candidate in split_candidates:
        if candidate.exists():
            split_path = candidate
            break

    if split_path is None:
        generated_path = split_candidates[0]
        generated = _auto_generate_dtd_split(generated_path, dtd_root)
        if not generated:
            return None
        split_path = generated_path

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_entries = payload.get("test", [])
    if not test_entries:
        return None

    image_dir = dtd_root / "images"

    samples = []
    max_label = max(int(entry[1]) for entry in test_entries)
    class_names = [""] * (max_label + 1)

    missing = 0
    for rel_path, label, class_name in test_entries:
        image_path = image_dir / str(rel_path)
        if not image_path.exists():
            missing += 1
            continue
        idx = int(label)
        samples.append((image_path, idx))
        class_names[idx] = str(class_name).strip().lower()

    if not samples:
        return None

    if missing > 0:
        print(f"[Warning] DTD split: skipped {missing} missing files from test set.")

    loader = DataLoader(
        SplitImageDataset(samples=samples, transform=preprocess),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return loader, class_names


def extract_dtd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    split_loaded = _load_dtd_split(preprocess=preprocess)
    if split_loaded is None:
        print("split_zhou_DescribableTextures.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_dtd()
        class_names = [c.replace("_", " ").lower() for c in class_names]
    else:
        loader, class_names = split_loaded
        print(f"Using split_zhou test split for DTD: {len(loader.dataset)} samples")

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

    templates = [
        "a cropped photo of {} texture.",
        "a photo of the {} pattern.",
        "{}",
    ]
    with torch.no_grad():
        text_feature_list = []
        for template in templates:
            prompts = [template.format(c.replace('_', ' ')) for c in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    np.save("data/processed/DTD_image_features.npy", image_features)
    np.save("data/processed/DTD_text_features.npy", text_features.cpu().numpy())
    np.save("data/processed/DTD_labels.npy", labels)

    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_dtd()
