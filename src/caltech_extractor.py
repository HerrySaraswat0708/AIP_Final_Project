import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_compat import get_clip_module, get_extraction_runtime
from src.caltech_loader import load_caltech


def _resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


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


def _load_caltech_split(preprocess, batch_size: int, num_workers: int, pin_memory: bool):
    split_path = _resolve_project_path("data", "splits", "split_zhou_Caltech101.json")
    root_candidates = [
        _resolve_project_path("data", "raw", "CALTECH_clean", "caltech101", "101_ObjectCategories"),
        _resolve_project_path("data", "raw", "CALTECH_fresh", "caltech101", "101_ObjectCategories"),
        _resolve_project_path("data", "raw", "CALTECH", "caltech101", "101_ObjectCategories"),
    ]

    if not split_path.exists():
        print(f"[Info] Caltech split file not found: {split_path}")
        return None

    root_dir = None
    for candidate in root_candidates:
        if candidate.exists():
            root_dir = candidate
            break

    if root_dir is None:
        print("[Info] Caltech split root not found under expected data/raw directories.")
        return None

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_entries = payload.get("test", [])
    if not test_entries:
        print(f"[Info] Caltech split file has no test entries: {split_path}")
        return None

    samples = []
    max_label = max(entry[1] for entry in test_entries)
    class_names = [""] * (max_label + 1)
    missing_paths = []

    for rel_path, label, class_name in test_entries:
        image_path = root_dir / rel_path
        if not image_path.exists():
            missing_paths.append(str(image_path))
            continue
        samples.append((image_path, int(label)))
        class_names[int(label)] = str(class_name).strip().lower()

    if not samples:
        print(
            "[Warning] Caltech split test set could not be loaded because no listed files were found. "
            f"Root checked: {root_dir}"
        )
        if missing_paths:
            print(f"[Warning] Example missing Caltech split file: {missing_paths[0]}")
        return None

    if missing_paths:
        print(f"[Warning] Caltech split: skipped {len(missing_paths)} missing files from split_zhou test set.")
        print(f"[Warning] Example missing Caltech split file: {missing_paths[0]}")

    dataset = SplitImageDataset(samples=samples, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, class_names


def extract_caltech():
    device, batch_size, num_workers, pin_memory = get_extraction_runtime()
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Batch size:", batch_size)
    print("DataLoader workers:", num_workers)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    clip = get_clip_module()
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    split_loaded = _load_caltech_split(
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if split_loaded is None:
        print("split_zhou_Caltech101.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_caltech(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        class_names = [c.replace("_", " ").lower() for c in class_names]
        loader_source = "torchvision Caltech101 loader"
    else:
        loader, class_names = split_loaded
        print(f"Using split_zhou test split for Caltech101: {len(loader.dataset)} samples")
        loader_source = "split_zhou test split"

    dataset_size = len(loader.dataset)
    print(f"Caltech sample count from {loader_source}: {dataset_size}")
    if dataset_size == 0:
        raise RuntimeError(
            "Caltech feature extraction received an empty dataset. "
            f"Source: {loader_source}. cwd={Path.cwd()} project_root={PROJECT_ROOT}"
        )

    image_features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            image_features.append(features.detach())
            labels.append(targets)

    if not image_features:
        raise RuntimeError(
            "Caltech feature extraction produced zero batches before concatenation. "
            f"Source: {loader_source}. Dataset size: {dataset_size}."
        )

    image_features = torch.cat(image_features).cpu().numpy()
    labels = torch.cat(labels).numpy()

    templates = [
        "itap of a {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "a clean photo of a {}.",
    ]
    with torch.no_grad():
        text_feature_list = []
        for template in templates:
            prompts = [template.format(c) for c in class_names]
            tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feature_list.append(text_features)
        text_features = torch.stack(text_feature_list, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = _resolve_project_path("data", "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "caltech_image_features.npy", image_features)
    np.save(output_dir / "caltech_text_features.npy", text_features.cpu().numpy())
    np.save(output_dir / "caltech_labels.npy", labels)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_caltech()
