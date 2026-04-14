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

from src.clip_compat import get_clip_module
from src.pet_loader import load_pets


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


def _load_pets_split(preprocess):
    split_path = Path("data/splits/split_zhou_OxfordPets.json")
    root_candidates = [
        Path("data/raw/PET_fresh/oxford-iiit-pet/images"),
        Path("data/raw/PET/oxford-iiit-pet/images"),
    ]

    if not split_path.exists():
        return None

    root_dir = None
    for candidate in root_candidates:
        if candidate.exists():
            root_dir = candidate
            break

    if root_dir is None:
        return None

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_entries = payload["test"]

    samples = []
    max_label = max(entry[1] for entry in test_entries)
    class_names = [""] * (max_label + 1)

    missing = 0
    for rel_name, label, class_name in test_entries:
        image_path = root_dir / rel_name
        if not image_path.exists():
            missing += 1
            continue
        samples.append((image_path, int(label)))
        class_names[int(label)] = str(class_name).strip().lower()

    if not samples:
        return None
    if missing > 0:
        print(f"[Warning] OxfordPets split: skipped {missing} missing files from split_zhou test set.")

    dataset = SplitImageDataset(samples=samples, transform=preprocess)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    return loader, class_names


def extract_pets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    clip = get_clip_module()
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    split_loaded = _load_pets_split(preprocess=preprocess)
    if split_loaded is None:
        print("split_zhou_OxfordPets.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_pets()
        class_names = [c.replace("_", " ").lower() for c in class_names]
    else:
        loader, class_names = split_loaded
        print(f"Using split_zhou test split for OxfordPets: {len(loader.dataset)} samples")

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
        "a photo of a {}, a type of pet.",
        "a photo of the pet {}.",
        "a cute photo of a {}.",
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

    np.save("data/processed/pets_image_features.npy", image_features)
    np.save("data/processed/pets_text_features.npy", text_features.cpu().numpy())
    np.save("data/processed/pets_labels.npy", labels)
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_pets()
