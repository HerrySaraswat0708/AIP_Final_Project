import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.eurosat_loader import load_eurosat


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


NEW_CLASSNAMES = {
    "annualcrop": "annual crop land",
    "forest": "forest",
    "herbaceousvegetation": "brushland or shrubland",
    "highway": "highway or road",
    "industrial": "industrial buildings or commercial buildings",
    "pasture": "pasture land",
    "permanentcrop": "permanent crop land",
    "residential": "residential buildings or homes or apartments",
    "river": "river",
    "sealake": "lake or sea",
    "annual crop land": "annual crop land",
    "herbaceous vegetation land": "brushland or shrubland",
    "highway or road": "highway or road",
    "industrial buildings": "industrial buildings or commercial buildings",
    "pasture land": "pasture land",
    "permanent crop land": "permanent crop land",
    "residential buildings": "residential buildings or homes or apartments",
    "sea or lake": "lake or sea",
}


def _load_eurosat_split(preprocess):
    split_path = Path("data/splits/split_zhou_EuroSAT.json")
    root_candidates = [
        Path("data/raw/EUROSAT/eurosat/2750"),
        Path("data/raw/EUROSAT_fresh/eurosat/2750"),
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

    for rel_path, label, class_name in test_entries:
        image_path = root_dir / rel_path
        if not image_path.exists():
            return None
        samples.append((image_path, int(label)))
        raw_name = str(class_name).strip()
        mapped = NEW_CLASSNAMES.get(raw_name.lower().replace(" ", ""), None)
        if mapped is None:
            mapped = NEW_CLASSNAMES.get(raw_name.lower(), raw_name.lower())
        class_names[int(label)] = mapped

    dataset = SplitImageDataset(samples=samples, transform=preprocess)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    return loader, class_names


def extract_eurosat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    split_loaded = _load_eurosat_split(preprocess=preprocess)
    if split_loaded is None:
        print("split_zhou_EuroSAT.json not available/readable, falling back to torchvision loader.")
        loader, class_names = load_eurosat()
        class_names = [NEW_CLASSNAMES.get(c.lower(), c.replace("_", " ").lower()) for c in class_names]
    else:
        loader, class_names = split_loaded
        print(f"Using split_zhou test split for EuroSAT: {len(loader.dataset)} samples")

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
        "aerial imagery showing {}.",
        "an orthographic view of {}.",
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

    np.save("data/processed/eurosat_image_features.npy", image_features)
    np.save("data/processed/eurosat_text_features.npy", text_features.cpu().numpy())
    np.save("data/processed/eurosat_labels.npy", labels)
    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_eurosat()
