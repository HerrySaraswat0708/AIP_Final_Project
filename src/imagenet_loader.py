from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet


def load_imagenet(root: str = "data/raw/IMAGENET", batch_size: int = 32) -> Tuple[DataLoader, List[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    dataset = ImageNet(root=str(Path(root)), split="val", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset.classes
