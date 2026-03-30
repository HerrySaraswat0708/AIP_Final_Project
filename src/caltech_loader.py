# import torch
# from torchvision.datasets import Caltech101
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def load_caltech(batch_size=32):

#     transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     dataset = Caltech101(
#         root="data/raw/CALTECH",
#         download=True,
#         transform=transform
#     )

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     return loader, dataset.categories

# import torch
from pathlib import Path
from torchvision.datasets import Caltech101
from torchvision import transforms
from torch.utils.data import DataLoader


def load_caltech():

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std=[0.26862954,0.26130258,0.27577711]
        )
    ])

    if Path("data/raw/CALTECH_clean").exists():
        root_dir = "data/raw/CALTECH_clean"
    elif Path("data/raw/CALTECH_fresh").exists():
        root_dir = "data/raw/CALTECH_fresh"
    else:
        root_dir = "data/raw/CALTECH"
    dataset = Caltech101(
        root=root_dir,
        download=True,
        transform=transform
    )

    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader, dataset.categories
