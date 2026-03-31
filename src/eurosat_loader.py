# from torchvision.datasets import EuroSAT
# from torchvision import transforms
# from torch.utils.data import DataLoader


# def load_eurosat(batch_size=32):

#     transform=transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466,0.4578275,0.40821073],
#             std=[0.26862954,0.26130258,0.27577711]
#         )
#     ])

#     dataset=EuroSAT(
#         root="data/raw/EUROSAT",
#         download=True,
#         transform=transform
#     )

#     loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)

#     return loader,dataset.classes

from pathlib import Path
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader


def load_eurosat():

    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std=[0.26862954,0.26130258,0.27577711]
        )
    ])

    root_dir = "data/raw/EUROSAT_fresh" if Path("data/raw/EUROSAT_fresh").exists() else "data/raw/EUROSAT"
    dataset=EuroSAT(
        root=root_dir,
        download=True,
        transform=transform
    )

    # loader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return loader,dataset.classes
