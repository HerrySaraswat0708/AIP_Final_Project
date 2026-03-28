# import torch
# from torchvision.datasets import DTD
# from torchvision import transforms
# from torch.utils.data import DataLoader


# class DatasetLoader:

#     def __init__(self, batch_size=1):

#         self.batch_size = batch_size

#         self.transform = transforms.Compose([
#             transforms.Resize(224),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.48145466, 0.4578275, 0.40821073],
#                 std=[0.26862954, 0.26130258, 0.27577711]
#             )
#         ])

#     def load_dataset(self):

#         dataset = DTD(
#             root="data/raw/DTD",
#             split="test",
#             download=True,
#             transform=self.transform
#         )

#         loader = DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=2
#         )

#         class_names = dataset.classes

#         return loader, class_names

import torch
from torchvision.datasets import DTD
from torchvision import transforms
from torch.utils.data import DataLoader


def load_dtd():
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
    dataset = DTD(
        root="data/raw/DTD",
        split="test",
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=128,       
        shuffle=True,
        num_workers=0,         
        pin_memory=True        
        )

    class_names = dataset.classes

    return loader, class_names
