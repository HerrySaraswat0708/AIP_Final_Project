# import torch
# import clip
# import numpy as np
# from tqdm import tqdm
# from src.caltech_loader import load_caltech


# def extract_caltech():

#     self.device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
#     print('Device :',self.device)
#     model,_ = clip.load("ViT-B/16", device=self.device)

#     loader, class_names = load_caltech()

#     image_features=[]
#     labels=[]

#     with torch.no_grad():
#         for images,targets in tqdm(loader):

#             images=images.to(self.device)

#             features=model.encode_image(images)

#             features=features/features.norm(dim=-1,keepdim=True)

#             image_features.append(features.cpu().numpy())
#             labels.append(targets.numpy())

#     image_features=np.concatenate(image_features)
#     labels=np.concatenate(labels)

#     prompts=[f"a photo of a {c}" for c in class_names]

#     tokens=clip.tokenize(prompts).to(self.device)

#     with torch.no_grad():

#         text_features=model.encode_text(tokens)

#         text_features=text_features/text_features.norm(dim=-1,keepdim=True)

#     np.save("data/processed/caltech_image_features.npy",image_features)
#     np.save("data/processed/caltech_text_features.npy",text_features.cpu().numpy())
#     np.save("data/processed/caltech_labels.npy",labels)

# if __name__ == "__main__":
#     extract_caltech()


import torch
import clip
import numpy as np
from tqdm import tqdm
from src.caltech_loader import load_caltech


def extract_caltech():

    # ------------------ DEVICE SETUP ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------ LOAD MODEL ------------------
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()

    # ------------------ LOAD DATA ------------------
    loader, class_names = load_caltech()

    image_features = []
    labels = []

    # ------------------ FEATURE EXTRACTION ------------------
    with torch.no_grad():
        for images, targets in tqdm(loader):

            # FAST transfer to GPU
            images = images.to(device, non_blocking=True)

            # Forward pass
            features = model.encode_image(images)

            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)

            # KEEP ON GPU (avoid CPU transfer every iteration)
            image_features.append(features.detach())
            labels.append(targets)

    # SINGLE transfer (FAST)
    image_features = torch.cat(image_features).cpu().numpy()
    labels = torch.cat(labels).numpy()

    # ------------------ TEXT FEATURES ------------------
    prompts = [f"a satellite photo of {c}" for c in class_names]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # ------------------ SAVE ------------------
    np.save("data/processed/caltech_image_features.npy", image_features)
    np.save("data/processed/caltech_text_features.npy", text_features.cpu().numpy())
    np.save("data/processed/caltech_labels.npy", labels)

    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_caltech()