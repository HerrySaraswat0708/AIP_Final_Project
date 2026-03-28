# import os
# import torch
# from src.dtd_loader import DatasetLoader
# import clip
# import numpy as np 
# from tqdm import tqdm

# class CLIPFeatureExtractor:

#     def __init__(self, batch_size=1):

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         print("Device:", self.device)

#         self.model, _ = clip.load("ViT-B/16", device=self.device)

#         loader = DatasetLoader(batch_size=batch_size)

#         self.dataloader, self.class_names = loader.load_dataset()

#         print("Number of classes:", len(self.class_names))

#     def extract_image_features(self):

#         image_features = []
#         labels = []

#         with torch.no_grad():

#             for images, targets in tqdm(self.dataloader):

#                 images = images.to(self.device)

#                 features = self.model.encode_image(images)

#                 features = features / features.norm(dim=-1, keepdim=True)

#                 image_features.append(features.cpu().numpy())
#                 labels.append(targets.numpy())

#         image_features = np.concatenate(image_features)
#         labels = np.concatenate(labels)

#         return image_features, labels

#     def extract_text_features(self):

#         prompts = [f"a photo of {c} texture" for c in self.class_names]

#         tokens = clip.tokenize(prompts).to(self.device)

#         with torch.no_grad():

#             text_features = self.model.encode_text(tokens)

#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         return text_features.cpu().numpy()

#     def save_features(self, image_features, text_features, labels):

#         os.makedirs("data/processed", exist_ok=True)

#         np.save("data/processed/DTD_image_features.npy", image_features)
#         np.save("data/processed/DTD_text_features.npy", text_features)
#         np.save("data/processed/DTD_labels.npy", labels)

#         print("Features saved.")

#     def run(self):

#         print("Extracting image features...")
#         image_features, labels = self.extract_image_features()

#         print("Extracting text features...")
#         text_features = self.extract_text_features()

#         self.save_features(image_features, text_features, labels)


# if __name__ == "__main__":

#     extractor = CLIPFeatureExtractor()

#     extractor.run()

import torch
import clip
import numpy as np
from tqdm import tqdm
from src.dtd_loader import load_dtd


def extract_dtd():

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
    loader, class_names = load_dtd()

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
    np.save("data/processed/DTD_image_features.npy", image_features)
    np.save("data/processed/DTD_text_features.npy", text_features.cpu().numpy())
    np.save("data/processed/DTD_labels.npy", labels)

    print("Feature extraction completed!")


if __name__ == "__main__":
    extract_dtd()