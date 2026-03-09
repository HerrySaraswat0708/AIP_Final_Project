import torch
import clip
import numpy as np
from tqdm import tqdm
from src.eurosat_loader import load_eurosat


def extract_eurosat():

    device="cuda" if torch.cuda.is_available() else "cpu"

    model,_=clip.load("ViT-B/16",device=device)

    loader,class_names=load_eurosat()

    image_features=[]
    labels=[]

    with torch.no_grad():

        for images,targets in tqdm(loader):

            images=images.to(device)

            features=model.encode_image(images)

            features=features/features.norm(dim=-1,keepdim=True)

            image_features.append(features.cpu().numpy())
            labels.append(targets.numpy())

    image_features=np.concatenate(image_features)
    labels=np.concatenate(labels)

    prompts=[f"a satellite photo of {c}" for c in class_names]

    tokens=clip.tokenize(prompts).to(device)

    with torch.no_grad():

        text_features=model.encode_text(tokens)

        text_features=text_features/text_features.norm(dim=-1,keepdim=True)

    np.save("data/processed/eurosat_image_features.npy",image_features)
    np.save("data/processed/eurosat_text_features.npy",text_features.cpu().numpy())
    np.save("data/processed/eurosat_labels.npy",labels)

if __name__ == "__main__":
    extract_eurosat()