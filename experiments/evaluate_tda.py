import numpy as np
import torch
from models.TDA import TDA


def evaluate(dataset, cache_size, alpha, beta, device):

    image_features = np.load(f"data/processed/{dataset}_image_features.npy")
    text_features = np.load(f"data/processed/{dataset}_text_features.npy")
    labels = np.load(f"data/processed/{dataset}_labels.npy")

    image_features = torch.tensor(image_features).float().to(device)
    text_features = torch.tensor(text_features).float().to(device)

    model = TDA(text_features=text_features, cache_size=cache_size, alpha=alpha, beta=beta, device=device)

    correct = 0

    for i in range(len(labels)):

        x = image_features[i]   # already on device

        pred, confidence, final_logits = model.predict(x)

        if pred.item() == labels[i]:
            correct += 1

    return correct / len(labels)