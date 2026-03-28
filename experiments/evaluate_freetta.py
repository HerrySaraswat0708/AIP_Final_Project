import numpy as np
import torch
from models.FreeTTA import FreeTTA


def evaluate(dataset, alpha, beta, device):

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    image_features = np.load(f"data/processed/{dataset}_image_features.npy")
    text_features = np.load(f"data/processed/{dataset}_text_features.npy")
    labels = np.load(f"data/processed/{dataset}_labels.npy")

    image_features = torch.tensor(image_features).float().to(device)
    text_features = torch.tensor(text_features).float().to(device)

    model = FreeTTA(text_features, alpha=alpha, beta=beta, device=device)

    correct = 0

    for i in range(len(labels)):

        x = image_features[i]   # already on device

        clip_logits = 100 * (x @ text_features.T)

        clip_probs = torch.softmax(clip_logits, dim=-1)

        pred = model.predict(x, clip_probs, clip_logits)

        if pred.item() == labels[i]:
            correct += 1

    return correct / len(labels)