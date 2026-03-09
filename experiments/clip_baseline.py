import numpy as np
import torch

image_features = np.load("data/processed/image_features.npy")
text_features = np.load("data/processed/text_features.npy")
labels = np.load("data/processed/labels.npy")

image_features = torch.tensor(image_features).float()
text_features = torch.tensor(text_features).float()

correct = 0

for i in range(len(labels)):

    x = image_features[i]

    logits = x @ text_features.T

    pred = torch.argmax(logits)

    if pred == labels[i]:
        correct += 1

acc = correct / len(labels)

print("CLIP Zero-shot accuracy:", acc * 100)