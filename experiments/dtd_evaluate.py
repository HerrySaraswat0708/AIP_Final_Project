import numpy as np
import torch
from tqdm import tqdm

# from models.online_em import OnlineEM

from models.online_em import FreeTTA_OnlineEM

def load_features():

    image_features = np.load("data/processed/image_features.npy")
    text_features = np.load("data/processed/text_features.npy")
    labels = np.load("data/processed/labels.npy")

    return image_features, text_features, labels


def compute_clip_logits(x, text_features):

    # CLIP cosine similarity with temperature scaling
    logits = 100.0 * (x @ text_features.T)

    probs = torch.softmax(logits, dim=-1)

    return probs, logits


def evaluate():

    print("Loading features...")

    image_features, text_features, labels = load_features()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_features = torch.tensor(image_features).float().to(device)
    text_features = torch.tensor(text_features).float().to(device)

    model = FreeTTA_OnlineEM(text_features)

    correct = 0
    total = len(labels)

    print("\nRunning FreeTTA evaluation...\n")

    for i in tqdm(range(total)):

        x = image_features[i]

        clip_probs, clip_logits = compute_clip_logits(x, text_features)

        pred = model.process_sample(x, clip_probs, clip_logits)

        if pred.item() == labels[i]:
            correct += 1

    accuracy = 100 * correct / total

    print("\n================================")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("================================")


if __name__ == "__main__":

    evaluate()