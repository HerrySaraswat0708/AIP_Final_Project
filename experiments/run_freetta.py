import numpy as np
import torch

from models.online_em import OnlineEM


text_features = np.load("data/processed/text_features.npy")
image_features = np.load("data/processed/image_features.npy")

model = OnlineEM(text_features)

for i in range(10):

    x = torch.tensor(image_features[i]).float().to(model.device)

    clip_probs = torch.softmax(torch.randn(model.num_classes), dim=0).to(model.device)
    clip_logits = torch.log(clip_probs)

    pred = model.process_sample(x, clip_probs, clip_logits)

    print("Prediction:", pred)