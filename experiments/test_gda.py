import numpy as np
import torch
from models.gda_model import GDAModel

text_features = np.load("data/processed/text_features.npy")
image_features = np.load("data/processed/image_features.npy")

gda = GDAModel(text_features)

x = torch.tensor(image_features[0]).float().to(gda.device)

prediction, posterior = gda.predict(x)

print("Prediction:", prediction)
print("Posterior:", posterior)