import mlflow
import numpy as np
import torch
from models.online_em import OnlineEM


def evaluate(dataset):

    image_features=np.load(f"data/processed/{dataset}_image_features.npy")
    text_features=np.load(f"data/processed/{dataset}_text_features.npy")
    labels=np.load(f"data/processed/{dataset}_labels.npy")

    image_features=torch.tensor(image_features).float()
    text_features=torch.tensor(text_features).float()

    model=OnlineEM(text_features)

    correct=0

    for i in range(len(labels)):

        x=image_features[i]

        clip_logits=100*(x@text_features.T)

        clip_probs=torch.softmax(clip_logits,dim=-1)

        pred=model.process_sample(x,clip_probs,clip_logits)

        if pred.item()==labels[i]:
            correct+=1

    acc=correct/len(labels)

    return acc