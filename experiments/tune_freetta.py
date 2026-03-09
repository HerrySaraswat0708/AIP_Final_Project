import mlflow
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from models.online_em import FreeTTA_OnlineEM

def load_features():
    image_features = np.load("data/processed/image_features.npy")
    text_features = np.load("data/processed/text_features.npy")
    labels = np.load("data/processed/labels.npy")
    return torch.from_numpy(image_features).float(), torch.from_numpy(text_features).float(), labels

def compute_clip_logits(x, text_features):
    # Standard CLIP logit calculation
    logits = 100.0 * (x @ text_features.T)
    probs = torch.softmax(logits, dim=-1)
    return probs, logits

def run_experiment(image_features, text_features, labels, alpha, beta):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    
    # Initialize model with tuning params
    model = FreeTTA_OnlineEM(text_features, alpha=alpha, beta=beta)
    
    correct = 0
    total = len(labels)
    
    for i in range(total):
        x = image_features[i]
        clip_probs, clip_logits = compute_clip_logits(x, text_features)
        
        # Process sample according to Eq 11, 12, 13
        pred = model.process_sample(x, clip_probs, clip_logits)
        
        if pred.item() == labels[i]:
            correct += 1
            
    return 100 * correct / total

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    img_feats, txt_feats, labels = load_features()
    
    # Target search space based on paper performance
    alphas = [0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
    betas = [4,4.5,5,5.5,6,6.5,7,8,9,10]

    mlflow.set_experiment("FreeTTA_Hyperparameter_Search")

    for a in alphas:
        for b in betas:
            # Start MLflow run for each combination
            with mlflow.start_run(run_name=f"a_{a}_b_{b}"):
                acc = run_experiment(img_feats, txt_feats, labels, a, b)
                
                # Log hyperparameters and results
                mlflow.log_param("alpha", a)
                mlflow.log_param("beta", b)
                mlflow.log_metric("accuracy", acc)
                
                print(f"Alpha: {a}, Beta: {b} | Accuracy: {acc:.2f}%")