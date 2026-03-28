import mlflow
from experiments.evaluate_tda import evaluate
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets = [
    "dtd",
    "caltech",
    "eurosat",
    "pets"
]

alpha_list = [0.05]
beta_list = [8]
cache_list = [100]
# alpha_list = [0.04,0.05,0.06]
# beta_list = [6,7,8,9]
# cache_list = [500,1000,1500]

mlflow.set_experiment("FreeTTA_Hyperparameter_Tuning")

for dataset in datasets:
    for cache_size in cache_list:
        for alpha in alpha_list:
            for beta in beta_list:
                    with mlflow.start_run():
                        acc = evaluate(dataset,cache_size, alpha, beta, device)
                        mlflow.log_param("dataset", dataset)
                        mlflow.log_param("cache size", cache_size)
                        mlflow.log_param("alpha", alpha)
                        mlflow.log_param("beta", beta)
                        mlflow.log_metric("accuracy", acc)
                        print(dataset, cache_size, alpha, beta, acc)