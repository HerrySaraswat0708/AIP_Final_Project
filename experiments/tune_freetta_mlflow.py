import mlflow
from experiments.evaluate_freetta import evaluate

datasets = [
    # "dtd",
    # "caltech",
    "eurosat",
    "pets"
]

# alpha_list = [0.005, 0.01, 0.02]
# beta_list = [3.0, 4.5, 6.0]
alpha_list = [0.03,0.04,0.05,0.06,0.07]
beta_list = [6,7,8,9,10]

mlflow.set_experiment("FreeTTA_Hyperparameter_Tuning")

for dataset in datasets:

    for alpha in alpha_list:

        for beta in beta_list:

                with mlflow.start_run():

                    acc = evaluate(dataset, alpha, beta)

                    mlflow.log_param("dataset", dataset)
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("beta", beta)

                    mlflow.log_metric("accuracy", acc)

                    print(dataset, alpha, beta, acc)