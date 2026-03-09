import mlflow
from experiments.eval_dataset import evaluate

datasets=["dtd","caltech","eurosat","pets"]

mlflow.set_experiment("FreeTTA_MultiDataset")

for dataset in datasets:

    with mlflow.start_run(run_name=dataset):

        acc=evaluate(dataset)

        mlflow.log_param("dataset",dataset)
        mlflow.log_metric("accuracy",acc)

        print(dataset,acc)