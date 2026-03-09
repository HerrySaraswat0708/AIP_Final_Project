import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Connect to your MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["FreeTTA_Hyperparameter_Search"])

# 2. Extract and clean data
df = runs[['params.alpha', 'params.beta', 'metrics.accuracy']].copy()
df['params.alpha'] = pd.to_numeric(df['params.alpha'])
df['params.beta'] = pd.to_numeric(df['params.beta'])
df['metrics.accuracy'] = pd.to_numeric(df['metrics.accuracy'])

# 3. Handle duplicates by taking the maximum accuracy for each pair
pivot_table = df.pivot_table(
    index='params.beta', 
    columns='params.alpha', 
    values='metrics.accuracy', 
    aggfunc='max'
)

# 4. Plotting
plt.figure(figsize=(12, 10))

# annot=True ensures numbers are visible inside the boxes
sns.heatmap(
    pivot_table, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Accuracy (%)'}
)

plt.title('FreeTTA Accuracy: Alpha vs Beta (DTD Dataset)', fontsize=15)

# USE RAW STRINGS (r'') TO FIX THE TRACEBACK ERROR
plt.xlabel(r'Alpha ($\alpha$)', fontsize=12)
plt.ylabel(r'Beta ($\beta$)', fontsize=12)

plt.tight_layout()
plt.show()