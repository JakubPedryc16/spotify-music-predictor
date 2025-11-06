import pandas as pd
import numpy as np
from src.graphs import run_analysis


df = pd.read_csv("data/data_original.csv")
df = df.rename(columns={'Unnamed: 0': 'id'})

print(df.head())
print(df.info())

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

columns_to_mask = df.columns.difference(['id'])

n_rows = df.shape[0]
n_cols = len(columns_to_mask)
random_matrix = np.random.rand(n_rows, n_cols)

df_mask = pd.DataFrame(
    random_matrix < 0.1,
    columns=columns_to_mask
)

df[columns_to_mask] = df[columns_to_mask].mask(df_mask)

df.to_csv("data/data_original.csv", index=False)

if __name__ == "__main__":
    run_analysis(show=True) 

