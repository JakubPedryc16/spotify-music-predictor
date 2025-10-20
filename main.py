import pandas as pd
import numpy as np
from graphs import run_analysis

df = pd.read_csv("dataset.csv")
df = df.rename(columns={'Unnamed: 0': 'id'})

print(df.head())
print(df.info())

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

columns_to_mask = df.columns.difference(['id'])


df_mask = pd.DataFrame(np.random.rand(
    df.shape[0],
    len(columns_to_mask)) < 0.1,
    columns=columns_to_mask
)

df[columns_to_mask] = df[columns_to_mask].mask(df_mask)

df.to_csv("dataset_with_missing.csv", index=False)

if __name__ == "__main__":
    run_analysis(show=True)   # show=True żeby wyświetlało wykresy

