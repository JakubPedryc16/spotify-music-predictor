import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")
print(df.head())
print(df.info())

print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

if df.isnull().sum().sum() < 10:
    df_mask = pd.DataFrame(np.random.rand(*df.shape) < 0.1, columns=df.columns)
    df = df.mask(df_mask)


df.to_csv("dataset_with_missing.csv", index=False)