import pandas as pd
import numpy as np
import os

def min_max_scaling(df, columns):
    df_scaled = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_scaled[col] = (df[col] - min_val) / (max_val - min_val)
    return df_scaled

def standardization(df, columns):
    df_scaled = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df_scaled[col] = (df[col] - mean_val) / std_val
    return df_scaled

def scale_dataset(file_path):
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(file_path)

    columns_to_scale = [
        'popularity', 'duration_ms', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo'
    ]

    columns_to_scale = [c for c in columns_to_scale if c in df.columns]

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    df_minmax = df.copy()
    df_minmax[columns_to_scale] = min_max_scaling(df, columns_to_scale)[columns_to_scale]
    df_minmax.to_csv(f"data/{base_name}_minmax.csv", index=False)

    df_standardized = df.copy()
    df_standardized[columns_to_scale] = standardization(df, columns_to_scale)[columns_to_scale]
    df_standardized.to_csv(f"data/{base_name}_standardized.csv", index=False)

if __name__ == "__main__":
    scale_dataset("data/data_mean.csv")
    scale_dataset("data/data_median.csv")
    scale_dataset("data/data_knn.csv")
    scale_dataset("data/data_auto.csv")