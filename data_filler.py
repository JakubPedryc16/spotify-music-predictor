import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def impute_knn(df, columns, n_neighbors=5):
    knn = KNNImputer(n_neighbors=n_neighbors)
    df[columns] = knn.fit_transform(df[columns])
    return df

def impute_mean(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def impute_median(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def fill_missing_data(filling_method):
    df = pd.read_csv("dataset_with_missing.csv")
    df = df.rename(columns={'Unnamed: 0': 'id'})

    if 'track_genre' in df.columns:
        unique_genres = df['track_genre'].dropna().unique()
        genre_map = {genre: i for i, genre in enumerate(unique_genres)}
        df['track_genre'] = df['track_genre'].map(genre_map)
    
    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(float)

    columns_to_impute = df.select_dtypes(include=[np.number]).columns.difference(['id', 'track_genre'])
    df = filling_method(df, columns_to_impute)

    if 'track_genre' in df.columns:
        df['track_genre'] = df['track_genre'].fillna(df['track_genre'].mean()).round().astype('Int64')

    int_columns = ['key', 'mode', 'time_signature']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    df.to_csv("dataset_imputed.csv", index=False)
    print("Plik zapisany jako dataset_imputed.csv")

if __name__ == "__main__":
    fill_missing_data(impute_mean)
