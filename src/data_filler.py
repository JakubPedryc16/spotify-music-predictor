import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_auto_sklearn(df, columns):
    df_imputed = df.copy()
    imputer = IterativeImputer(random_state=0)
    df_imputed.loc[:, columns] = imputer.fit_transform(df_imputed[columns])
    return df_imputed

def knn_impute(df, columns, n_neighbors=5):
    df_imputed = df.copy()
    data = df[columns].to_numpy()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                target = data[i, :]
                valid_rows = ~np.isnan(data[:, j])

                if np.sum(valid_rows) == 0:
                    continue

                candidates = data[valid_rows, :]
                mask = ~np.isnan(target)
                mask[j] = False

                if np.sum(mask) == 0:
                    continue

                diffs = candidates[:, mask] - target[mask]
                distances = np.sqrt(np.nansum(diffs ** 2, axis=1))
                k_idx = np.argsort(distances)[:n_neighbors]
                neighbors_values = candidates[k_idx, j]
                data[i, j] = np.nanmean(neighbors_values)

    df_imputed[columns] = data
    return df_imputed


def impute_mean(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def impute_median(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_data(filling_method, name=""):
    df = pd.read_csv("data/data_missing.csv")
    df = df.rename(columns={'Unnamed: 0': 'id'})

    if 'track_genre' in df.columns:
        unique_genres = df['track_genre'].dropna().unique()
        genre_map = {genre: i for i, genre in enumerate(unique_genres)}
        df['track_genre'] = df['track_genre'].map(genre_map)
    
    if 'track_id' in df.columns: 
        df = df.groupby('track_id', as_index=False).first()

    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(float)

    columns_to_impute = df.select_dtypes(include=[np.number]).columns.difference(['id'])
    df = filling_method(df, columns_to_impute)

    int_columns = ['key', 'mode', 'time_signature', 'track_genre']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    df.to_csv(f"data/data{name}.csv", index=False)
    print(f"Plik zapisany jako data{name}.csv")



if __name__ == "__main__":
    fill_missing_data(impute_mean, "_mean")
    fill_missing_data(impute_median, "_median")
    fill_missing_data(knn_impute, "_knn")
    fill_missing_data(impute_auto_sklearn, "_auto")