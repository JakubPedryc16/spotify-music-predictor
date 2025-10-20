import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

    df['track_genre'] = df['track_genre'].fillna('unknown')
    le = LabelEncoder()
    df['track_genre'] = le.fit_transform(df['track_genre'])

    df['explicit'] = df['explicit'].astype(float)

    columns_to_impute = df.select_dtypes(include=[np.number]).columns.difference(['id'])
    df = filling_method(df, columns_to_impute)

    int_columns = ['key', 'time_signature', 'track_genre']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    df.to_csv("dataset_imputed.csv", index=False)
    print("Plik zapisany jako dataset_imputed.csv")

if __name__ == "__main__":
    fill_missing_data(impute_mean)