import os
import pandas as pd
import numpy as np
from src.optimalisation.genetics_optimization import run_genetic_search
from src.optimalisation.auto_gluon_optimization import run_autogluon_search
from src.optimalisation.optuna_optimization import run_optuna_search
from src.optimalisation.grid_optimization import run_grid_search
from src.all_data_pipeline import run_pipeline
from src.optimalisation.feature_optimization import run_feature_optimization
from src.prediction_visualisation.graphs import run_analysis


# df = pd.read_csv("data/data_original.csv")
# df = df.rename(columns={'Unnamed: 0': 'id'})

# print(df.head())
# print(df.info())

# print(df.isnull().sum())
# print(df[df.isnull().any(axis=1)])

# columns_to_mask = df.columns.difference(['id'])

# n_rows = df.shape[0]
# n_cols = len(columns_to_mask)
# random_matrix = np.random.rand(n_rows, n_cols)

# df_mask = pd.DataFrame(
#     random_matrix < 0.1,
#     columns=columns_to_mask
# )

# df[columns_to_mask] = df[columns_to_mask].mask(df_mask)

# df.to_csv("data/data_original.csv", index=False)

# if __name__ == "__main__":
#     data_variants = [
#         "data_auto_minmax.csv",
#         "data_auto_standardized.csv",
#         "data_knn_minmax.csv",
#         "data_knn_standardized.csv",
#         "data_mean_minmax.csv",
#         "data_mean_standardized.csv",
#         "data_median_minmax.csv",
#         "data_median_standardized.csv",
#     ]

#     base_dir = "data"
    
#     print(f"Rozpoczynanie pipeline dla {len(data_variants)} wariantów danych...")

#     for variant in data_variants:
#         file_path = os.path.join(base_dir, variant)
#         print(f"\n--- Przetwarzanie: {file_path} ---")
#         try:
#             run_pipeline(file_path)
#         except FileNotFoundError:
#             print(f"Błąd: Nie znaleziono pliku {file_path}. Upewnij się, że ścieżka jest poprawna.")
#         except Exception as e:
#             print(f"Wystąpił błąd podczas przetwarzania {file_path}: {e}")

#     print("\nAutomatyzacja zakończona. Wyniki zapisane w plikach 'results_*.csv'.")

if __name__ == "__main__":
    run_feature_optimization()
    run_grid_search()
    run_optuna_search() 
    run_genetic_search()
    run_autogluon_search()
    