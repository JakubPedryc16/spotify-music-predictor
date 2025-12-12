import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, make_scorer

from src.data.data_loader import load_data
from src.models.manual_metrics import accuracy, precision, recall, f1 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILE_PATH = "data/data_mean_minmax.csv"
OUT_DIR = Path("optimization_results") / "grid_hyperparameter_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
F1_SCORER = make_scorer(f1_score, average='macro')

BEST_K_FEATURES = 8

def prepare_data_with_selection(file_path: str, n_splits: int, k_features: int) -> tuple[np.ndarray, np.ndarray, list]:
    df = load_data(file_path)
    target_col = "track_genre"
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= n_splits].index
    df = df[df[target_col].isin(valid_classes)].copy()

    X_full_df = df[numeric_cols].drop(target_col, axis=1, errors='ignore')
    y = df[target_col].to_numpy()
    
    X = X_full_df.to_numpy()

    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_opt = selector.fit_transform(X, y) 
    
    selected_indices = selector.get_support(indices=True)
    feature_names = X_full_df.columns.tolist()
    selected_features = [feature_names[i] for i in selected_indices]
    
    return X_opt, y, selected_features


def run_grid_search():
    print("--- Etap 3.1: Grid Search CV (kNN) ---")
    
    X_opt, y, selected_features = prepare_data_with_selection(FILE_PATH, N_SPLITS, BEST_K_FEATURES)
    print(f"Dane przygotowane. Użyte cechy ({len(selected_features)}): {selected_features}")
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15], 
        'weights': ['uniform', 'distance'],  
        'metric': ['euclidean', 'manhattan'], 
    }
    
    knn = KNeighborsClassifier()
    
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring=F1_SCORER,
        cv=N_SPLITS,
        verbose=3,
        n_jobs=-1,
    )

    print(f"\nRozpoczęcie przeszukiwania Grid Search na {len(X_opt)} próbkach, CV={N_SPLITS}...")
    
    start_time = time()
    grid_search.fit(X_opt, y)
    end_time = time()
    
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    
    df_results = pd.DataFrame({
        "Method": ["Grid Search CV"],
        "Time_seconds": [end_time - start_time],
        "Best_F1_CV": [best_score],
        "Best_Parameters": [best_params]
    })
    
    out_csv = OUT_DIR / "knn_grid_search_results.csv"
    df_results.to_csv(out_csv, index=False)
    
    print("\n--- Wynik końcowy Grid Search CV ---")
    print(f"Czas przeszukiwania: {end_time - start_time:.2f} s")
    print(f"Najlepszy F1-score (macro, CV={N_SPLITS}): {best_score:.4f}")
    print(f"Najlepsze parametry: {best_params}")
    print(f"Wyniki zapisane do: {out_csv}")
    
    return df_results.iloc[0]

if __name__ == "__main__":
    run_grid_search()