import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import json
import optuna

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

from src.data.data_loader import load_data
from src.models.manual_metrics import accuracy, precision, recall, f1 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILE_PATH = "data/data_mean_minmax.csv"
OUT_DIR = Path("optimization_results") / "optuna_hyperparameter_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
BEST_K_FEATURES = 8
N_TRIALS = 50

def prepare_data_with_selection(file_path: str, n_splits: int, k_features: int) -> tuple[np.ndarray, np.ndarray]:
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
    
    return X_opt, y


def objective(trial, X, y):
    param = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
    }
    
    knn = KNeighborsClassifier(**param)
    
    cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    f1_scores = []
    
    for train_index, val_index in cv_splitter.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        score = f1_score(y_val, y_pred, average='macro') 
        f1_scores.append(score)
        
    return np.mean(f1_scores)


def run_optuna_search():
    print("--- Etap 3.2: Optymalizacja Hiperparametrów - Optuna (kNN) ---")
    
    X_opt, y = prepare_data_with_selection(FILE_PATH, N_SPLITS, BEST_K_FEATURES)

    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    print(f"\nRozpoczęcie przeszukiwania Optuna (N_TRIALS={N_TRIALS}, CV={N_SPLITS})...")
    
    start_time = time()
    study.optimize(
        lambda trial: objective(trial, X_opt, y), 
        n_trials=N_TRIALS, 
        n_jobs=-1,
        show_progress_bar=True
    )
    end_time = time()
    
    best_score = study.best_value
    best_params = study.best_params
    
    df_results = pd.DataFrame({
        "Method": ["Optuna (TPE)"],
        "Time_seconds": [end_time - start_time],
        "Best_F1_CV": [best_score],
        "Best_Parameters": [best_params]
    })
    
    out_csv = OUT_DIR / "knn_optuna_search_results.csv"
    df_results.to_csv(out_csv, index=False)
    
    print("\n--- Wynik końcowy Optuna ---")
    print(f"Liczba prób (Trials): {N_TRIALS}")
    print(f"Czas przeszukiwania: {end_time - start_time:.2f} s")
    print(f"Najlepszy F1-score (macro, CV={N_SPLITS}): {best_score:.4f}")
    print(f"Najlepsze parametry: {best_params}")
    print(f"Wyniki zapisane do: {out_csv}")
    
    return df_results.iloc[0]

if __name__ == "__main__":
    run_optuna_search()