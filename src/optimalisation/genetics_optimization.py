import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

from src.data.data_loader import load_data 

from sklearn_genetic.space import Integer, Categorical
from sklearn_genetic import GASearchCV 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILE_PATH = "data/data_mean_minmax.csv"
OUT_DIR = Path("optimization_results") / "genetic_hyperparameter_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
BEST_K_FEATURES = 8
N_GENERATIONS = 20
POPULATION_SIZE = 40
N_JOBS = -1

def prepare_data_with_selection(file_path: str, n_splits: int, k_features: int) -> tuple[np.ndarray, np.ndarray]:
    
    try:
        df = load_data(file_path)
    except ImportError:
        df = pd.read_csv(file_path)

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

def run_genetic_search():
    print(f"--- Etap 3: Optymalizacja Hiperparametrów - Algorytm Genetyczny (kNN) ---")
    
    X_opt, y = prepare_data_with_selection(FILE_PATH, N_SPLITS, BEST_K_FEATURES)
    print(f"Dane wejściowe: {X_opt.shape[0]} próbek, {X_opt.shape[1]} cech (K={BEST_K_FEATURES}).")
    
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': Integer(3, 31),               
        'weights': Categorical(['uniform', 'distance']), 
        'metric': Categorical(['euclidean', 'manhattan', 'minkowski']), 
    }
    
    ga_search = GASearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED),
        scoring='f1_macro',
        n_jobs=N_JOBS,
        verbose=True, 
        generations=N_GENERATIONS, 
        population_size=POPULATION_SIZE
    )
    
    print(f"\nRozpoczęcie przeszukiwania genetycznego (Generacji: {N_GENERATIONS}, Populacja: {POPULATION_SIZE})...")
    
    ga_search.fit(X_opt, y)
    
    best_score = ga_search.best_score_
    best_params = ga_search.best_params_
    
    results = {
        "Method": "Genetic Algorithm Search (GA)",
        "Features_K": BEST_K_FEATURES,
        "Best_F1_CV": best_score,
        "Best_Parameters": best_params
    }
    
    results_df = pd.DataFrame([results])
    out_csv = OUT_DIR / "knn_genetic_optimization_results.csv"
    results_df.to_csv(out_csv, index=False)
    
    print("\n--- Wyniki Optymalizacji Genetycznej ---")
    print(f"Najlepszy F1-score (CV): {best_score:.4f}")
    print(f"Najlepsze hiperparametry: {best_params}")
    print(f"Wyniki zapisane do: {out_csv}")
    
    return results

if __name__ == "__main__":
    run_genetic_search()