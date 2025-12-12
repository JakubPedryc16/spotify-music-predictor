import numpy as np
import pandas as pd
from pathlib import Path
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
OUT_DIR = Path("optimization_results") / "feature_optimization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
N_TRIALS_OPTUNA = 50

def evaluate_model_cv(X: np.ndarray, y: np.ndarray, model: KNeighborsClassifier, n_splits: int = N_SPLITS):
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    f1_scores = []
    
    for train_index, val_index in cv_splitter.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        score = f1_score(y_val, y_pred, average='macro') 
        f1_scores.append(score)
        
    return np.mean(f1_scores), np.std(f1_scores)

def prepare_data(file_path: str, n_splits: int) -> tuple[pd.DataFrame, np.ndarray, list]:
    df = load_data(file_path)
    target_col = "track_genre"
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= n_splits].index
    df = df[df[target_col].isin(valid_classes)].copy()

    X_full_df = df[numeric_cols].drop(target_col, axis=1, errors='ignore')
    y = df[target_col].to_numpy()
    
    return X_full_df, y, X_full_df.columns.tolist()

def objective_optuna_features(trial, X: np.ndarray, y: np.ndarray, max_features: int):
    k = trial.suggest_int('k_features', 3, max_features)
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y) 
    
    knn = KNeighborsClassifier(n_neighbors=5) 
    
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_index, val_index in cv_splitter.split(X_selected, y):
        X_train, X_val = X_selected[train_index], X_selected[val_index]
        y_train, y_val = y[train_index], y[val_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        score = f1_score(y_val, y_pred, average='macro') 
        f1_scores.append(score)
        
    return np.mean(f1_scores)

def run_feature_optimization():
    print("--- Etap 2: Optymalizacja Cech (kNN + CV) ---")
    
    X_full_df, y, feature_names = prepare_data(FILE_PATH, N_SPLITS)
    X = X_full_df.to_numpy()
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    results = []
    
    print(f"Dane: {X.shape[0]} próbek, {len(np.unique(y))} klas.")
    
    mean_f1_base, std_f1_base = evaluate_model_cv(X, y, knn_model)
    results.append({
        "Method": "Benchmark (Full Features)",
        "Features_Count": len(feature_names),
        "Mean_F1_CV": mean_f1_base,
        "STD_F1_CV": std_f1_base,
        "Features_Used": ", ".join(feature_names)
    })
    print(f"\n[BENCHMARK] Pełen zestaw cech ({len(feature_names)}): F1-score: {mean_f1_base:.4f} +/- {std_f1_base:.4f}")
    
    corr_matrix = X_full_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    if len(to_drop) > 0:
        X_corr_filtered = X_full_df.drop(columns=to_drop)
        feature_names_corr = X_corr_filtered.columns.tolist()
        X_corr = X_corr_filtered.to_numpy()
        
        mean_f1_corr, std_f1_corr = evaluate_model_cv(X_corr, y, knn_model)
        results.append({
            "Method": "Correlation Filter (r > 0.90)",
            "Features_Count": len(feature_names_corr),
            "Mean_F1_CV": mean_f1_corr,
            "STD_F1_CV": std_f1_corr,
            "Features_Used": ", ".join(feature_names_corr)
        })
        print(f"[CORR FILTER] Usunięto: {', '.join(to_drop)}. F1-score: {mean_f1_corr:.4f} +/- {std_f1_corr:.4f}")
    else:
        print("[CORR FILTER] Nie znaleziono cech do usunięcia o korelacji > 0.90. Używamy pełnego zestawu.")
        X_corr_filtered = X_full_df
        feature_names_corr = feature_names
    

    k_values = [5, 8, 10, 12] 
    
    print("\n--- SelectKBest (f_classif) ---")
    
    for k in k_values:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected_k = selector.fit_transform(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        mean_f1_k, std_f1_k = evaluate_model_cv(X_selected_k, y, knn_model)
        
        results.append({
            "Method": f"SelectKBest (K={k})",
            "Features_Count": k,
            "Mean_F1_CV": mean_f1_k,
            "STD_F1_CV": std_f1_k,
            "Features_Used": ", ".join(selected_features)
        })
        print(f"[SELECT K={k}] F1-score: {mean_f1_k:.4f} +/- {std_f1_k:.4f}. Cechy: {selected_features}")

    print("\n--- Optuna dla SelectKBest (Optymalizacja K) ---")
    
    max_k = len(feature_names_corr) 
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    print(f"Rozpoczęcie przeszukiwania Optuna dla K (N_TRIALS={N_TRIALS_OPTUNA}, Max K={max_k})...")
    
    study.optimize(
        lambda trial: objective_optuna_features(trial, X_corr_filtered.to_numpy(), y, max_k), 
        n_trials=N_TRIALS_OPTUNA, 
        n_jobs=-1,
        show_progress_bar=True
    )
    
    best_k = study.best_params['k_features']
    
    selector_optuna = SelectKBest(score_func=f_classif, k=best_k)
    X_selected_optuna = selector_optuna.fit_transform(X_corr_filtered.to_numpy(), y)
    
    selected_indices_optuna = selector_optuna.get_support(indices=True)
    selected_features_optuna = [feature_names_corr[i] for i in selected_indices_optuna]
    
    mean_f1_optuna, std_f1_optuna = evaluate_model_cv(X_selected_optuna, y, knn_model)
    
    results.append({
        "Method": f"Optuna for SelectKBest (K={best_k})",
        "Features_Count": best_k,
        "Mean_F1_CV": mean_f1_optuna,
        "STD_F1_CV": std_f1_optuna,
        "Features_Used": ", ".join(selected_features_optuna)
    })
    
    print(f"[OPTUNA K] Najlepsze K={best_k}. F1-score: {mean_f1_optuna:.4f} +/- {std_f1_optuna:.4f}")
    
    
    df_results = pd.DataFrame(results)
    df_results['Mean_F1_CV_formatted'] = df_results.apply(
        lambda row: f"{row['Mean_F1_CV']:.4f} +/- {row['STD_F1_CV']:.4f}", axis=1
    )
    
    best_row = df_results.loc[df_results['Mean_F1_CV'].idxmax()]
    
    print("\n--- Podsumowanie Optymalizacji Cech ---")
    print(df_results[['Method', 'Features_Count', 'Mean_F1_CV_formatted']].to_markdown(index=False))
    
    print(f"\nNajlepszy wynik uzyskano metodą: {best_row['Method']} (F1-score: {best_row['Mean_F1_CV']:.4f}).")
    print(f"Wybrany zestaw cech ({best_row['Features_Count']} cech) zostanie użyty w optymalizacji hiperparametrów.")
    
    out_csv = OUT_DIR / "knn_feature_optimization_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"Wyniki zapisane do: {out_csv}")
    
    return best_row

if __name__ == "__main__":
    run_feature_optimization()