import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

from src.data.data_loader import load_data

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FILE_PATH = "data/data_mean_minmax.csv"
OUT_DIR = Path("optimization_results") / "auto_ml_autogluon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
BEST_K_FEATURES = 8

def prepare_data_for_autogluon(file_path: str, n_splits: int, k_features: int) -> tuple[pd.DataFrame, str, list]:
    df = load_data(file_path)
    target_col = "track_genre"
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= n_splits].index
    df = df[df[target_col].isin(valid_classes)].copy()

    X_full_df = df[numeric_cols].drop(target_col, axis=1, errors='ignore')
    y = df[target_col]
    
    X = X_full_df.to_numpy()

    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_for_selection = X_full_df.reset_index(drop=True).to_numpy()
    y_for_selection = y.reset_index(drop=True)
    
    selector.fit(X_for_selection, y_for_selection) 
    
    selected_indices = selector.get_support(indices=True)
    feature_names = X_full_df.columns.tolist()
    selected_features = [feature_names[i] for i in selected_indices]
    
    X_opt_df = pd.DataFrame(selector.transform(X_for_selection), columns=selected_features)
    
    X_opt_df[target_col] = y_for_selection
    
    return X_opt_df, target_col, selected_features

def run_autogluon_search():
    print("--- Etap 4: Auto-ML - AutoGluon (Klasyfikacja Wieloklasowa) ---")
    
    df_opt, target_col, selected_features = prepare_data_for_autogluon(FILE_PATH, N_SPLITS, BEST_K_FEATURES)
    
    print(f"Dane przygotowane. Użyte cechy ({len(selected_features)}): {selected_features}")
    
    save_path = str(OUT_DIR / 'Autogluon_models')
    
    predictor = TabularPredictor(
        label=target_col, 
        eval_metric='f1_macro',
        path=save_path, 
        verbosity=2, 
        problem_type='multiclass',
    )

    print("\nRozpoczęcie treningu AutoGluon (Z użyciem 8 cech, optymalizacja pod F1-macro)...")
    
    start_time = time()
    
    predictor.fit(
        train_data=df_opt, 
        presets='medium_quality_faster_train', 
        time_limit=3600,
    )
    
    end_time = time()

    evaluation = predictor.evaluate(df_opt, silent=True)
    final_score = evaluation['f1_macro']
    
    leaderboard = predictor.leaderboard(df_opt, silent=True)
    best_model_name = leaderboard.iloc[0]['model']
    
    df_results = pd.DataFrame({
        "Method": ["AutoGluon Auto-ML"],
        "Time_seconds": [end_time - start_time],
        "Best_F1_CV": [final_score],
        "Best_Model": [best_model_name],
    })
    
    out_csv = OUT_DIR / "autogluon_results.csv"
    df_results.to_csv(out_csv, index=False)
    
    print("\n--- Wynik końcowy AutoGluon ---")
    print(f"Czas przeszukiwania: {end_time - start_time:.2f} s")
    print(f"Najlepszy F1-score (macro): {final_score:.4f}")
    print(f"Najlepszy model (z leaderboard): {best_model_name}")
    print(f"Wyniki zapisane do: {out_csv}")
    
    return df_results.iloc[0]

if __name__ == "__main__":
    run_autogluon_search()