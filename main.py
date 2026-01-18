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

if __name__ == "__main__":
    # --- ETAP 0: Maskowanie danych (Opcjonalne) ---
    # Odkomentuj poniższy blok, aby wygenerować braki danych (NaN) w pliku oryginalnym
    """
    path_to_original = "data/data_original.csv"
    if os.path.exists(path_to_original):
        df = pd.read_csv(path_to_original)
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'id'})
        
        # Wybór kolumn do maskowania (wszystkie poza ID i cechą docelową)
        columns_to_mask = df.columns.difference(['id', 'track_genre'])
        
        # Tworzenie maski losowej - 10% wartości zostanie zastąpione przez NaN
        df_mask = pd.DataFrame(
            np.random.rand(*df[columns_to_mask].shape) < 0.1,
            columns=columns_to_mask
        )
        
        df[columns_to_mask] = df[columns_to_mask].mask(df_mask)
        df.to_csv(path_to_original, index=False)
        print("Pomyślnie nałożono maskę braków danych na plik oryginalny.")
    """

    # --- ETAP 1: Pipeline dla wariantów danych ---
    # Upewnij się, że pliki w data/ istnieją przed uruchomieniem
    data_variants = [
        "data_auto_minmax.csv", "data_auto_standardized.csv",
        "data_knn_minmax.csv", "data_knn_standardized.csv",
        "data_mean_minmax.csv", "data_mean_standardized.csv",
        "data_median_minmax.csv", "data_median_standardized.csv",
    ]
    
    base_dir = "data"
    print(f"Rozpoczynanie pipeline dla {len(data_variants)} wariantów...")

    for variant in data_variants:
        file_path = os.path.join(base_dir, variant)
        if os.path.exists(file_path):
            print(f"\n--- Przetwarzanie: {file_path} ---")
            try:
                run_pipeline(file_path)
            except Exception as e:
                print(f"Błąd podczas przetwarzania {file_path}: {e}")
        else:
            print(f"Pominięto {variant}: plik nie istnieje.")

    # --- ETAP 2: Optymalizacja i analizy ---
    print("\n--- Uruchamianie optymalizacji cech i parametrów ---")
    run_feature_optimization()
    run_grid_search()
    run_optuna_search() 
    run_genetic_search()
    
    print("\n--- Uruchamianie AutoGluon ---")
    run_autogluon_search()
    
    # --- ETAP 3: Wizualizacja ---
    print("\n--- Generowanie analiz końcowych ---")
    run_analysis()