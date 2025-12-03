import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from src.data.data_loader import load_data
from src.models.manual_metrics import accuracy, precision, recall, f1

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path("wyniki") / "cross_validation_manual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5

def run_knn_manual_cross_validation():
    print("--- Ręczna implementacja 5-krotnej Walidacji Krzyżowej dla kNN ---")
    
    file_path = "data/data_mean_minmax.csv"
    map_path = "data/genre_map_mean.json"

    df = load_data(file_path)

    with open(map_path, 'r', encoding='utf-8') as f:
        int_to_genre_map = {int(k): v for k, v in json.load(f).items()}

    target_col = "track_genre"
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= N_SPLITS].index
    df = df[df[target_col].isin(valid_classes)].copy()

    y = df[target_col].to_numpy()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    X = df[numeric_cols].drop(target_col, axis=1, errors='ignore').to_numpy()

    print(f"Dane: {X.shape[0]} próbek, {len(np.unique(y))} klas.")
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    cv_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
    
    print(f"\nRozpoczęcie walidacji {N_SPLITS}-krotnej...")
    
    for fold, (train_index, val_index) in enumerate(cv_splitter.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_val)

        acc_scores.append(accuracy(y_val, y_pred))
        prec_scores.append(precision(y_val, y_pred))
        rec_scores.append(recall(y_val, y_pred))
        f1_scores.append(f1(y_val, y_pred))
        
        print(f"Fold {fold + 1}/{N_SPLITS} - Accuracy: {acc_scores[-1]:.4f}")

    all_scores = {
        'Fold': [f'Fold {i+1}' for i in range(N_SPLITS)],
        'Accuracy': acc_scores,
        'Precision': prec_scores,
        'Recall': rec_scores,
        'F1-Score': f1_scores
    }
    df_scores = pd.DataFrame(all_scores)

    mean_results = df_scores[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
    std_results = df_scores[['Accuracy', 'Precision', 'Recall', 'F1-Score']].std()

    df_final = pd.DataFrame({
        'Model': ['kNN (CV)'],
        'Accuracy (Średnia +/- STD)': [f"{mean_results['Accuracy']:.4f} +/- {std_results['Accuracy']:.4f}"],
        'Precision (Średnia +/- STD)': [f"{mean_results['Precision']:.4f} +/- {std_results['Precision']:.4f}"],
        'Recall (Średnia +/- STD)': [f"{mean_results['Recall']:.4f} +/- {std_results['Recall']:.4f}"],
        'F1-Score (Średnia +/- STD)': [f"{mean_results['F1-Score']:.4f} +/- {std_results['F1-Score']:.4f}"]
    })
    
    out_csv = OUT_DIR / "knn_manual_cross_validation_final_result.csv"
    df_final.to_csv(out_csv, index=False)

    print("\n--- Wynik końcowy Walidacji Krzyżowej (kNN) ---")
    print(df_final.to_markdown(index=False))
    print(f"\nUśredniony wynik zapisany do: {out_csv}")
    
    return df_final.iloc[0]

if __name__ == "__main__":
    cv_result = run_knn_manual_cross_validation()