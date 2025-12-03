import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from src.data.data_loader import load_data
from src.split.manual_split import manual_split
from src.models.base_models import get_base_models
from src.models.train import train_model
from src.models.ensemble import get_voting, get_stacking
from src.models.manual_metrics import accuracy, precision, recall, f1

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path("wyniki") / "confusion_data_mean_minmax"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(context="notebook", style="whitegrid")


def plot_confusion_matrix(cm: np.ndarray,
                          classes: np.ndarray,
                          title: str,
                          out_path: Path,
                          normalize: bool = False,
                          subset_labels: list = None) -> None:
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm_norm)

    labels = classes
    if subset_labels:
        indices = [np.where(classes == label)[0][0] for label in subset_labels]
        cm = cm[indices][:, indices]
        labels = subset_labels

    plt.figure(figsize=(12, 10))
    
    ax = sns.heatmap(
        cm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.xlabel("Przewidziany gatunek")
    plt.ylabel("Prawdziwy gatunek")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_confusion_analysis_mean_minmax():
    file_path = "data/data_mean_minmax.csv"
    map_path = "data/genre_map_mean.json"
    
    #N_ROWS_LIMIT = 10000

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku {file_path}. Upewnij się, że wcześniej wygenerowałeś data_mean_minmax.csv."
        )
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku {map_path}. Upewnij się, że wcześniej wygenerowałeś genre_map_mean.json."
        )

    print(f"Laduję dane z: {file_path}")
    df = load_data(file_path)
    
    with open(map_path, 'r', encoding='utf-8') as f:
        int_to_genre_map = {int(k): v for k, v in json.load(f).items()}
    
    target_col = "track_genre"
    if target_col not in df.columns:
        raise ValueError(f"W danych brakuje kolumny docelowej '{target_col}'.")

    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df[target_col].isin(valid_classes)].copy()

    y = df[target_col].to_numpy()
    
    unique_classes_int = sorted(np.unique(y))
    genre_labels = np.array([int_to_genre_map[i] for i in unique_classes_int])

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    X = df[numeric_cols].drop(target_col, axis=1, errors='ignore').to_numpy()

    X_train, X_test, y_train, y_test = manual_split(
        X, y, random_state=RANDOM_SEED
    )

    base_models = get_base_models()
    trained_models = {}

    print("\nTrenuję modele bazowe...")
    for name, model in base_models.items():
        print(f" - {name}")
        trained_models[name] = train_model(model, X_train, y_train)

    simple_models_v2 = {
        "nb": trained_models["nb"],
        "knn": trained_models["knn"],
    }

    print("\nTrenuję modele zespołowe (V2: nb + knn)...")
    voting_model_v2 = get_voting(simple_models_v2)
    voting_model_v2.fit(X_train, y_train)

    stacking_model_v2 = get_stacking(simple_models_v2, y_train)
    stacking_model_v2.fit(X_train, y_train)

    models_to_evaluate = {
        **trained_models,
        "voting_V2_nb_knn": voting_model_v2,
        "stacking_V2_nb_knn": stacking_model_v2,
    }

    excluded_models = ["nb", "perceptron"]

    interesting_models = [
        "logreg",
        "tree",
        "svm",
        "knn",
        "xgb",
        "lgbm",
        "voting_V2_nb_knn",
        "stacking_V2_nb_knn",
    ]

    results = []
    
    subset_labels_8x8 = genre_labels[:8].tolist()

    print("\nGeneruję macierze pomyłek...")
    for name, model in models_to_evaluate.items():
        if name in excluded_models:
            continue
        if name not in interesting_models:
            continue

        print(f"\n=== Model: {name} ===")

        y_pred_encoded = model.predict(X_test)
        
        acc = accuracy(y_test, y_pred_encoded)
        prec = precision(y_test, y_pred_encoded)
        rec = recall(y_test, y_pred_encoded)
        f1_score = f1(y_test, y_pred_encoded)
        
        print(f" accuracy:  {acc:.4f}")
        print(f" precision: {prec:.4f}")
        print(f" recall:    {rec:.4f}")
        print(f" f1:        {f1_score:.4f}")

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1_score,
            }
        )
        
        cm = confusion_matrix(y_test, y_pred_encoded, labels=unique_classes_int) 
        
        # =================================================================
        # ZAPIS MACIERZY PUMYŁEK JAKO CSV (NOWA FUNKCJONALNOŚĆ)
        # =================================================================
        cm_df = pd.DataFrame(cm, index=genre_labels, columns=genre_labels)
        out_cm_csv = OUT_DIR / f"cm_{name}_full_raw.csv"
        cm_df.index.name = "Prawdziwy Gatunek"
        cm_df.columns.name = "Przewidziany Gatunek"
        cm_df.to_csv(out_cm_csv)
        print(f"Macierz pomyłek zapisana jako: {out_cm_csv}")
        # =================================================================

        out_raw = OUT_DIR / f"cm_{name}_full_raw.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (liczby, wszystkie klasy) – {name}",
            out_path=out_raw,
            normalize=False,
        )

        out_norm_full = OUT_DIR / f"cm_{name}_full_normalized.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (procenty, wszystkie klasy) – {name}",
            out_path=out_norm_full,
            normalize=True,
        )

        out_subset = OUT_DIR / f"cm_{name}_subset_8x8_raw.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (liczby, wycinek 8x8) – {name}",
            out_path=out_subset,
            normalize=False,
            subset_labels=subset_labels_8x8,
        )
        
        out_subset_norm = OUT_DIR / f"cm_{name}_subset_8x8_normalized.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (procenty, wycinek 8x8) – {name}",
            out_path=out_subset_norm,
            normalize=True,
            subset_labels=subset_labels_8x8,
        )

    if results:
        df_results = pd.DataFrame(results)
        out_csv = OUT_DIR / "confusion_models_metrics_mean_minmax.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"\nMetryki modeli zapisane do: {out_csv}")

    print(
        "\nGotowe! Pliki zapisane w folderze:\n"
        f"  {OUT_DIR}\n"
        "Macierze pomyłek zapisane jako cm_<model>_full_raw.csv oraz pliki graficzne."
    )


if __name__ == "__main__":
    run_confusion_analysis_mean_minmax()