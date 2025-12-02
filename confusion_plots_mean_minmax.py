import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
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


def _numeric_cols(df: pd.DataFrame) -> list:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop in ["id"]:
        if drop in cols:
            cols.remove(drop)
    return cols


def plot_confusion_matrix(cm: np.ndarray,
                          classes: np.ndarray,
                          title: str,
                          out_path: Path,
                          normalize: bool = False) -> None:
    """
    Rysuje macierz pomyłek z nazwami gatunków na osiach.
    """
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Przewidziany gatunek")
    plt.ylabel("Prawdziwy gatunek")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_confusion_analysis_mean_minmax():
    file_path = "data/data_mean_minmax.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku {file_path}. "
            f"Upewnij się, że wcześniej wygenerowałeś data_mean_minmax.csv."
        )

    print(f"📂 Ładuję dane z: {file_path}")
    df = load_data(file_path)

    target_col = "track_genre"
    if target_col not in df.columns:
        raise ValueError(f"W danych brakuje kolumny docelowej '{target_col}'.")

    # usunięcie rzadkich klas (tak jak w all_data_pipeline)
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df[target_col].isin(valid_classes)].copy()

    # LabelEncoder na gatunek
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    # cechy numeryczne
    numeric_cols = _numeric_cols(df)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols].to_numpy()

    # ręczny podział train/test (manual_split jak w pipeline)
    X_train, X_test, y_train, y_test = manual_split(
        X, y, random_state=RANDOM_SEED
    )

    # modele bazowe
    base_models = get_base_models()
    trained_models = {}

    print("\n🚂 Trenuję modele bazowe...")
    for name, model in base_models.items():
        print(f"  - {name}")
        trained_models[name] = train_model(model, X_train, y_train)

    # proste zestawy do voting & stacking (V2: nb + knn)
    simple_models_v2 = {
        "nb": trained_models["nb"],
        "knn": trained_models["knn"],
    }

    print("\n🧮 Trenuję modele zespołowe (V2: nb + knn)...")
    voting_model_v2 = get_voting(simple_models_v2)
    voting_model_v2.fit(X_train, y_train)

    stacking_model_v2 = get_stacking(simple_models_v2, y_train)
    stacking_model_v2.fit(X_train, y_train)

    # pełny słownik modeli do ewaluacji
    models_to_evaluate = {
        **trained_models,
        "voting_V2_nb_knn": voting_model_v2,
        "stacking_V2_nb_knn": stacking_model_v2,
    }

    # nie będziemy robić macierzy dla nb i perceptron (jak w pipeline)
    excluded_models = ["nb", "perceptron"]

    # Możesz zawęzić listę modeli do ładnych macierzy:
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

    genre_labels = le.classes_

    print("\n📊 Generuję macierze pomyłek...")
    for name, model in models_to_evaluate.items():
        if name in excluded_models:
            continue
        if name not in interesting_models:
            continue

        print(f"\n=== Model: {name} ===")

        # predykcje na zbiorze testowym
        y_pred_encoded = model.predict(X_test)

        y_true = le.inverse_transform(y_test)
        y_pred = le.inverse_transform(y_pred_encoded)

        # metryki (ręczne implementacje)
        acc = accuracy(y_true, y_pred)
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f1_score = f1(y_true, y_pred)

        print(f"  accuracy:  {acc:.4f}")
        print(f"  precision: {prec:.4f}")
        print(f"  recall:    {rec:.4f}")
        print(f"  f1:        {f1_score:.4f}")

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1_score,
            }
        )

        # macierz pomyłek (liczby)
        cm = confusion_matrix(y_true, y_pred, labels=genre_labels)

        # surowa macierz
        out_raw = OUT_DIR / f"cm_{name}_raw.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (liczby) – {name}",
            out_path=out_raw,
            normalize=False,
        )

        # znormalizowana po wierszach (procenty)
        out_norm = OUT_DIR / f"cm_{name}_normalized.png"
        plot_confusion_matrix(
            cm,
            classes=genre_labels,
            title=f"Macierz pomyłek (procenty) – {name}",
            out_path=out_norm,
            normalize=True,
        )

    # opcjonalnie: zapis tabelki z metrykami dla tych modeli
    if results:
        df_results = pd.DataFrame(results)
        out_csv = OUT_DIR / "confusion_models_metrics_mean_minmax.csv"
        df_results.to_csv(out_csv, index=False)
        print(f"\n📄 Metryki modeli zapisane do: {out_csv}")

    print(
        "\n✅ Gotowe! Macierze pomyłek zapisane w folderze:\n"
        f"   {OUT_DIR}\n"
        "Pliki: cm_<model>_raw.png, cm_<model>_normalized.png"
    )


if __name__ == "__main__":
    run_confusion_analysis_mean_minmax()
