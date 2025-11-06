# graphs.py  — tylko wykresy dla 8 zbiorów (4 imputacje × 2 skalowania)
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional



# --- ustawienia globalne
sns.set(context="notebook", style="whitegrid")
OUT_DIR = Path("wyniki")


def _ensure_duration_min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration_ms" in df.columns and "duration_min" not in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000.0
    return df


def _numeric_cols(df: pd.DataFrame) -> list:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # opcjonalnie usuń identyfikatory
    for drop in ["id"]:
        if drop in cols:
            cols.remove(drop)
    return cols


def plot_pairplot(df: pd.DataFrame, out_path: Path, max_rows: int = 2000) -> None:
    """Pairplot z próbkowaniem (żeby nie zalać wykresu)."""
    df_num = df[_numeric_cols(df)].dropna()
    if df_num.empty:
        return
    if len(df_num) > max_rows:
        df_num = df_num.sample(max_rows, random_state=42)
    g = sns.pairplot(df_num, corner=True, diag_kind="hist", plot_kws=dict(alpha=0.5, s=10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=150)
    plt.close("all")


def plot_corr_heatmaps(df: pd.DataFrame, out_base: Path) -> None:
    """Heatmapy korelacji (Pearson i Spearman)."""
    df_num = df[_numeric_cols(df)]
    if df_num.empty:
        return

    for method in ["pearson", "spearman"]:
        corr = df_num.corr(method=method)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False, cbar=True)
        plt.title(f"Macierz korelacji ({method.capitalize()})")
        plt.tight_layout()
        out_file = out_base.parent / f"{out_base.name}_{method}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        plt.close()


def plot_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    """Boxploty wybranych cech (wizualna analiza odstających)."""
    df = df.copy()
    df = _ensure_duration_min(df)
    selected = [
        "popularity", "tempo", "energy", "danceability",
        "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "duration_min"
    ]
    selected = [c for c in selected if c in df.columns]
    if not selected:
        return

    n = len(selected)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(5 * cols, 3.8 * rows))
    for i, col in enumerate(selected, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(col)
        plt.xlabel("")
        plt.grid(True, axis="x", alpha=0.2)
    plt.suptitle("Boxploty wybranych cech", y=1.02, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _load_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"⏭️  Pomijam — brak pliku: {path}")
        return None
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    df = _ensure_duration_min(df)
    return df


def run_analysis(show: bool = False) -> None:
    """
    Generuje wykresy dla 8 zbiorów:
      data_mean.csv, data_median.csv, data_knn.csv, data_auto.csv
      + ich wersje _minmax.csv i _standardized.csv
    Pliki są spodziewane w katalogu ./data
    """
    OUT_DIR.mkdir(exist_ok=True)

    base_names = ["data_mean", "data_median", "data_knn", "data_auto"]
    variants = ["", "_minmax", "_standardized"]  # "" to wersja bez skalowania (po imputacji)

    for base in base_names:
        for var in variants:
            in_path = Path("data") / f"{base}{var}.csv"
            df = _load_df(in_path)
            if df is None:
                continue

            tag = f"{base}{var}" if var else base
            save_dir = OUT_DIR / tag
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📊 Generuję wykresy dla: {in_path}")

            # 1) Pairplot
            plot_pairplot(df, save_dir / "pairplot.png")

            # 2) Heatmapy korelacji
            plot_corr_heatmaps(df, save_dir / "corr_heatmap")

            # 3) Boxploty
            plot_boxplots(df, save_dir / "boxplots.png")

            if show:
                # nic nie pokazujemy na żywo (ciężkie wykresy) – trzymamy tylko zapisy
                pass

    print("\n✅ Gotowe. Wszystkie wykresy zapisane w folderze 'wyniki/'.")
    print("Struktura: wyniki/<nazwa_zbioru>/{pairplot.png, corr_heatmap_*.png, boxplots.png}")


if __name__ == "__main__":
    run_analysis(show=False)
