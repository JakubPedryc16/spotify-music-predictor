# graphs.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- analiza histogramów ---
def analyze_histograms(df, cols, out_csv="wyniki/hist_summary.csv"):
    rows = []
    for col in cols:
        data = df[col].dropna()
        if data.empty:
            continue
        if col == "duration_min" and data.max() > 20:
            data = data.clip(upper=np.percentile(data, 99))
        rows.append({
            "feature": col,
            "count": int(data.count()),
            "mean": round(data.mean(), 4),
            "median": round(data.median(), 4),
            "std": round(data.std(), 4),
            "min": round(data.min(), 4),
            "p25": round(np.percentile(data, 25), 4),
            "p75": round(np.percentile(data, 75), 4),
            "max": round(data.max(), 4),
            "skew": round(data.skew(), 4),
            "kurtosis": round(data.kurtosis(), 4)
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    print("\n✅ Analiza histogramów (opis statystyczny):")
    print(summary)
    print(f"\nZapisano: {out_csv}")

# --- główna funkcja ---
def run_analysis(show=False):
    df = pd.read_csv("dataset_imputed.csv")
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000

    Path("wyniki").mkdir(exist_ok=True)

    # --- ANALIZA (Pandas) ---
    print("\n==== ANALIZA DANYCH (PANDAS) ====")

    print("\nTop 10 gatunków wg średniej popularności:")
    top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(10)
    print(top_genres)
    top_genres.to_csv("wyniki/top_genres.csv")

    print("\nTop 10 najczęstszych artystów:")
    top_artists = df["artists"].dropna().str.split(";").explode().str.strip().value_counts().head(10)
    print(top_artists)
    top_artists.to_csv("wyniki/top_artists.csv")

    print("\nPorównanie popularności: explicit vs non-explicit:")
    explicit_stats = df.groupby("explicit")["popularity"].mean()
    print(explicit_stats)
    explicit_stats.to_csv("wyniki/explicit_vs_popularity.csv")

    # --- ZAawansowany wykres ---
    print("\n==== WYKRES SCATTER ====")
    scatter_df = df[["energy", "danceability", "popularity", "loudness"]].dropna()
    scatter_df = scatter_df.sample(min(40000, len(scatter_df)), random_state=42)

    sizes = 30 + (scatter_df["loudness"] - scatter_df["loudness"].min()) * 5

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        scatter_df["energy"], scatter_df["danceability"],
        c=scatter_df["popularity"], s=sizes, cmap="viridis", alpha=0.5
    )
    plt.colorbar(scatter, label="Popularity")
    plt.xlabel("Energy")
    plt.ylabel("Danceability")
    plt.title("Energy vs Danceability (kolor=popularność, wielkość=loudness)")
    plt.savefig("wyniki/advanced_scatter.png")
    if show:
        plt.show()
    plt.close()

    # --- Histogramy ---
    print("\n==== HISTOGRAMY ====")
    hist_cols = ["popularity", "tempo", "energy", "danceability", "duration_min", "loudness"]
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(hist_cols):
        plt.subplot(2, 3, i + 1)
        plt.hist(df[col].dropna(), bins=40, color="blue", alpha=0.7)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Ilość wystąpień")

    plt.tight_layout()
    plt.savefig("wyniki/histograms.png")
    if show:
        plt.show()
    plt.close()

    # --- ANALIZA histogramów ---
    analyze_histograms(df, hist_cols)

    print("\n✅ Analiza zakończona. Wyniki zapisano w folderze 'wyniki/'.")


if __name__ == "__main__":
    run_analysis(show=True)
