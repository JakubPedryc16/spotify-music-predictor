# Spotify Tracks Genre Classification

# Spotify Tracks Genre Classification

![Python](https://img.shields.io/badge/python-3.13.7-blue.svg?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-green.svg?style=flat)
![AutoGluon](https://img.shields.io/badge/AutoGluon-AutoML-blue.svg?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-Optimization-brightgreen.svg?style=flat)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg?style=flat)

This project focuses on the multi-class classification of music genres based on audio features using the Spotify Tracks Dataset. The system covers the full machine learning project lifecycle: from simulating missing data and preprocessing, through advanced optimization, to building complex model ensembles.

## Key Features

* **Multi-variant Data Pipeline**: Automated processing and comparison of 8 data variants (utilizing different imputation techniques: Mean, Median, KNN, Iterative; and scaling methods: MinMax, Standard).
* **Custom ML Engine**: Proprietary implementations of evaluation metrics (Accuracy, Precision, Recall, F1-Score) and a `manual_split` data partitioning algorithm with class stratification.
* **Advanced Optimization**:
    * **Feature Optimization**: Utilizing SelectKBest and Optuna for dimensionality reduction and key audio feature selection.
    * **Hyperparameters**: Comparison of Grid Search, Bayesian search (Optuna), and Genetic Algorithms (GASearchCV).
* **AutoML**: Integration with the AutoGluon library to achieve maximum classification precision.
* **Analysis and Visualization**: Generation of confusion matrices (full views and 8x8 slices), correlation heatmaps, and box plots.



## Dataset

The dataset used in this project is the **Spotify Tracks Dataset**, available on Kaggle:
[Link to Dataset](https://www.kaggle.com/datasets/maharshinaik/spotify-tracks-dataset)

The collection contains over 114,000 tracks, each described by 20 attributes. This project focuses on audio features that best describe the musical characteristics of different genres.

### Features Used:
* **Acoustic Features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo.
* **Metadata**: popularity, duration_ms.
* **Target Variable**: `track_genre` – the musical genre.

### Data Processing Characteristics:
* **Filtration**: Genres with an insufficient number of samples were removed to ensure the statistical reliability of the models.
* **Masking**: Missing data (NaN) was simulated at a 10% level to test the effectiveness of various imputation methods.



## Project Structure

```text
├── data/                         # Datasets (raw, preprocessed) and genre mappings
├── optimization_results/         # Logs and results from optimization processes
├── results/                      # Charts, confusion matrices (PNG/CSV), and final metrics
├── src/
│   ├── data/                     # Data loader and cleaning modules
│   ├── split/                    # Implementation of manual_split
│   ├── models/                   # Base models, Voting, Stacking, and metrics
│   ├── optimalisation/           # Optuna, Genetic Algorithm, and AutoGluon scripts
│   ├── prediction_visualisation/ # Charts, heatmaps, and model comparator
│   └── all_data_pipeline.py      # Automation of training for all data variants
├── main.py                       # Main script controlling the project
└── README.md
```

## Technologies

* **Language**: Python 3.x
* **Libraries**: scikit-learn, pandas, numpy, xgboost, lightgbm, autogluon, optuna, sklearn-genetic-opt
* **Visualization**: matplotlib, seaborn

## Methodology and Results

### Model Ranking (F1-Score Macro)

Based on the conducted tests and the final report, the models achieved the following results:

| Model | Optimization Method | F1-Score |
| :--- | :--- | :--- |
| **AutoGluon** | AutoML (Ensemble) | **0.9312** |
| **Stacking** | NB + KNN (Meta: LogReg) | **0.9274** |
| **kNN** | Genetic Algorithm / Optuna | **0.9234** |
| **XGBoost** | Grid Search / Default | **0.9105** |


### Experimental Conclusions

* **Preprocessing**: The best results were achieved using the data variant with Mean Imputation and MinMax Scaling.
* **Feature Selection**: The optimal number of features was determined to be 8. This reduction significantly accelerated computations with a minimal impact on precision.
* **Classification**: Genres with similar acoustic characteristics proved to be the most difficult to distinguish, as highlighted by the confusion matrices.


## Installation and Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script to start the pipeline:
   ```bash
   python main.py
   ```

## Data Preparation

The `main.py` script conducts experiments on 8 data variants, which differ in their missing value imputation techniques and scaling methods.

### Required Input Files
Before running the full process, the following files should be present in the `data/` folder:
* `data_auto_minmax.csv` / `data_auto_standardized.csv`
* `data_knn_minmax.csv` / `data_knn_standardized.csv`
* `data_mean_minmax.csv` / `data_mean_standardized.csv`
* `data_median_minmax.csv` / `data_median_standardized.csv`

### Generating Missing Data (Masking)
The `main.py` file includes commented-out logic to simulate missing data (10% of values) in the original file:
1. Uncomment the "masking" section in `main.py`.
2. Run the script to generate `data_original.csv` with the applied missing value mask.
3. Then, use the preprocessing scripts (e.g., from `src/data/`) to create the 8 variants listed above.

## Authors

* Jakub Pedryc
* Maciej Łabuz

---
Project completed as part of the **Advanced Machine Learning Methods** course.