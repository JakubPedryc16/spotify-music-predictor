from sklearn.preprocessing import LabelEncoder
from src.data.data_loader import load_data
from src.split.manual_split import manual_split
from src.models.base_models import get_base_models
from src.models.train import train_model
from src.models.ensemble import get_voting, get_stacking
from src.models.manual_metrics import accuracy, precision, recall, f1
import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_pipeline(file_path):
    file_base_name = os.path.basename(file_path).replace(".csv", "")
    output_file_name = f"results_combined_{file_base_name}.csv"


    df = load_data(file_path)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    target_col = 'track_genre'

    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df[target_col].isin(valid_classes)]

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    X = df[numeric_cols].drop(target_col, axis=1).to_numpy()

    X_train, X_test, y_train, y_test = manual_split(X, y, random_state=RANDOM_SEED)

    base_models = get_base_models()
    trained_models = {}
    
    for name, model in base_models.items():
        trained_models[name] = train_model(model, X_train, y_train)

    simple_models_v1 = {
        "nb": trained_models["nb"],
        "perceptron": trained_models["perceptron"]
    }
    simple_models_v2 = {
        "nb": trained_models["nb"],
        "knn": trained_models["knn"]
    }

    #V1 gives terrible results and can be commented I guess
    voting_model_v1 = get_voting(simple_models_v1)
    voting_model_v1.fit(X_train, y_train)
    stacking_model_v1 = get_stacking(simple_models_v1, y_train)
    stacking_model_v1.fit(X_train, y_train)

    voting_model_v2 = get_voting(simple_models_v2)
    voting_model_v2.fit(X_train, y_train)
    stacking_model_v2 = get_stacking(simple_models_v2, y_train)
    stacking_model_v2.fit(X_train, y_train)

    results = []
    
    models_to_evaluate = {
        **trained_models,
        "voting_V1_nb_perceptron": voting_model_v1,
        "stacking_V1_nb_perceptron": stacking_model_v1,
        "voting_V2_nb_knn": voting_model_v2,
        "stacking_V2_nb_knn": stacking_model_v2,
    }
    
    #These are only for the stack and vote
    excluded_models = ["nb", "perceptron"] 

    for name, model in models_to_evaluate.items():
        if name in excluded_models:
            continue
            
        y_pred_encoded = model.predict(X_test)

        y_pred = le.inverse_transform(y_pred_encoded)
        y_true = le.inverse_transform(y_test)

        results.append({
            "model": name,
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1": f1(y_true, y_pred),
            "data_variant": file_base_name
        })

    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_file_name, index=False)
    print(f"Wyniki (Wariant Połączony) dla {file_base_name} zapisane do {output_file_name}")
    print(results_df)

    return results_df