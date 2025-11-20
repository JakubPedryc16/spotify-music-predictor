from sklearn.preprocessing import LabelEncoder
from src.data.data_loader import load_data
from src.split.manual_split import manual_split
from src.models.base_models import get_base_models
from src.models.train import train_model
from src.models.ensemble import get_voting, get_stacking
from src.models.manual_metrics import accuracy, precision, recall, f1
import numpy as np
import pandas as pd

def run_pipeline(file_path):
    n_rows=5000
    df = load_data(file_path, n_rows=n_rows)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    target_col = 'track_genre'

    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df[target_col].isin(valid_classes)]

    le = LabelEncoder()
    y = le.fit_transform(df[target_col])

    X = df[numeric_cols].drop(target_col, axis=1).to_numpy()

    X_train, X_test, y_train, y_test = manual_split(X, y)
    
    base_models = get_base_models()
    trained_models = {}
    for name, model in base_models.items():
        trained_models[name] = train_model(model, X_train, y_train)
    
    voting_model = get_voting(trained_models)
    voting_model.fit(X_train, y_train)
    
    stacking_model = get_stacking(trained_models, y_train)
    stacking_model.fit(X_train, y_train)
    
    results = []

    for name, model in {
        **trained_models,
        "voting": voting_model,
        "stacking": stacking_model
    }.items():

        y_pred_encoded = model.predict(X_test)

        y_pred = le.inverse_transform(y_pred_encoded)
        y_true = le.inverse_transform(y_test)

        results.append({
            "model": name,
            "accuracy": accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1": f1(y_true, y_pred)
        })

    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df
