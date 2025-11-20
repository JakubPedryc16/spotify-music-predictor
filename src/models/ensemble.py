from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def get_voting(models_dict):
    models = [(name, mdl) for name, mdl in models_dict.items()]
    return VotingClassifier(models, voting="hard")

import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_stacking(models_dict, y_train):
    models = [(name, mdl) for name, mdl in models_dict.items()]
    
    min_samples_per_class = np.min(np.bincount(y_train))
    n_splits = min(5, min_samples_per_class) 
    
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)
    
    return StackingClassifier(
        estimators=models,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=cv
    )
