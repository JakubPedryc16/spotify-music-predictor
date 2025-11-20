from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_base_models():
    return {
        "logreg": LogisticRegression(max_iter=5000, solver='lbfgs'),
        "tree": DecisionTreeClassifier(),
        "svm": SVC(probability=True),
        "knn": KNeighborsClassifier(),
        
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "lgbm": LGBMClassifier(
            min_child_samples=5,
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=300,
            min_split_gain=0.001,
            max_depth=4,  
        ),
    }
