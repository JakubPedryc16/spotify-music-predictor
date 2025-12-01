from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_base_models():
    seed = 42
    return {
        "logreg": LogisticRegression(random_state=seed),
        "tree": DecisionTreeClassifier(random_state=seed),
        "svm": SVC(random_state=seed),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier(random_state=seed),
        "lgbm": LGBMClassifier(random_state=seed),
        "nb": GaussianNB(),
        "perceptron": Perceptron(random_state=seed),
    }
