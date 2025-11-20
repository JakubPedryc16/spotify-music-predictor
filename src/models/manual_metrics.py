import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        precisions.append(tp / (tp + fp + 1e-9))
    return np.mean(precisions)

def recall(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        recalls.append(tp / (tp + fn + 1e-9))
    return np.mean(recalls)

def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    rc = recall(y_true, y_pred)
    return 2 * pr * rc / (pr + rc + 1e-9)
