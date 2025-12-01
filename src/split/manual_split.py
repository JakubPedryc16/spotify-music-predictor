import numpy as np

def manual_split(X, y, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    
    train_idx = []
    test_idx = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)

        test_size = int(len(cls_idx) * test_ratio)
        test_size = max(1, test_size)
        test_size = min(len(cls_idx) - 1, test_size)

        test_idx.extend(cls_idx[:test_size])
        train_idx.extend(cls_idx[test_size:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]