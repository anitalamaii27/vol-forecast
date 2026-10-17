import numpy as np

def make_windows(returns, targets, lookback=20):
    X, y = [], []
    for i in range(lookback - 1, len(returns)):
        X.append(returns[i - lookback + 1:i + 1])
        y.append(targets[i])
    X = np.array(X)[:, :, None]
    return X, np.array(y)