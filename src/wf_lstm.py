import numpy as np
from features import build_df
from windows import make_windows
from walkforward import make_folds
from lstm_lib import fit_predict

df, split = build_df()
X, y = make_windows(df["log_ret"].values, df["target"].values, 20)
off = 19  # window k ends on day k + 19


def run(seed):
    preds = []
    for start, end in make_folds(split, len(df)):
        n_train = start - off - 5
        p = fit_predict(X[:n_train], y[:n_train], X[start - off:end - off], seed)
        preds.append(p)
    return np.concatenate(preds)


y_test = df["target"].values[split:]
scores = []
for seed in range(3):
    pred = run(seed)
    scores.append(np.sqrt(((y_test - pred) ** 2).mean()))
    print(seed, len(pred), len(y_test), scores[-1])
print("mean:", np.mean(scores), "std:", np.std(scores))