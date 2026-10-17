from windows import make_windows
from arch import arch_model
import numpy as np 
import pandas as pd 
import yfinance as yf 

def load_spy(start="2010-01-01"):
    df = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

if __name__ == "__main__":
    df = load_spy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rv"] = df["log_ret"].rolling(5).std()
    df["target"] = df["rv"].shift(-5)

    df = df.dropna()

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    pred = test["rv"]
    for w in [5, 20, 60, 120]:
        vol = df["log_ret"].rolling(w).std()
        p = vol.loc[test.index]
        r = np.sqrt(((test["target"] - p) ** 2).mean())
        print(w, "day baseline RMSE:", r)
    rmse = np.sqrt(((test["target"] - pred) ** 2).mean())
    print("train rows:", len(train), "test rows:", len(test))
    print("persistence RMSE:", rmse)


    r100 = df["log_ret"] * 100
    am = arch_model(r100, vol="Garch", p=1, q=1, mean="Zero")
    res = am.fit(last_obs=split, disp="off")
    fc = res.forecast(horizon=5, start=split, reindex=False)
    garch_pred = np.sqrt(fc.variance.mean(axis=1)) / 100
    g = np.sqrt(((test["target"] - garch_pred) ** 2).mean())
    print("GARCH RMSE:", g)

    X, y = make_windows(df["log_ret"].values, df["target"].values)
    print(X.shape, y.shape)

    cut = split - 19
    X_train, y_train = X[:cut - 5], y[:cut - 5]
    X_test, y_test = X[cut:], y[cut:]

    mu = X_train.mean()
    sd = X_train.std()
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd
    print(X_train.shape, X_test.shape)
