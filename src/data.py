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
