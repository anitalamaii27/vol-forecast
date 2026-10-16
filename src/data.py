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
    print(df.tail())
    print(df.shape)
