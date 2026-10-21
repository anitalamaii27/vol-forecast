import numpy as np
from data import load_spy


def build_df():
    df = load_spy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["rv"] = df["log_ret"].rolling(5).std()
    df["target"] = df["rv"].shift(-5)
    df = df.dropna()
    split = int(len(df) * 0.8)
    return df, split