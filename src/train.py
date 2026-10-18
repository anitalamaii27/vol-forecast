import numpy as np
import torch
import torch.nn as nn
from data import load_spy
from windows import make_windows
from model import VolLSTM

torch.manual_seed(0)

df = load_spy()
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["rv"] = df["log_ret"].rolling(5).std()
df["target"] = df["rv"].shift(-5)
df = df.dropna()
split = int(len(df) * 0.8)

X, y = make_windows(df["log_ret"].values, df["target"].values)
cut = split - 19
X_train, y_train = X[:cut - 5], y[:cut - 5]
X_test, y_test = X[cut:], y[cut:]

mu, sd = X_train.mean(), X_train.std()
X_train = (X_train - mu) / sd
X_test = (X_test - mu) / sd

Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train * 100, dtype=torch.float32)
Xte = torch.tensor(X_test, dtype=torch.float32)

model = VolLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(30):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy() / 100
    rmse = np.sqrt(((y_test - pred) ** 2).mean())
    print(epoch, "test RMSE:", rmse)