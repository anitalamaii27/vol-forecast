import numpy as np
import torch
import torch.nn as nn
from model import VolLSTM


def fit_predict(X_train, y_train, X_test, seed, hidden=32, epochs=30):
    mu, sd = X_train.mean(), X_train.std()
    Xtr = torch.tensor((X_train - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor(y_train * 100, dtype=torch.float32)
    Xte = torch.tensor((X_test - mu) / sd, dtype=torch.float32)
    torch.manual_seed(seed)
    model = VolLSTM(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
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
        return model(Xte).numpy() / 100