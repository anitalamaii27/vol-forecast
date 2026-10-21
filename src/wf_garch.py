import numpy as np
from arch import arch_model
from features import build_df
from walkforward import make_folds

df, split = build_df()
r100 = df["log_ret"] * 100

preds = []
for start, end in make_folds(split, len(df)):
    am = arch_model(r100, vol="GARCH", p=1, q=1, mean="Zero")
    res = am.fit(last_obs=start, disp="off")
    fc = res.forecast(horizon=5, start=start, reindex=False)
    p = np.sqrt(fc.variance.mean(axis=1)).values[: end - start] / 100
    preds.append(p)

garch_wf = np.concatenate(preds)
y = df["target"].values[split:]
print(len(garch_wf), len(y))
print("GARCH walk-forward RMSE:", np.sqrt(((y - garch_wf) ** 2).mean()))