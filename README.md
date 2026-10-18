# Forecasting SPY Volatility: LSTM vs GARCH

Forecasts the next 5 days of realized volatility for SPY using a PyTorch LSTM, and compares it to GARCH(1,1) and simple rolling-average baselines.

## Setup
    pip install -r requirements.txt
    python src/train.py

## Data and target
- Daily SPY prices from 2010 to today (yfinance)
- Target: standard deviation of log returns over the next 5 trading days
- Daily data only, so this is a proxy for realized volatility, not the intraday version

## Avoiding leakage
- Chronological split: first 80% train, last 20% test (843 days)
- 5-day purge gap so no training label overlaps the test period
- Scaler fit on training data only
- Windows contain only past returns

## Results (test set, RMSE, lower is better)
| Model | RMSE |
|---|---|
| 5-day persistence | 0.005845 |
| 20-day rolling average | 0.005787 |
| GARCH(1,1) | 0.005201 |
| LSTM (20-day lookback, 5 seeds) | 0.004615 +/- 0.000042 |

## Ablations (LSTM, 5 seeds each)
| Setting | Mean RMSE | Std |
|---|---|---|
| Lookback 10 | 0.004701 | 0.000075 |
| Lookback 20 | 0.004615 | 0.000042 |
| Lookback 60 | 0.004698 | 0.000132 |
| Hidden 16 | 0.004667 | 0.000045 |
| Hidden 32 | 0.004615 | 0.000042 |
| Hidden 64 | 0.004628 | 0.000025 |

## Takeaways
The LSTM beat GARCH on the test set, 0.004615 vs 0.005201 RMSE. That is about 11% lower error. It also beat the simple baselines by a wider margin, about 21% better than persistence. Across 5 seeds the std was only 0.000042, so the gap to GARCH is roughly 14 times the seed noise. I don't think it's a lucky run.

I changed the lookback (10, 20, 60 days) and the hidden size (16, 32, 64). Every version still beat GARCH. But the differences between those settings were mostly within seed noise, so I can't say one is the best. 20 days and 32 units had the best means, and that's all I'd claim.

This is one chronological split, not walk-forward, and GARCH was fit once instead of refit over time. So I'd treat the result as promising, not proven. Walk-forward validation is the next thing to add.

## Limitations
- Single train/test split, not walk-forward yet
- GARCH is fit once and not refit over time
- Differences between lookbacks and hidden sizes are within seed noise

