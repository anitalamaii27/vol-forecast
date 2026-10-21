# Forecasting SPY Volatility: LSTM vs GARCH

Forecasts the next 5 days of realized volatility for SPY using a PyTorch LSTM, and compares it to GARCH(1,1) and simple rolling-average baselines. Evaluated with walk-forward validation.

## Setup
    pip install -r requirements.txt
    python src/data.py        # baselines and single-fit GARCH
    python src/wf_garch.py    # walk-forward GARCH
    python src/wf_lstm.py     # walk-forward LSTM
    python src/train.py       # single-split LSTM ablations

## Data and target
- Daily SPY prices from 2010 to today (yfinance)
- Target: standard deviation of log returns over the next 5 trading days
- Daily data only, so this is a proxy for realized volatility, not the intraday version
- Prices are re-downloaded on every run, so numbers shift slightly as new trading days arrive

## Walk-forward validation
- Test period is the last 20% of the data (844 days in the latest run)
- Split into 14 chunks of up to 63 trading days (13 full chunks plus a short last one)
- Before each chunk, both GARCH and the LSTM are refit on all data before that chunk (expanding window)

## Avoiding leakage
- Chronological order only, no shuffling across the train/test boundary
- 5-day purge gap, so no training label overlaps the chunk being tested
- LSTM scaler is fit on each chunk's training data only
- GARCH is fit on past returns only
- Each LSTM window contains only past returns

## Results (walk-forward, RMSE, lower is better)
| Model | RMSE |
|---|---|
| 5-day persistence | 0.005841 |
| 20-day rolling average | 0.005782 |
| GARCH(1,1), fit once | 0.005195 |
| GARCH(1,1), refit every 63 days | 0.005163 |
| LSTM (20-day lookback), retrained every 63 days, 3 seeds | 0.004658 +/- 0.000009 |

## Ablations (LSTM, single split, 5 seeds each)
These ran before walk-forward, on a single 80/20 split with data through mid October 2026 (843 test days). On that split, GARCH scored 0.005201 and the LSTM scored 0.004615.

| Setting | Mean RMSE | Std |
|---|---|---|
| Lookback 10 | 0.004701 | 0.000075 |
| Lookback 20 | 0.004615 | 0.000042 |
| Lookback 60 | 0.004698 | 0.000132 |
| Hidden 16 | 0.004667 | 0.000045 |
| Hidden 32 | 0.004615 | 0.000042 |
| Hidden 64 | 0.004628 | 0.000025 |

## Takeaways
The LSTM beat GARCH under walk-forward validation, 0.004658 vs 0.005163 RMSE. That is about 10% lower error. It was also about 20% better than the persistence baseline. The gap is much bigger than the seed to seed spread, but I only ran 3 seeds in walk-forward, so I treat that spread as a rough estimate.

Refitting GARCH every 63 days only helped a little (0.005195 to 0.005163). So GARCH was not losing because its parameters were stale. The LSTM result also held up when I retrained it on every chunk, which is a stricter test than a single split.

In the single-split ablations, every lookback (10, 20, 60 days) and hidden size (16, 32, 64) still beat GARCH. The differences between those settings were mostly within seed noise, so I can't say one is the best. 20 days and 32 units had the best means, and that is all I would claim.

## Limitations
- Only 3 seeds in the walk-forward LSTM run
- I have not yet checked how the LSTM does chunk by chunk, so a few large chunks could be driving the average
- Ablations were run on a single split, before walk-forward, on slightly older data
- Scored with RMSE only. A volatility metric like QLIKE could rank the models differently
- Baselines are GARCH(1,1) and rolling averages. I did not try stronger ones such as HAR-RV
- One asset (SPY) and daily data only
- This is a forecasting experiment, not a trading strategy