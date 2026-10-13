"""
Performance metrics for backtests.

Every metric here answers a different question:
- CAGR: how fast did the money grow?
- Sharpe: was the return worth the volatility?
- Max drawdown: what was the worst peak-to-trough loss along the way?
- Win rate: how often were trading days profitable?
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns):
    equity = (1 + returns).cumprod()
    n_years = len(returns) / TRADING_DAYS
    if n_years == 0 or equity.iloc[-1] <= 0:
        return np.nan
    return equity.iloc[-1] ** (1 / n_years) - 1


def annualized_vol(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, risk_free_annual=0.0):
    rf_daily = risk_free_annual / TRADING_DAYS
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0:
        return np.nan
    return excess.mean() / vol * np.sqrt(TRADING_DAYS)


def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def win_rate(returns):
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return (active > 0).mean()


def summary_metrics(returns):
    return pd.Series(
        {
            "CAGR": cagr(returns),
            "Annualized Vol": annualized_vol(returns),
            "Sharpe Ratio": sharpe_ratio(returns),
            "Max Drawdown": max_drawdown(returns),
            "Win Rate": win_rate(returns),
        }
    )
