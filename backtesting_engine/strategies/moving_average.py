"""
Sample strategies for the backtesting engine.

Each strategy takes a price Series and returns a position Series:
+1 = long, 0 = flat, -1 = short. The engine handles execution delay
and transaction costs, so strategies stay pure signal logic.
"""

import numpy as np
import pandas as pd


def ma_crossover(prices, fast=20, slow=50):
    """Long when the fast moving average is above the slow one, else flat."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)


def momentum(prices, lookback=60):
    """Long when trailing return over the lookback window is positive."""
    trailing = prices.pct_change(lookback)
    return (trailing > 0).astype(float)


def mean_reversion_zscore(prices, window=20, entry_z=1.0):
    """
    Contrarian strategy: buy when price is more than entry_z standard
    deviations below its rolling mean, short when above. Flat in between.
    """
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std
    pos = pd.Series(0.0, index=prices.index)
    pos[z < -entry_z] = 1.0
    pos[z > entry_z] = -1.0
    return pos
