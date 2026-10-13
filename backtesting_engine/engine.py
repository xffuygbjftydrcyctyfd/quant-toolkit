"""
Backtesting engine.

Takes a price series and a strategy function, simulates the strategy
bar-by-bar, and reports risk-adjusted performance metrics.

A strategy is any function that maps a price DataFrame to a Series of
target positions (+1 = fully long, 0 = flat, -1 = fully short). Positions
are applied with a one-bar delay to avoid lookahead bias: the signal you
compute on today's close is traded at tomorrow's close.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.data_utils import fetch_prices, daily_returns  # noqa: E402
from backtesting_engine.metrics import summary_metrics      # noqa: E402


def run_backtest(prices, strategy, cost_bps=1.0):
    """
    Simulate a strategy on a single-asset price series.

    Parameters
    ----------
    prices : pd.Series
        Adjusted close prices.
    strategy : callable
        Function taking the price Series and returning a position Series
        (+1 long, 0 flat, -1 short) on the same index.
    cost_bps : float
        Round-trip transaction cost in basis points, charged whenever the
        position changes. Ignoring costs is the classic way backtests lie.

    Returns
    -------
    dict with equity curve, strategy returns, positions, and metrics.
    """
    prices = prices.dropna()
    positions = strategy(prices).reindex(prices.index).fillna(0)

    # Shift positions by one bar: today's signal earns tomorrow's return.
    held = positions.shift(1).fillna(0)

    asset_rets = prices.pct_change().fillna(0)
    gross = held * asset_rets

    # Transaction costs when the position changes
    turnover = positions.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 10_000)

    strat_rets = gross - costs
    equity = (1 + strat_rets).cumprod()

    return {
        "equity": equity,
        "returns": strat_rets,
        "positions": positions,
        "metrics": summary_metrics(strat_rets),
    }


def compare_to_buy_and_hold(prices, strategy, cost_bps=1.0):
    """Run the strategy and print metrics next to buy-and-hold."""
    result = run_backtest(prices, strategy, cost_bps)
    bh_rets = daily_returns(prices)
    bh_metrics = summary_metrics(bh_rets)

    rows = pd.DataFrame(
        {"Strategy": result["metrics"], "Buy & Hold": bh_metrics}
    )
    return result, rows


if __name__ == "__main__":
    from backtesting_engine.strategies.moving_average import ma_crossover

    prices = fetch_prices("SPY", start="2021-01-01")["SPY"]
    result, table = compare_to_buy_and_hold(
        prices, lambda p: ma_crossover(p, fast=20, slow=50)
    )
    print("\n=== Moving Average Crossover (20/50) vs Buy & Hold ===")
    print(table.round(4).to_string())
