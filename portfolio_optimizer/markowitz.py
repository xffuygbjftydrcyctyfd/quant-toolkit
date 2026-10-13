"""
Markowitz mean-variance portfolio optimization.

Given a set of assets, find the weights that minimize portfolio variance
for a target level of expected return (or maximize the Sharpe ratio).
Solved with SciPy's SLSQP constrained optimizer; long-only by default.

The key insight of the model: an asset's contribution to portfolio risk
depends on its correlations with everything else, not just its own
volatility — which is why diversification is "the only free lunch."
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parents[1]))
from shared.data_utils import fetch_prices, daily_returns  # noqa: E402

TRADING_DAYS = 252


def annualized_stats(prices):
    """Annualized expected returns vector and covariance matrix."""
    rets = daily_returns(prices)
    mu = rets.mean() * TRADING_DAYS
    cov = rets.cov() * TRADING_DAYS
    return mu, cov


def portfolio_performance(weights, mu, cov):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol


def min_variance_weights(mu, cov, target_return=None, long_only=True):
    """
    Minimize variance, optionally subject to a target expected return.
    Weights sum to 1; long-only unless specified otherwise.
    """
    n = len(mu)
    x0 = np.full(n, 1 / n)
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mu - target_return}
        )

    result = minimize(
        lambda w: w @ cov @ w, x0,
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def max_sharpe_weights(mu, cov, risk_free=0.0, long_only=True):
    """Maximize (return - rf) / volatility."""
    n = len(mu)
    x0 = np.full(n, 1 / n)
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w, mu, cov)
        return -(ret - risk_free) / vol

    result = minimize(neg_sharpe, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


def optimize(tickers, risk_free=0.04, start="2021-01-01"):
    """End-to-end: fetch data, compute stats, return key portfolios."""
    prices = fetch_prices(tickers, start=start)
    mu, cov = annualized_stats(prices)

    w_minvar = min_variance_weights(mu, cov)
    w_sharpe = max_sharpe_weights(mu, cov, risk_free)

    report = pd.DataFrame({
        "Min Variance": w_minvar,
        "Max Sharpe": w_sharpe,
    }, index=prices.columns)

    for name, w in [("Min Variance", w_minvar), ("Max Sharpe", w_sharpe)]:
        ret, vol = portfolio_performance(w, mu, cov)
        sharpe = (ret - risk_free) / vol
        report.loc["— Exp. Return", name] = ret
        report.loc["— Volatility", name] = vol
        report.loc["— Sharpe", name] = sharpe

    return report, mu, cov, prices


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GLD"]
    report, mu, cov, _ = optimize(tickers)
    print("\n=== Optimal Portfolios (long-only) ===")
    print(report.round(4).to_string())
