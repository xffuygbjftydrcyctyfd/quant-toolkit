"""
Shared data utilities for the quant toolkit.

Tries to pull real historical data with yfinance. If yfinance is not
installed or there is no internet connection, it falls back to generating
realistic synthetic price paths (geometric Brownian motion) so every tool
in this repo can still be run and tested offline.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def fetch_prices(tickers, start="2021-01-01", end=None, seed=42):
    """
    Return a DataFrame of adjusted close prices, one column per ticker.

    Tries yfinance first; falls back to synthetic GBM paths if the download
    fails for any reason (no internet, no yfinance, bad ticker).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    try:
        import yfinance as yf
        data = yf.download(tickers, start=start, end=end, progress=False)
        # yfinance returns different shapes for 1 vs many tickers
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        else:
            prices = data["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices = prices.dropna()
        if len(prices) < 50:
            raise ValueError("Downloaded data too short, falling back.")
        print(f"[data_utils] Loaded real data for {list(prices.columns)}")
        return prices
    except Exception as e:
        print(f"[data_utils] yfinance unavailable ({e.__class__.__name__}); "
              f"using synthetic data.")
        return synthetic_prices(tickers, seed=seed)


def synthetic_prices(tickers, n_days=756, seed=42):
    """
    Generate correlated geometric Brownian motion price paths.

    Each ticker gets its own drift and volatility, and returns are drawn
    from a multivariate normal so the assets are realistically correlated
    (important for the portfolio optimizer and pairs-style analysis).
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)

    mu = rng.uniform(0.04, 0.14, size=n) / TRADING_DAYS          # daily drift
    sigma = rng.uniform(0.15, 0.35, size=n) / np.sqrt(TRADING_DAYS)

    # Build a random but valid correlation matrix
    A = rng.uniform(-1, 1, size=(n, n))
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    # Blend toward identity so correlations are moderate, not extreme
    corr = 0.5 * corr + 0.5 * np.eye(n)

    cov_daily = corr * np.outer(sigma, sigma)
    rets = rng.multivariate_normal(mu - 0.5 * sigma**2, cov_daily, size=n_days)

    start_prices = rng.uniform(40, 400, size=n)
    prices = start_prices * np.exp(np.cumsum(rets, axis=0))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


def daily_returns(prices):
    """Simple daily returns from a price DataFrame or Series."""
    return prices.pct_change().dropna()


def log_returns(prices):
    """Log daily returns from a price DataFrame or Series."""
    return np.log(prices / prices.shift(1)).dropna()
