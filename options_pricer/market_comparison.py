"""
Compare Black-Scholes theoretical prices against real market option prices.

With internet access this pulls a live option chain via yfinance and plots
model price vs. market price across strikes, plus the implied volatility
smile — the market's systematic disagreement with the constant-volatility
assumption. Without internet, it demonstrates the same analysis on a
synthetic chain generated with a known volatility skew.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from options_pricer.black_scholes import call_price, implied_volatility  # noqa: E402

RISK_FREE = 0.04


def real_chain(ticker="SPY"):
    """Pull the nearest-expiry call chain from Yahoo Finance."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    expiry = tk.options[2]  # skip the very nearest expiries (noisy)
    chain = tk.option_chain(expiry).calls
    S = tk.history(period="1d")["Close"].iloc[-1]
    T = max((pd.Timestamp(expiry) - pd.Timestamp.today()).days, 1) / 365
    df = chain[["strike", "lastPrice"]].rename(
        columns={"strike": "K", "lastPrice": "market_price"})
    df = df[(df.K > 0.8 * S) & (df.K < 1.2 * S)].reset_index(drop=True)
    return df, S, T


def synthetic_chain(S=100.0, T=0.25, seed=7):
    """
    Build a synthetic option chain with a realistic volatility skew:
    out-of-the-money puts (low strikes) trade at higher implied vol
    because markets price in crash risk. Small noise mimics bid-ask spread.
    """
    rng = np.random.default_rng(seed)
    strikes = np.arange(80, 121, 2.5)
    skew_vol = 0.22 + 0.15 * np.maximum(0, (S - strikes) / S) \
        + 0.05 * np.maximum(0, (strikes - S) / S)
    prices = np.array([
        call_price(S, K, T, RISK_FREE, v) * (1 + rng.normal(0, 0.01))
        for K, v in zip(strikes, skew_vol)
    ])
    return pd.DataFrame({"K": strikes, "market_price": prices}), S, T


def analyze(df, S, T, source_label):
    flat_vol = 0.22  # what a naive constant-vol model would assume

    df["bs_price_flat_vol"] = [
        call_price(S, K, T, RISK_FREE, flat_vol) for K in df.K
    ]
    df["implied_vol"] = [
        implied_volatility(p, S, K, T, RISK_FREE, "call")
        for p, K in zip(df.market_price, df.K)
    ]
    df = df.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df.K, df.market_price, "o-", label="Market price")
    axes[0].plot(df.K, df.bs_price_flat_vol, "s--",
                 label=f"Black-Scholes (flat {flat_vol:.0%} vol)")
    axes[0].axvline(S, color="gray", ls=":", label="Spot")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Call price")
    axes[0].set_title(f"Model vs Market ({source_label})")
    axes[0].legend()

    axes[1].plot(df.K, df.implied_vol, "o-", color="darkred")
    axes[1].axvline(S, color="gray", ls=":")
    axes[1].axhline(flat_vol, color="gray", ls="--", alpha=0.5)
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("Implied volatility")
    axes[1].set_title("Implied Volatility Smile / Skew")

    plt.tight_layout()
    out = Path(__file__).parent / "model_vs_market.png"
    plt.savefig(out, dpi=120)
    print(f"Saved plot to {out}")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    try:
        df, S, T = real_chain("SPY")
        label = "SPY, real data"
    except Exception as e:
        print(f"[market_comparison] Live data unavailable "
              f"({e.__class__.__name__}); using synthetic chain with skew.")
        df, S, T = synthetic_chain()
        label = "synthetic chain"
    analyze(df, S, T, label)
