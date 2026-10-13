"""
Efficient frontier visualization.

Traces the curve of minimum-variance portfolios across a range of target
returns, scatters random portfolios underneath it for contrast, and marks
the max-Sharpe and min-variance portfolios. The gap between the random
cloud and the frontier is the visual argument for optimization: most
portfolios you could pick are strictly worse than one on the curve.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from portfolio_optimizer.markowitz import (  # noqa: E402
    annualized_stats, min_variance_weights, max_sharpe_weights,
    portfolio_performance,
)
from shared.data_utils import fetch_prices  # noqa: E402


def plot_frontier(tickers, risk_free=0.04, n_random=3000,
                  start="2021-01-01", out_path=None):
    prices = fetch_prices(tickers, start=start)
    mu, cov = annualized_stats(prices)
    rng = np.random.default_rng(0)

    # Random portfolios (Dirichlet gives valid long-only weights)
    rand_w = rng.dirichlet(np.ones(len(mu)), size=n_random)
    rand_ret = rand_w @ mu.values
    rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov.values, rand_w))
    rand_sharpe = (rand_ret - risk_free) / rand_vol

    # Frontier: sweep target returns, minimize variance at each
    targets = np.linspace(mu.min(), mu.max(), 60)
    frontier_vol = []
    for t in targets:
        w = min_variance_weights(mu, cov, target_return=t)
        _, vol = portfolio_performance(w, mu, cov)
        frontier_vol.append(vol)

    # Key portfolios
    w_ms = max_sharpe_weights(mu, cov, risk_free)
    ms_ret, ms_vol = portfolio_performance(w_ms, mu, cov)
    w_mv = min_variance_weights(mu, cov)
    mv_ret, mv_vol = portfolio_performance(w_mv, mu, cov)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(rand_vol, rand_ret, c=rand_sharpe, cmap="viridis",
                    s=8, alpha=0.35, label="Random portfolios")
    ax.plot(frontier_vol, targets, "r-", lw=2.5, label="Efficient frontier")
    ax.scatter([ms_vol], [ms_ret], marker="*", s=350, c="gold",
               edgecolors="black", zorder=5, label="Max Sharpe")
    ax.scatter([mv_vol], [mv_ret], marker="D", s=120, c="deepskyblue",
               edgecolors="black", zorder=5, label="Min variance")

    fig.colorbar(sc, label="Sharpe ratio")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title(f"Efficient Frontier — {', '.join(prices.columns)}")
    ax.legend()
    plt.tight_layout()

    out = Path(out_path or Path(__file__).parent / "efficient_frontier.png")
    plt.savefig(out, dpi=120)
    print(f"Saved plot to {out}")
    print(f"\nMax Sharpe portfolio: return={ms_ret:.2%}, vol={ms_vol:.2%}, "
          f"Sharpe={(ms_ret - risk_free) / ms_vol:.2f}")
    print("Weights:", dict(zip(prices.columns, np.round(w_ms, 3))))


if __name__ == "__main__":
    plot_frontier(["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GLD"])
