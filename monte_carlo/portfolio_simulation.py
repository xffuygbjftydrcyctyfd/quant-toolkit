"""
Monte Carlo portfolio risk simulation.

Takes the max-Sharpe portfolio from the Markowitz optimizer and simulates
thousands of one-year paths of its value, using the same estimated returns
and covariance. Reports Value at Risk (VaR) and Expected Shortfall (CVaR):
not just "what do we expect," but "how bad does the bottom tail get."

This module deliberately consumes the optimizer's output — the toolkit's
pieces validate and extend each other rather than standing alone.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from portfolio_optimizer.markowitz import (  # noqa: E402
    annualized_stats, max_sharpe_weights, portfolio_performance,
)
from shared.data_utils import fetch_prices, daily_returns  # noqa: E402

TRADING_DAYS = 252


def simulate_paths(weights, daily_mu, daily_cov, n_days=TRADING_DAYS,
                   n_sims=5000, initial=10_000, seed=1):
    """
    Simulate portfolio value paths with multivariate-normal daily returns
    (preserving the estimated correlations between assets).
    """
    rng = np.random.default_rng(seed)
    asset_rets = rng.multivariate_normal(daily_mu, daily_cov,
                                         size=(n_sims, n_days))
    port_rets = asset_rets @ weights
    paths = initial * np.cumprod(1 + port_rets, axis=1)
    return paths


def risk_report(paths, initial=10_000, alpha=0.05):
    terminal = paths[:, -1]
    rets = terminal / initial - 1
    var = np.quantile(rets, alpha)
    cvar = rets[rets <= var].mean()
    return {
        "mean_return": rets.mean(),
        "median_return": np.median(rets),
        "prob_loss": (rets < 0).mean(),
        f"VaR_{int((1 - alpha) * 100)}": var,
        f"CVaR_{int((1 - alpha) * 100)}": cvar,
        "best": rets.max(),
        "worst": rets.min(),
    }


def run(tickers, risk_free=0.04, initial=10_000, out_path=None):
    prices = fetch_prices(tickers)
    mu_ann, cov_ann = annualized_stats(prices)
    weights = max_sharpe_weights(mu_ann, cov_ann, risk_free)

    rets = daily_returns(prices)
    paths = simulate_paths(weights, rets.mean().values, rets.cov().values,
                           initial=initial)
    report = risk_report(paths, initial)

    exp_ret, exp_vol = portfolio_performance(weights, mu_ann, cov_ann)
    print("Max-Sharpe weights:",
          dict(zip(prices.columns, np.round(weights, 3))))
    print(f"Optimizer expectation: return={exp_ret:.2%}, vol={exp_vol:.2%}\n")
    print("=== One-Year Monte Carlo Risk Report ===")
    for k, v in report.items():
        print(f"{k:>15}: {v:.2%}" if abs(v) < 10 else f"{k:>15}: {v:,.0f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    idx = np.random.default_rng(2).choice(len(paths), 150, replace=False)
    axes[0].plot(paths[idx].T, lw=0.5, alpha=0.35, color="steelblue")
    axes[0].plot(paths.mean(axis=0), lw=2.5, color="darkred", label="Mean path")
    axes[0].axhline(initial, color="gray", ls=":")
    axes[0].set_xlabel("Trading day")
    axes[0].set_ylabel("Portfolio value ($)")
    axes[0].set_title("Simulated One-Year Portfolio Paths (150 of 5,000)")
    axes[0].legend()

    terminal_rets = paths[:, -1] / initial - 1
    var_key = [k for k in report if k.startswith("VaR")][0]
    axes[1].hist(terminal_rets, bins=100, density=True,
                 alpha=0.75, color="steelblue")
    axes[1].axvline(report[var_key], color="red", ls="--",
                    label=f"{var_key}: {report[var_key]:.1%}")
    axes[1].axvline(0, color="gray", ls=":")
    axes[1].set_xlabel("One-year return")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of Outcomes — the point of Monte Carlo:\n"
                      "a full distribution, not a single forecast")
    axes[1].legend()

    plt.tight_layout()
    out = Path(out_path or Path(__file__).parent / "mc_portfolio_risk.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    run(["AAPL", "MSFT", "JPM", "XOM", "JNJ", "GLD"])
