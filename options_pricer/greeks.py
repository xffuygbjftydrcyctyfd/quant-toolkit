"""
The Greeks: sensitivity analysis for option prices.

Each Greek answers "if this one input moves, how much does the option's
price move?" — a decomposition of a single output into the influence of
each input. Analytic formulas are used (derived from Black-Scholes),
and a finite-difference check is included to verify them numerically.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from options_pricer.black_scholes import d1_d2, call_price, put_price  # noqa: E402


def delta(S, K, T, r, sigma, kind="call"):
    """Price sensitivity to a $1 move in the underlying."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """How fast delta itself changes — convexity of the option."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Sensitivity to a 1-point (100%) change in volatility.
    Divide by 100 for the per-vol-point convention used on desks."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, kind="call"):
    """Time decay: price change per year of elapsed time (usually negative).
    Divide by 365 for per-calendar-day theta."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if kind == "call":
        return common - r * K * np.exp(-r * T) * norm.cdf(d2)
    return common + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho(S, K, T, r, sigma, kind="call"):
    """Sensitivity to a 1-point change in the risk-free rate."""
    _, d2 = d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def all_greeks(S, K, T, r, sigma, kind="call"):
    return {
        "delta": delta(S, K, T, r, sigma, kind),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, kind),
        "rho": rho(S, K, T, r, sigma, kind),
    }


def finite_difference_check(S, K, T, r, sigma, kind="call", h=1e-4):
    """
    Verify the analytic Greeks numerically by bumping each input.
    If analytic and numerical disagree, something is wrong — this is the
    Greeks equivalent of the put-call parity self-test.
    """
    pricer = call_price if kind == "call" else put_price
    num = {
        "delta": (pricer(S + h, K, T, r, sigma) - pricer(S - h, K, T, r, sigma)) / (2 * h),
        "gamma": (pricer(S + h, K, T, r, sigma) - 2 * pricer(S, K, T, r, sigma)
                  + pricer(S - h, K, T, r, sigma)) / h**2,
        "vega": (pricer(S, K, T, r, sigma + h) - pricer(S, K, T, r, sigma - h)) / (2 * h),
        "theta": -(pricer(S, K, T + h, r, sigma) - pricer(S, K, T - h, r, sigma)) / (2 * h),
        "rho": (pricer(S, K, T, r + h, sigma) - pricer(S, K, T, r - h, sigma)) / (2 * h),
    }
    return num


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    analytic = all_greeks(S, K, T, r, sigma, "call")
    numeric = finite_difference_check(S, K, T, r, sigma, "call")

    print(f"{'Greek':<8}{'Analytic':>12}{'Numerical':>12}")
    for g in analytic:
        print(f"{g:<8}{analytic[g]:>12.6f}{numeric[g]:>12.6f}")
