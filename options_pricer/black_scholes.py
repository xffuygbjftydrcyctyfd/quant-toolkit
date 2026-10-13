"""
Black-Scholes option pricing, implemented from scratch.

The only external math used is the standard normal CDF/PDF from SciPy —
the pricing formula itself is written out in full so every term is visible.

Model assumptions worth remembering (and questioning):
- Constant volatility and interest rate over the option's life
- Lognormally distributed prices (no jumps, no fat tails)
- No dividends (this version), continuous frictionless trading
Real markets violate all of these, which is exactly why comparing
model prices to market prices (market_comparison.py) is interesting.
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    """
    The two quantities at the heart of Black-Scholes.

    d1 relates to the expected value of the stock conditional on the option
    finishing in the money; d2 is the risk-neutral probability the option
    expires in the money (for a call).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    """European call: C = S*N(d1) - K*e^(-rT)*N(d2)."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """European put via the same terms: P = K*e^(-rT)*N(-d2) - S*N(-d1)."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_call_parity_check(S, K, T, r, sigma, tol=1e-8):
    """
    Sanity check: C - P should equal S - K*e^(-rT).
    If this fails, the implementation is wrong. Cheap self-test.
    """
    lhs = call_price(S, K, T, r, sigma) - put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)
    return abs(lhs - rhs) < tol


def implied_volatility(market_price, S, K, T, r, kind="call",
                       lo=1e-4, hi=5.0, tol=1e-6, max_iter=100):
    """
    Back out the volatility the market is implying from an option's price,
    using bisection. This inverts the pricing formula: instead of
    vol -> price, we solve price -> vol.
    """
    pricer = call_price if kind == "call" else put_price

    f_lo = pricer(S, K, T, r, lo) - market_price
    f_hi = pricer(S, K, T, r, hi) - market_price
    if f_lo * f_hi > 0:
        return np.nan  # price outside achievable range

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = pricer(S, K, T, r, mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)
    print(f"Call price: {c:.4f}")
    print(f"Put price:  {p:.4f}")
    print(f"Put-call parity holds: {put_call_parity_check(S, K, T, r, sigma)}")

    iv = implied_volatility(c, S, K, T, r, kind="call")
    print(f"Recovered implied vol from call price: {iv:.4f} (input was {sigma})")
