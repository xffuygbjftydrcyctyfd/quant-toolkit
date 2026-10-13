"""
Monte Carlo option pricing.

Simulates thousands of possible terminal stock prices under the same
lognormal assumptions as Black-Scholes, computes the option payoff in
each scenario, and discounts the average back to today.

The point of this module is validation: the Monte Carlo estimate should
converge to the analytic Black-Scholes price as simulations increase.
If two completely different methods agree, both implementations are
probably right — the same cross-checking instinct as put-call parity
and the finite-difference Greeks test.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from options_pricer.black_scholes import call_price, put_price  # noqa: E402


def simulate_terminal_prices(S, T, r, sigma, n_sims, seed=0):
    """
    Draw terminal prices under the risk-neutral measure:
    S_T = S * exp((r - sigma^2/2)T + sigma*sqrt(T)*Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def mc_option_price(S, K, T, r, sigma, kind="call", n_sims=100_000, seed=0):
    """Monte Carlo price with a standard-error estimate."""
    ST = simulate_terminal_prices(S, T, r, sigma, n_sims, seed)
    payoff = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_err = discounted.std(ddof=1) / np.sqrt(n_sims)
    return price, std_err, ST, payoff


def convergence_study(S, K, T, r, sigma, kind="call", out_path=None):
    """Show the MC estimate converging to the analytic price."""
    analytic = (call_price if kind == "call" else put_price)(S, K, T, r, sigma)
    sim_counts = np.logspace(2, 6, 20).astype(int)

    estimates, errors = [], []
    for n in sim_counts:
        p, se, _, _ = mc_option_price(S, K, T, r, sigma, kind, n_sims=n)
        estimates.append(p)
        errors.append(se)
    estimates, errors = np.array(estimates), np.array(errors)

    _, _, ST, payoff = mc_option_price(S, K, T, r, sigma, kind, n_sims=200_000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].axhline(analytic, color="red", ls="--",
                    label=f"Black-Scholes: {analytic:.4f}")
    axes[0].errorbar(sim_counts, estimates, yerr=1.96 * errors, fmt="o-",
                     capsize=3, label="MC estimate (95% CI)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of simulations")
    axes[0].set_ylabel(f"{kind.capitalize()} price")
    axes[0].set_title("Monte Carlo Convergence to Analytic Price")
    axes[0].legend()

    axes[1].hist(ST, bins=120, density=True, alpha=0.7, color="steelblue")
    axes[1].axvline(K, color="red", ls="--", label=f"Strike = {K}")
    axes[1].axvline(S, color="gray", ls=":", label=f"Spot = {S}")
    itm = (payoff > 0).mean()
    axes[1].set_xlabel("Terminal price $S_T$")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Simulated Terminal Price Distribution "
                      f"({itm:.1%} finish in the money)")
    axes[1].legend()

    plt.tight_layout()
    out = Path(out_path or Path(__file__).parent / "mc_option_convergence.png")
    plt.savefig(out, dpi=120)
    print(f"Saved plot to {out}")

    final_p, final_se, _, _ = mc_option_price(S, K, T, r, sigma, kind,
                                              n_sims=1_000_000)
    print(f"\nAnalytic {kind} price:     {analytic:.4f}")
    print(f"MC price (1M sims):        {final_p:.4f} ± {1.96 * final_se:.4f}")
    print(f"Difference:                {abs(final_p - analytic):.4f}")


if __name__ == "__main__":
    convergence_study(S=100, K=105, T=0.5, r=0.04, sigma=0.25, kind="call")
