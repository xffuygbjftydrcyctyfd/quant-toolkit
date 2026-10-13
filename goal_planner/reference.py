"""
Goal Planner — Python reference implementation.

This is the validation baseline for the browser app. Every formula the
JavaScript version uses is implemented here first with scipy/numpy, and
the JS is required to match it before shipping. Run this file to print
the reference numbers and export test vectors to /tmp/goal_test_vectors.json.

Model summary:
- Asset classes with stated long-run capital market assumptions
- Long-only mean-variance optimization: maximize w'mu - (lambda/2) w'Sigma w
  over the simplex, swept across risk-aversion lambda to trace the frontier
- Risk questionnaire score maps to lambda (log scale)
- Goal engine: monthly lognormal Monte Carlo with contributions,
  reporting probability of hitting an inflation-grown target
"""

import json

import numpy as np
from scipy.optimize import minimize

# ---- Capital market assumptions (annual, nominal). Deliberately visible. ----
ASSETS = ["US Stocks", "US Small Cap", "Intl Stocks", "Bonds", "Cash", "Gold"]
MU = np.array([0.085, 0.095, 0.075, 0.040, 0.030, 0.045])
VOL = np.array([0.16, 0.20, 0.17, 0.055, 0.01, 0.15])
CORR = np.array([
    [1.00, 0.90, 0.80, 0.10, 0.05, 0.05],
    [0.90, 1.00, 0.75, 0.05, 0.05, 0.05],
    [0.80, 0.75, 1.00, 0.10, 0.05, 0.10],
    [0.10, 0.05, 0.10, 1.00, 0.05, 0.15],
    [0.05, 0.05, 0.05, 0.05, 1.00, 0.00],
    [0.05, 0.05, 0.10, 0.15, 0.00, 1.00],
])
COV = CORR * np.outer(VOL, VOL)
INFLATION = 0.025


def long_only_weights(lam, mu=MU, cov=COV):
    """Maximize w'mu - (lam/2) w'cov w subject to w >= 0, sum(w) = 1."""
    n = len(mu)
    res = minimize(
        lambda w: 0.5 * lam * w @ cov @ w - w @ mu,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return res.x


def portfolio_stats(w, mu=MU, cov=COV):
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def score_to_lambda(score):
    """Risk score 0-100 -> risk aversion lambda, log scale 100 -> 0.5."""
    return 10 ** (2 - 2.3 * score / 100)


def adjusted_score(raw_score, horizon_years):
    """Horizon caps/boosts the questionnaire score. Stated in methodology."""
    s = raw_score
    if horizon_years < 3:
        s = min(s, 25)
    elif horizon_years < 7:
        s = min(s, 60)
    elif horizon_years > 15:
        s = min(100, s + 8)
    return s


def goal_probability(w, target, current, monthly, years,
                     n_sims=20_000, seed=0, mu=MU, cov=COV,
                     inflation=INFLATION):
    """
    Monte Carlo of monthly portfolio value with contributions.
    Portfolio is treated as a single lognormal process at its blended
    mu/vol (equivalent to assuming continuous rebalancing to constant
    weights). Target is grown by inflation to keep purchasing power.
    """
    p_mu, p_vol = portfolio_stats(w, mu, cov)
    months = int(round(years * 12))
    m_drift = (p_mu - 0.5 * p_vol**2) / 12
    m_sd = p_vol / np.sqrt(12)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, months))
    growth = np.exp(m_drift + m_sd * Z)

    values = np.full(n_sims, float(current))
    paths_pct = []  # store percentiles per month for the fan chart
    for t in range(months):
        values = values * growth[:, t] + monthly
        if t % 3 == 0 or t == months - 1:
            paths_pct.append([
                t + 1,
                float(np.percentile(values, 10)),
                float(np.percentile(values, 50)),
                float(np.percentile(values, 90)),
            ])

    target_infl = target * (1 + inflation) ** years
    prob = float((values >= target_infl).mean())
    return {
        "prob": prob,
        "target_inflated": target_infl,
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "portfolio_mu": p_mu,
        "portfolio_vol": p_vol,
        "fan": paths_pct,
    }


def resample_allocations(lam, n_draws=200, mu_uncertainty=0.015, seed=1):
    """
    Estimation-uncertainty resampling: expected returns are not knowable
    to the decimal, so perturb each mu by N(0, 1.5%) and re-optimize.
    Returns per-asset 10th/90th percentile weights across draws.
    """
    rng = np.random.default_rng(seed)
    all_w = []
    for _ in range(n_draws):
        mu_pert = MU + rng.normal(0, mu_uncertainty, size=len(MU))
        all_w.append(long_only_weights(lam, mu=mu_pert))
    all_w = np.array(all_w)
    return {
        "lo": np.percentile(all_w, 10, axis=0).tolist(),
        "hi": np.percentile(all_w, 90, axis=0).tolist(),
    }


if __name__ == "__main__":
    vectors = {"assets": ASSETS, "mu": MU.tolist(), "cov": COV.tolist(),
               "frontier": [], "goal_cases": []}

    print("=== Long-only frontier (reference) ===")
    for lam in [100, 20, 5, 1.5, 0.5]:
        w = long_only_weights(lam)
        ret, vol = portfolio_stats(w)
        vectors["frontier"].append(
            {"lam": lam, "w": w.tolist(), "ret": ret, "vol": vol})
        print(f"lambda={lam:>6}: ret={ret:.4f} vol={vol:.4f} "
              f"w={np.round(w, 3)}")

    print("\n=== Goal probability reference cases ===")
    cases = [
        # (score, target, current, monthly, years)
        (70, 500_000, 20_000, 800, 25),
        (35, 60_000, 10_000, 500, 8),
        (90, 1_000_000, 50_000, 1_500, 30),
    ]
    for score, target, current, monthly, years in cases:
        lam = score_to_lambda(adjusted_score(score, years))
        w = long_only_weights(lam)
        r = goal_probability(w, target, current, monthly, years)
        vectors["goal_cases"].append({
            "score": score, "target": target, "current": current,
            "monthly": monthly, "years": years, "lam": lam,
            "w": w.tolist(), "prob": r["prob"],
            "p10": r["p10"], "p50": r["p50"], "p90": r["p90"],
            "portfolio_mu": r["portfolio_mu"],
            "portfolio_vol": r["portfolio_vol"],
        })
        print(f"score={score} target=${target:,} {years}y: "
              f"P(goal)={r['prob']:.3f}  "
              f"median=${r['p50']:,.0f}  "
              f"[p10=${r['p10']:,.0f}, p90=${r['p90']:,.0f}]")

    with open("/tmp/goal_test_vectors.json", "w") as f:
        json.dump(vectors, f)
    print("\nWrote /tmp/goal_test_vectors.json")
