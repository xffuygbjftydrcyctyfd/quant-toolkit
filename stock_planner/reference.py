"""
Stock Planner — Python reference for the constrained optimizer.

Solves: minimize (lam/2) w'Sigma w - w'mu subject to
  - fully invested: sum(w) = 1
  - long-only with per-stock caps: 0 <= w_i <= stock_cap
    (equities capped, fixed income uncapped up to 100%)
  - per-sector caps on equity sectors: sum(w in sector) <= sector_cap

Caps loosen automatically when the user narrows the universe (fewer
sectors selected), otherwise the problem can become infeasible or force
everything into fixed income. The exact same rule ships in the JS.

Also computes marginal risk contributions (the explainability layer)
and exports validation vectors for the JS port.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATA = json.load(open(Path(__file__).parent / "web_app" / "stocks.json"))
TICKERS = DATA["tickers"]
SECTORS = DATA["sectors"]
MU_ALL = np.array(DATA["mu"])
COV_ALL = np.array(DATA["cov"])
FI_SECTOR = "Fixed Income"


def effective_caps(n_equity_sectors):
    """Caps loosen when the universe narrows. Mirrored exactly in JS."""
    sector_cap = max(0.35, 0.9 / max(n_equity_sectors, 1))
    stock_cap = max(0.15, sector_cap / 4)
    return stock_cap, sector_cap


def build_problem(excluded_sectors=()):
    """Subset the universe and construct caps + sector groups."""
    idx = [i for i, s in enumerate(SECTORS)
           if s == FI_SECTOR or s not in excluded_sectors]
    mu = MU_ALL[idx]
    cov = COV_ALL[np.ix_(idx, idx)]
    secs = [SECTORS[i] for i in idx]
    eq_sectors = sorted({s for s in secs if s != FI_SECTOR})
    stock_cap, sector_cap = effective_caps(len(eq_sectors))
    caps = np.array([1.0 if s == FI_SECTOR else stock_cap for s in secs])
    groups = {s: [k for k, ss in enumerate(secs) if ss == s]
              for s in eq_sectors}
    return idx, mu, cov, caps, groups, sector_cap, stock_cap


def solve(lam, mu, cov, caps, groups, sector_cap):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    for members in groups.values():
        cons.append({"type": "ineq",
                     "fun": lambda w, m=members: sector_cap - w[m].sum()})
    res = minimize(
        lambda w: 0.5 * lam * w @ cov @ w - w @ mu,
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0, c) for c in caps],
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 3000},
    )
    return res.x


def risk_contributions(w, cov):
    """MRC_i = w_i (Sigma w)_i / (w' Sigma w) — fraction of total variance
    attributable to each holding. Sums to 1. The explainability core."""
    Sw = cov @ w
    total = w @ Sw
    return w * Sw / total if total > 0 else np.zeros_like(w)


if __name__ == "__main__":
    vectors = {"cases": []}
    scenarios = [
        ("all sectors", ()),
        ("no defense/oil", ("Defense", "Oil & Gas")),
        ("only healthcare", tuple(s for s in sorted(set(SECTORS))
                                  if s not in (FI_SECTOR, "Healthcare"))),
    ]
    for label, excluded in scenarios:
        idx, mu, cov, caps, groups, sector_cap, stock_cap = \
            build_problem(excluded)
        print(f"\n=== {label}: {len(mu)} assets, sector_cap={sector_cap:.2f}, "
              f"stock_cap={stock_cap:.2f} ===")
        for lam in [100, 5, 1.5, 0.5]:
            w = solve(lam, mu, cov, caps, groups, sector_cap)
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov @ w))
            obj = float(0.5 * lam * w @ cov @ w - w @ mu)
            rc = risk_contributions(w, cov)
            top = np.argsort(w)[::-1][:5]
            print(f"lam={lam:>5}: ret={ret:.4f} vol={vol:.4f} | top: " +
                  ", ".join(f"{TICKERS[idx[i]]}={w[i]:.3f}" for i in top))
            # verify constraints hold
            assert abs(w.sum() - 1) < 1e-8
            assert (w <= caps + 1e-8).all() and (w >= -1e-8).all()
            for members in groups.values():
                assert w[members].sum() <= sector_cap + 1e-8
            vectors["cases"].append({
                "label": label, "excluded": list(excluded), "lam": lam,
                "idx": idx, "w": w.tolist(), "ret": ret, "vol": vol,
                "obj": obj, "sector_cap": sector_cap,
                "caps": caps.tolist(), "rc": rc.tolist(),
            })
    json.dump(vectors, open("/tmp/stock_test_vectors.json", "w"))
    print("\nAll constraint assertions passed. "
          "Wrote /tmp/stock_test_vectors.json")
