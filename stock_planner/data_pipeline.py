"""
Stock Planner data pipeline.

Builds the investable universe: ~44 real tickers across 7 equity sectors
plus 2 fixed-income ETFs. Pulls 5 years of daily prices via yfinance and
computes annualized mean returns and a Ledoit-Wolf shrunk covariance
matrix (a raw 44x44 sample covariance from ~5y of data is too noisy to
optimize on). Without internet, generates daily returns from a seeded
factor model (market + sector + idiosyncratic) and runs the identical
statistics path, so the whole pipeline is testable offline.

Output: stocks.json consumed by the web app and by reference.py.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

TRADING_DAYS = 252

UNIVERSE = {
    "Defense": [("LMT", "Lockheed Martin"), ("RTX", "RTX Corp"),
                ("NOC", "Northrop Grumman"), ("GD", "General Dynamics"),
                ("BA", "Boeing"), ("HII", "Huntington Ingalls")],
    "Oil & Gas": [("XOM", "Exxon Mobil"), ("CVX", "Chevron"),
                  ("COP", "ConocoPhillips"), ("SLB", "Schlumberger"),
                  ("EOG", "EOG Resources"), ("HAL", "Halliburton")],
    "Healthcare": [("JNJ", "Johnson & Johnson"), ("UNH", "UnitedHealth"),
                   ("PFE", "Pfizer"), ("ABT", "Abbott Labs"),
                   ("MRK", "Merck"), ("TMO", "Thermo Fisher")],
    "Utilities & Energy": [("NEE", "NextEra Energy"), ("DUK", "Duke Energy"),
                           ("AEP", "American Electric"), ("SO", "Southern Co"),
                           ("D", "Dominion Energy"), ("ED", "Consolidated Edison")],
    "Commercial & Retail": [("WMT", "Walmart"), ("AMZN", "Amazon"),
                            ("COST", "Costco"), ("TGT", "Target"),
                            ("HD", "Home Depot"), ("MCD", "McDonald's")],
    "Consumer Staples": [("PG", "Procter & Gamble"), ("KO", "Coca-Cola"),
                         ("PEP", "PepsiCo"), ("MDLZ", "Mondelez"),
                         ("HSY", "Hershey"), ("GIS", "General Mills")],
    "Agricultural": [("ADM", "Archer-Daniels"), ("BG", "Bunge"),
                     ("MOS", "Mosaic"), ("NTR", "Nutrien"),
                     ("DE", "John Deere"), ("CTVA", "Corteva")],
    "Fixed Income": [("BND", "Total Bond ETF"), ("BIL", "1-3mo T-Bill ETF")],
}

# Factor-model parameters for the synthetic fallback (annualized).
SECTOR_PARAMS = {
    #                (beta_mkt, sector_vol, idio_vol, base_mu)
    "Defense":            (0.75, 0.10, 0.18, 0.085),
    "Oil & Gas":          (1.00, 0.16, 0.22, 0.080),
    "Healthcare":         (0.70, 0.09, 0.17, 0.085),
    "Utilities & Energy": (0.55, 0.08, 0.13, 0.065),
    "Commercial & Retail": (0.95, 0.09, 0.19, 0.090),
    "Consumer Staples":   (0.60, 0.07, 0.14, 0.070),
    "Agricultural":       (0.90, 0.13, 0.21, 0.080),
}
MKT_VOL = 0.16


def flat_universe():
    tickers, names, sectors = [], [], []
    for sector, members in UNIVERSE.items():
        for t, n in members:
            tickers.append(t)
            names.append(n)
            sectors.append(sector)
    return tickers, names, sectors


def ledoit_wolf_cov(returns):
    """
    Ledoit-Wolf (2004) shrinkage toward a scaled-identity target.
    returns: (T, N) array of daily returns.
    Returns the shrunk covariance and the shrinkage intensity delta.
    """
    T, N = returns.shape
    X = returns - returns.mean(axis=0)
    S = X.T @ X / T

    m = np.trace(S) / N                     # mean eigenvalue
    F = m * np.eye(N)                       # shrinkage target
    d2 = np.sum((S - F) ** 2) / N           # distance sample -> target

    b2_sum = 0.0
    for t in range(T):
        xt = X[t][:, None]
        b2_sum += np.sum((xt @ xt.T - S) ** 2)
    b2 = min(b2_sum / (T ** 2 * N), d2)     # estimation error, capped

    delta = b2 / d2 if d2 > 0 else 0.0
    return delta * F + (1 - delta) * S, delta


def synthetic_daily_returns(n_days=1260, seed=42):
    """Seeded factor model: r = mu/252 + beta*mkt + sector + idio."""
    tickers, names, sectors = flat_universe()
    rng = np.random.default_rng(seed)
    n = len(tickers)
    sector_list = list(SECTOR_PARAMS)

    f_mkt = rng.normal(0, MKT_VOL / np.sqrt(TRADING_DAYS), n_days)
    f_sec = {s: rng.normal(0, SECTOR_PARAMS[s][1] / np.sqrt(TRADING_DAYS), n_days)
             for s in sector_list}

    rets = np.zeros((n_days, n))
    for i, (t, sec) in enumerate(zip(tickers, sectors)):
        if sec == "Fixed Income":
            mu, vol = (0.040, 0.055) if t == "BND" else (0.030, 0.008)
            rets[:, i] = mu / TRADING_DAYS + rng.normal(
                0, vol / np.sqrt(TRADING_DAYS), n_days)
        else:
            beta, _, idio, base_mu = SECTOR_PARAMS[sec]
            beta_i = beta * rng.uniform(0.8, 1.2)
            mu_i = base_mu + rng.uniform(-0.015, 0.02)
            idio_i = idio * rng.uniform(0.8, 1.2)
            rets[:, i] = (mu_i / TRADING_DAYS
                          + beta_i * f_mkt
                          + f_sec[sec]
                          + rng.normal(0, idio_i / np.sqrt(TRADING_DAYS), n_days))
    return rets


def real_daily_returns(tickers, years=5):
    import yfinance as yf
    data = yf.download(tickers, period=f"{years}y", progress=False)
    prices = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) \
        else data["Close"]
    prices = prices[tickers].dropna()
    if len(prices) < 500:
        raise ValueError("insufficient history")
    rets = np.log(prices / prices.shift(1)).dropna().values
    last_close = np.round(prices.iloc[-1].values, 2)
    return rets, last_close


def build(out_path=None, seed=42, universe="curated"):
    """universe: 'curated' (44 assets) or 'sp500' (503 + fixed income,
    requires internet for the Wikipedia constituents table)."""
    if universe == "sp500":
        try:
            tickers, names, sectors = sp500_universe()
            for (t, n) in UNIVERSE["Fixed Income"]:
                tickers.append(t); names.append(n); sectors.append("Fixed Income")
            print(f"[pipeline] S&P 500 universe: {len(tickers)} assets")
        except Exception as e:
            print(f"[pipeline] S&P 500 list unavailable ({e.__class__.__name__}); "
                  f"falling back to curated universe.")
            tickers, names, sectors = flat_universe()
    else:
        tickers, names, sectors = flat_universe()
    source = "live"
    prices_last = None
    caps = value = fundamentals = earnings_in = sector_mom = None
    stress = None
    try:
        rets, prices_last = real_daily_returns(tickers)
        print(f"[pipeline] Real data: {rets.shape[0]} days x {rets.shape[1]} assets")
        import yfinance as yf
        caps_l, pe_l, pb_l, fun = [], [], [], []
        for t in tickers:
            info = {}
            try:
                info = yf.Ticker(t).info or {}
            except Exception:
                pass
            caps_l.append(info.get("marketCap") or 1e9)
            pe = info.get("trailingPE"); pb = info.get("priceToBook")
            pe_l.append(pe if pe and pe > 0 else 20.0)
            pb_l.append(pb if pb and pb > 0 else 3.0)
            fun.append({"pe": round(pe_l[-1], 1), "pb": round(pb_l[-1], 2),
                        "revG": round(info.get("revenueGrowth") or 0.0, 3),
                        "de": round((info.get("debtToEquity") or 100) / 100, 2)})
            ed = None
            try:
                cal = yf.Ticker(t).calendar
                ed = (cal.get("Earnings Date") or [None])[0]
            except Exception:
                pass
            fun[-1]["earnIn"] = None
        caps = np.array(caps_l, dtype=float)
        value = 1 / np.array(pe_l) + 1 / np.array(pb_l)
        fundamentals = fun
        earnings_in = [f.pop("earnIn") for f in fun]
        # 3-month sector momentum from the return history itself
        sec_names = sorted(set(s for s in sectors if s != "Fixed Income"))
        sector_mom = {}
        for s in sec_names:
            idxs = [i for i, ss in enumerate(sectors) if ss == s]
            sector_mom[s] = round(float(rets[-63:, idxs].mean(axis=1).sum()), 4)
        stress = stress_returns_live(tickers)
    except Exception as e:
        print(f"[pipeline] yfinance unavailable ({e.__class__.__name__}); "
              f"using seeded factor-model returns.")
        tickers, names, sectors = flat_universe()
        rets = synthetic_daily_returns(seed=seed)
        rng = np.random.default_rng(seed + 1)
        prices_last = np.round(rng.uniform(25, 450, len(tickers)), 2)
        prices_last[tickers.index("BND")] = 72.50
        prices_last[tickers.index("BIL")] = 91.60
        source = "synthetic"
        caps, value, fundamentals, earnings_in, sector_mom = \
            synthetic_extras(tickers, sectors, seed)

    # v2 expected returns: Black-Litterman equilibrium + sector-shrunk
    # historical + momentum tilt (see black_litterman_momentum_mu docstring)
    mu, _, _ = black_litterman_momentum_mu(rets, sectors, caps)
    betas, _ = factor_betas(rets, caps, value)
    if stress is None:
        stress = stress_returns_offline(betas[:, 0])

    cov_daily, delta = ledoit_wolf_cov(rets)
    cov = cov_daily * TRADING_DAYS
    print(f"[pipeline] Ledoit-Wolf shrinkage intensity: {delta:.3f}")
    print(f"[pipeline] Condition number: raw={np.linalg.cond(np.cov(rets.T)):.0f} "
          f"-> shrunk={np.linalg.cond(cov_daily):.0f}")

    data = {
        "asOf": date.today().isoformat(),
        "source": source,
        "shrinkage": round(delta, 4),
        "tickers": tickers, "names": names, "sectors": sectors,
        "mu": np.round(mu, 6).tolist(),
        "cov": np.round(cov, 8).tolist(),
        "prices": np.round(prices_last, 2).tolist(),
        "returnsModel": "BL(50) + sector-shrunk hist(30) + momentum(20)",
        "caps": np.round(caps, 0).tolist(),
        "betas": np.round(betas, 4).tolist(),
        "factorNames": ["MKT", "SMB", "HML"],
        "fundamentals": fundamentals,
        "earningsIn": earnings_in,
        "sectorMom3m": sector_mom,
        "stress": stress,
    }
    out = Path(out_path or Path(__file__).parent / "web_app" / "stocks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w"))
    print(f"[pipeline] Wrote {out}")
    return data




# ====================== v2: expanded universe & richer stats ======================
STRESS_WINDOWS = [
    ("2008 Financial Crisis", "2007-10-09", "2009-03-09"),
    ("2020 COVID Crash", "2020-02-19", "2020-03-23"),
    ("2022 Rate-Hike Drawdown", "2022-01-03", "2022-10-12"),
]
# Modeled factor shocks for offline mode (market total return over window)
STRESS_SHOCKS = {"2008 Financial Crisis": -0.55,
                 "2020 COVID Crash": -0.34,
                 "2022 Rate-Hike Drawdown": -0.25}


def sp500_universe():
    """Pull the current S&P 500 constituent table from Wikipedia.
    Only works with internet access; callers fall back to the curated list."""
    import pandas as pd
    tables = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
    names = df["Security"].tolist()
    sectors = df["GICS Sector"].tolist()
    return tickers, names, sectors


def black_litterman_momentum_mu(rets, sectors, caps, blend=(0.5, 0.3, 0.2),
                                delta=2.5, mom_tilt=0.02):
    """
    Expected returns as a disclosed blend:
      50% Black-Litterman equilibrium (reverse-optimized from cap weights:
          pi = delta * Sigma * w_mkt — the returns that would make the
          market portfolio optimal),
      30% sector-shrunk historical means (the previous estimator),
      20% cross-sectional momentum tilt (12-1 month return z-score,
          capped at +/- mom_tilt annually).
    """
    T, N = rets.shape
    mu_hist = rets.mean(axis=0) * TRADING_DAYS
    mu_shrunk = np.empty(N)
    for i, sec in enumerate(sectors):
        sector_mean = np.mean([mu_hist[j] for j in range(N)
                               if sectors[j] == sec])
        mu_shrunk[i] = 0.7 * sector_mean + 0.3 * mu_hist[i]

    cov_ann = ledoit_wolf_cov(rets)[0] * TRADING_DAYS
    w_mkt = caps / caps.sum()
    pi = delta * cov_ann @ w_mkt  # BL equilibrium (no views)

    # 12-1 momentum: skip most recent 21 days, use prior 231
    if T >= 260:
        mom = rets[-252:-21].sum(axis=0)
    else:
        mom = rets[: max(T - 21, 1)].sum(axis=0)
    z = (mom - mom.mean()) / (mom.std() + 1e-12)
    tilt = np.clip(z, -2, 2) / 2 * mom_tilt

    a, b, c = blend
    return a * pi + b * mu_shrunk + c * (mu_shrunk + tilt), pi, w_mkt


def factor_betas(rets, caps, value_score):
    """
    Per-stock betas on three factors built from the universe itself:
      MKT = cap-weighted universe return; SMB = small minus big
      (bottom vs top cap tercile); HML = value minus growth (by value_score).
    Returns (betas Nx3, factor_series Tx3).
    """
    T, N = rets.shape
    w_mkt = caps / caps.sum()
    mkt = rets @ w_mkt
    small = caps <= np.quantile(caps, 1 / 3)
    big = caps >= np.quantile(caps, 2 / 3)
    smb = rets[:, small].mean(axis=1) - rets[:, big].mean(axis=1)
    hi_v = value_score >= np.quantile(value_score, 2 / 3)
    lo_v = value_score <= np.quantile(value_score, 1 / 3)
    hml = rets[:, hi_v].mean(axis=1) - rets[:, lo_v].mean(axis=1)
    F = np.column_stack([mkt, smb, hml])
    X = np.column_stack([np.ones(T), F])
    betas = np.zeros((N, 3))
    coef, *_ = np.linalg.lstsq(X, rets, rcond=None)
    betas = coef[1:].T  # drop intercept
    return betas, F


def stress_returns_offline(betas_mkt):
    """Offline: modeled scenario return per stock = beta_mkt * market shock."""
    return {name: {"coverage": 1.0,
                   "rets": (betas_mkt * shock).round(4).tolist()}
            for name, shock in STRESS_SHOCKS.items()}


def stress_returns_live(tickers):
    """Per-stock total return over each historical stress window."""
    import yfinance as yf
    out = {}
    for name, start, end in STRESS_WINDOWS:
        px = yf.download(tickers, start=start, end=end,
                         progress=False)["Close"]
        rets, covered = [], 0
        for t in tickers:
            s = px[t].dropna() if t in px else None
            if s is not None and len(s) > 10:
                rets.append(round(float(s.iloc[-1] / s.iloc[0] - 1), 4))
                covered += 1
            else:
                rets.append(None)
        out[name] = {"coverage": round(covered / len(tickers), 3),
                     "rets": rets}
    return out


def synthetic_extras(tickers, sectors, seed):
    """Caps, value scores, fundamentals, earnings days, and 3-mo sector
    momentum for offline mode — all seeded and clearly synthetic."""
    rng = np.random.default_rng(seed + 2)
    n = len(tickers)
    caps = rng.lognormal(mean=10.5, sigma=1.1, size=n)  # $B-ish scale
    pe = np.round(rng.uniform(8, 45, n), 1)
    pb = np.round(rng.uniform(0.8, 12, n), 2)
    value = 1 / pe + 1 / pb
    fundamentals = [{"pe": float(pe[i]), "pb": float(pb[i]),
                     "revG": round(float(rng.uniform(-0.05, 0.25)), 3),
                     "de": round(float(rng.uniform(0.1, 2.5)), 2)}
                    for i in range(n)]
    earnings_in = [int(x) for x in rng.integers(1, 95, n)]
    sec_names = sorted(set(s for s in sectors if s != "Fixed Income"))
    sector_mom = {s: round(float(rng.normal(0.02, 0.05)), 4)
                  for s in sec_names}
    return caps, value, fundamentals, earnings_in, sector_mom


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["curated", "sp500"],
                    default="curated")
    args = ap.parse_args()
    d = build(universe=args.universe)
    mu = np.array(d["mu"])
    print(f"\nUniverse: {len(d['tickers'])} assets, "
          f"mu range [{mu.min():.3f}, {mu.max():.3f}]")
