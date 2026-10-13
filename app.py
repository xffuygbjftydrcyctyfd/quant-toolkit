"""
Quant Finance Toolkit — interactive dashboard.

Wraps the four command-line tools (backtester, options pricer, portfolio
optimizer, Monte Carlo simulator) in a single shareable web app. Every
computation reuses the exact same functions as the standalone scripts —
this is a different front door onto the same code, not a rewrite.

Run locally with:  streamlit run app.py
Deploy for free at: https://share.streamlit.io
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from shared.data_utils import fetch_prices, daily_returns
from backtesting_engine.engine import compare_to_buy_and_hold
from backtesting_engine.strategies.moving_average import (
    ma_crossover, momentum, mean_reversion_zscore,
)
from options_pricer.black_scholes import call_price, put_price
from options_pricer.greeks import all_greeks
from portfolio_optimizer.markowitz import (
    annualized_stats, min_variance_weights, max_sharpe_weights,
    portfolio_performance,
)
from monte_carlo.option_simulation import mc_option_price
from monte_carlo.portfolio_simulation import simulate_paths, risk_report

st.set_page_config(page_title="Quant Finance Toolkit", layout="wide")

st.title("Quant Finance Toolkit")
st.caption(
    "Four tools, one shared foundation: a backtester, an options pricer "
    "built from Black-Scholes, a Markowitz portfolio optimizer, and a "
    "Monte Carlo simulator that cross-checks the other two."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Backtester", "Options Pricer", "Portfolio Optimizer", "Monte Carlo"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Backtester
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Strategy Backtester")
    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input("Ticker", value="SPY", key="bt_ticker")
        strategy_name = st.selectbox(
            "Strategy",
            ["Moving Average Crossover", "Momentum", "Mean Reversion (z-score)"],
        )
        if strategy_name == "Moving Average Crossover":
            fast = st.slider("Fast window", 5, 50, 20)
            slow = st.slider("Slow window", 20, 200, 50)
        elif strategy_name == "Momentum":
            lookback = st.slider("Lookback window (days)", 10, 200, 60)
        else:
            window = st.slider("Rolling window", 5, 60, 20)
            entry_z = st.slider("Entry z-score", 0.5, 3.0, 1.0)
        cost_bps = st.slider("Transaction cost (bps)", 0.0, 20.0, 1.0)
        run_bt = st.button("Run backtest", type="primary")

    with col2:
        if run_bt:
            with st.spinner("Fetching data and running backtest..."):
                prices = fetch_prices([ticker]).iloc[:, 0]
                if strategy_name == "Moving Average Crossover":
                    strat_fn = lambda p: ma_crossover(p, fast=fast, slow=slow)
                elif strategy_name == "Momentum":
                    strat_fn = lambda p: momentum(p, lookback=lookback)
                else:
                    strat_fn = lambda p: mean_reversion_zscore(
                        p, window=window, entry_z=entry_z)
                result, table = compare_to_buy_and_hold(
                    prices, strat_fn, cost_bps=cost_bps)

            st.line_chart(result["equity"], height=280)
            st.caption("Growth of $1 — strategy equity curve")
            st.dataframe(table.round(4), use_container_width=True)
        else:
            st.info("Set your parameters and run the backtest.")

# ---------------------------------------------------------------------------
# TAB 2 — Options Pricer
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Black-Scholes Options Pricer")
    col1, col2 = st.columns([1, 2])

    with col1:
        S = st.number_input("Spot price (S)", value=100.0, min_value=0.01)
        K = st.number_input("Strike price (K)", value=105.0, min_value=0.01)
        T = st.slider("Time to expiry (years)", 0.01, 2.0, 0.5)
        r = st.slider("Risk-free rate", 0.0, 0.10, 0.04)
        sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.25)

    with col2:
        c = call_price(S, K, T, r, sigma)
        p = put_price(S, K, T, r, sigma)
        m1, m2 = st.columns(2)
        m1.metric("Call price", f"${c:.4f}")
        m2.metric("Put price", f"${p:.4f}")

        st.markdown("**The Greeks**")
        greeks_df = pd.DataFrame({
            "Call": all_greeks(S, K, T, r, sigma, kind="call"),
            "Put": all_greeks(S, K, T, r, sigma, kind="put"),
        })
        st.dataframe(greeks_df.round(4), use_container_width=True)

        spot_range = np.linspace(S * 0.5, S * 1.5, 200)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(spot_range, np.maximum(spot_range - K, 0) - c, label="Call P&L")
        ax.plot(spot_range, np.maximum(K - spot_range, 0) - p, label="Put P&L")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(K, color="gray", ls=":", label="Strike")
        ax.set_xlabel("Stock price at expiry")
        ax.set_ylabel("Profit / loss")
        ax.legend()
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 3 — Portfolio Optimizer
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Markowitz Portfolio Optimizer")
    col1, col2 = st.columns([1, 2])

    with col1:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL,MSFT,JPM,XOM,JNJ,GLD", key="opt_tickers")
        risk_free = st.slider("Risk-free rate", 0.0, 0.10, 0.04, key="opt_rf")
        run_opt = st.button("Optimize portfolio", type="primary")

    with col2:
        if run_opt:
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            with st.spinner("Fetching data and optimizing..."):
                prices = fetch_prices(tickers)
                mu, cov = annualized_stats(prices)
                w_ms = max_sharpe_weights(mu, cov, risk_free)
                w_mv = min_variance_weights(mu, cov)

            weights_df = pd.DataFrame(
                {"Max Sharpe": w_ms, "Min Variance": w_mv}, index=prices.columns)
            st.bar_chart(weights_df, height=260)

            for name, w in [("Max Sharpe", w_ms), ("Min Variance", w_mv)]:
                ret, vol = portfolio_performance(w, mu, cov)
                st.write(f"**{name}** — return: {ret:.2%} · vol: {vol:.2%} · "
                        f"Sharpe: {(ret - risk_free) / vol:.2f}")

            rng = np.random.default_rng(0)
            rand_w = rng.dirichlet(np.ones(len(mu)), size=1500)
            rand_ret = rand_w @ mu.values
            rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov.values, rand_w))

            targets = np.linspace(mu.min(), mu.max(), 40)
            frontier_vol = [
                portfolio_performance(
                    min_variance_weights(mu, cov, target_return=t), mu, cov)[1]
                for t in targets
            ]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(rand_vol, rand_ret, s=6, alpha=0.3, label="Random portfolios")
            ax.plot(frontier_vol, targets, "r-", lw=2, label="Efficient frontier")
            ms_ret, ms_vol = portfolio_performance(w_ms, mu, cov)
            ax.scatter([ms_vol], [ms_ret], marker="*", s=250, c="gold",
                      edgecolors="black", label="Max Sharpe")
            ax.set_xlabel("Annualized volatility")
            ax.set_ylabel("Annualized expected return")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Enter tickers and optimize to see the allocation and frontier.")

# ---------------------------------------------------------------------------
# TAB 4 — Monte Carlo
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Monte Carlo Simulator")
    mode = st.radio("Mode", ["Option pricing", "Portfolio risk"], horizontal=True)

    if mode == "Option pricing":
        col1, col2 = st.columns([1, 2])
        with col1:
            S2 = st.number_input("Spot (S)", value=100.0, key="mc_S")
            K2 = st.number_input("Strike (K)", value=105.0, key="mc_K")
            T2 = st.slider("Time to expiry (yrs)", 0.01, 2.0, 0.5, key="mc_T")
            r2 = st.slider("Risk-free rate", 0.0, 0.10, 0.04, key="mc_r")
            sigma2 = st.slider("Volatility", 0.01, 1.0, 0.25, key="mc_sigma")
            n_sims = st.select_slider(
                "Simulations", options=[100, 1_000, 10_000, 100_000, 1_000_000],
                value=100_000)
            kind = st.radio("Type", ["call", "put"], horizontal=True)
        with col2:
            price_analytic = (call_price if kind == "call" else put_price)(
                S2, K2, T2, r2, sigma2)
            price_mc, se, ST, _ = mc_option_price(
                S2, K2, T2, r2, sigma2, kind, n_sims=n_sims)
            m1, m2 = st.columns(2)
            m1.metric("Black-Scholes price", f"${price_analytic:.4f}")
            m2.metric("Monte Carlo price", f"${price_mc:.4f}",
                      f"± {1.96*se:.4f} (95% CI)")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(ST, bins=100, density=True, alpha=0.7, color="steelblue")
            ax.axvline(K2, color="red", ls="--", label="Strike")
            ax.axvline(S2, color="gray", ls=":", label="Spot")
            ax.set_xlabel("Simulated terminal price")
            ax.legend()
            st.pyplot(fig)

    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            tickers_input2 = st.text_input(
                "Tickers (comma-separated)",
                value="AAPL,MSFT,JPM,XOM,JNJ,GLD", key="mc_tickers")
            risk_free2 = st.slider("Risk-free rate", 0.0, 0.10, 0.04, key="mc_rf")
            n_sims2 = st.slider("Simulated paths", 500, 10_000, 3000, step=500)
            run_mc = st.button("Run simulation", type="primary")
        with col2:
            if run_mc:
                tickers2 = [t.strip().upper() for t in tickers_input2.split(",") if t.strip()]
                with st.spinner("Fetching data and simulating..."):
                    prices2 = fetch_prices(tickers2)
                    mu2, cov2 = annualized_stats(prices2)
                    w2 = max_sharpe_weights(mu2, cov2, risk_free2)
                    rets2 = daily_returns(prices2)
                    paths = simulate_paths(
                        w2, rets2.mean().values, rets2.cov().values, n_sims=n_sims2)
                    report = risk_report(paths)

                st.write("**Max-Sharpe weights:**",
                        dict(zip(prices2.columns, np.round(w2, 3))))
                cols = st.columns(len(report))
                for c, (k, v) in zip(cols, report.items()):
                    c.metric(k, f"{v:.2%}")

                fig, ax = plt.subplots(figsize=(7, 4))
                idx = np.random.default_rng(2).choice(
                    len(paths), min(150, len(paths)), replace=False)
                ax.plot(paths[idx].T, lw=0.5, alpha=0.3, color="steelblue")
                ax.plot(paths.mean(axis=0), lw=2, color="darkred", label="Mean path")
                ax.set_xlabel("Trading day")
                ax.set_ylabel("Portfolio value ($)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Enter tickers and run the simulation to see the risk report.")
