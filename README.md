# Quant Finance Toolkit

I wanted to understand how financial models actually work, so I built them instead of just reading about them. This is the result, a set of tools that price options, build portfolios, simulate risk, and test trading strategies, all written from scratch in Python.

The thing I care about most here is that nothing trusts itself. The options pricer checks its own math with put-call parity, the Greeks are verified against numerical differentiation, and the Monte Carlo engine independently reproduces the Black-Scholes price it should get. If two different methods agree, I believe the answer. That habit of making a model prove itself is the whole point of the project.

This is an educational project. It is not investment advice, and the outputs are model results under stated assumptions, not recommendations to buy or sell anything.

## What's in here

```
quant-toolkit/
├── shared/data_utils.py      data pulls, with an offline synthetic fallback
├── backtesting_engine/       strategy simulation, Sharpe, drawdown, win rate
├── options_pricer/           Black-Scholes, the Greeks, implied vol, market comparison
├── portfolio_optimizer/      Markowitz mean-variance and the efficient frontier
├── monte_carlo/              option pricing and portfolio risk by simulation
├── goal_planner/             a goal-based planner that reports probabilities
└── stock_planner/            the full version, real stocks, live prices, a web app
```

## The core tools

**Backtesting engine.** You hand it a price series and a strategy, and it tells you whether that strategy would actually have made money. It trades the signal one bar late so it can't peek at the future, and it charges transaction costs, because ignoring those is the classic way a backtest lies to you. Out comes CAGR, Sharpe, max drawdown, and win rate, next to plain buy-and-hold so you can see if the strategy was even worth it. Comes with a few sample strategies, moving-average crossover, momentum, and mean reversion.

**Options pricer.** Black-Scholes written out term by term, not imported from a library, plus all five Greeks, implied volatility, and a comparison of the model's prices against a real option chain. That last part is my favorite, because it surfaces the volatility smile, which is really just the market politely disagreeing with one of Black-Scholes' core assumptions.

**Portfolio optimizer.** Classic Markowitz mean-variance. You give it tickers, it finds the best mix for a given level of risk and draws the efficient frontier, with a few thousand random portfolios scattered underneath so you can actually see what the optimization buys you.

**Monte Carlo simulator.** Instead of solving for one answer, it runs thousands of possible futures. In option mode it prices an option by simulation and shows the estimate converging on the Black-Scholes number. In portfolio mode it takes the optimizer's best portfolio and simulates a year of it thousands of times, so you get the whole range of outcomes and the ugly tail, not a single tidy forecast.

## The apps

A few of these grew into things you can actually click through.

**Portfolio optimizer (web).** `portfolio_optimizer/web_app/index.html` is a single self-contained page. It solves the Markowitz frontier right in the browser, no server, so it loads instantly every time. Shows clearly labeled sample data by default, and swaps in real market data if you run the fetch script.

**Goal planner.** `goal_planner/web_app/index.html` asks what you're saving for and how much risk you can stomach, then tells you the *probability* of getting there, with a fan chart of where you might land, rather than one falsely precise number. It also shows how much the "optimal" portfolio shifts when you nudge the assumptions, because pretending the inputs are certain is exactly how this kind of model fools people. Every formula in it is checked against a Python reference in `goal_planner/reference.py`.

**Stock planner.** This is the full version, in `stock_planner/`. Real tickers across defense, oil and gas, healthcare, utilities, retail, consumer staples, and agriculture, plus bond and T-bill ETFs to steady things. It runs a five-step flow, goal, sector preferences, a risk questionnaire, a disclaimer, then results, and for every stock it picks it shows why, in plain language, along with how sure it is. There's a live-price backend, drift tracking against your targets, and the whole thing installs on a phone like an app. Everything stays in your browser, and it's framed throughout as model output under stated assumptions, not advice.

A couple of notes on the stock planner internals, for anyone reading the code:

- The data pipeline pulls five years of history and cleans up the statistics with Ledoit-Wolf shrinkage, because a raw covariance matrix from that much data is too noisy to optimize on, and it shrinks expected returns toward sector averages for the same reason. Falls back to a seeded synthetic model when there's no internet, so it always runs.
- The optimizer is long-only with caps on how much can go into any one stock or sector, solved in the browser and checked against SciPy to make sure the browser version gets the same answer.
- The live backend keeps API keys on the server, never in the browser, which is the only reason it exists. Without keys it runs a simulated feed so you can still try everything.

## Running it

```bash
pip install -r requirements.txt

python -m backtesting_engine.engine
python -m options_pricer.black_scholes
python -m options_pricer.greeks
python -m options_pricer.market_comparison
python -m portfolio_optimizer.markowitz
python -m portfolio_optimizer.efficient_frontier
python -m monte_carlo.option_simulation
python -m monte_carlo.portfolio_simulation
```

If you're online it pulls real market data. If you're not, it quietly falls back to synthetic price paths so everything still runs and nothing breaks.

## A few things I decided on purpose

- Backtest signals from day *t* get traded on day *t+1*, never the same day, so nothing gets to cheat with hindsight.
- Transaction costs are always included.
- Everything checks itself: parity on the options, analytic-versus-numerical on the Greeks, simulation-versus-formula on the Monte Carlo.
- One shared data layer underneath all of it, so the optimizer and the risk simulator are always looking at the same world.
