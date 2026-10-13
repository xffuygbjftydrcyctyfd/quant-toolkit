"""
Stock Planner live-price backend.

Serves the web app plus a live price feed:
  GET  /api/prices   -> { asOf, source, prices: { TICKER: {p, pc, dc} } }
  WS   /ws/prices    -> same payload pushed every 2 seconds
  GET  /             -> the web app itself (static mount)

Price sources, chosen automatically:
  - Alpaca (real-time) when ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
    are set: polls the snapshots endpoint every 3s for latest trade price
    and previous daily close (for day-change).
  - Simulated feed otherwise: geometric Brownian motion steps around the
    last closes in stocks.json, with per-asset volatility from the
    covariance diagonal — so the whole system is testable with no keys.

Run:  uvicorn stock_planner.server.main:app --port 8000
Then open http://localhost:8000
API keys never touch the browser — that's the entire reason this
backend exists rather than calling Alpaca from the frontend.
"""

import asyncio
import json
import math
import os
import random
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parents[1] / "web_app"
DATA = json.load(open(WEB_DIR / "stocks.json"))
TICKERS = DATA["tickers"]

app = FastAPI(title="Stock Planner feed")

# Shared price cache: {ticker: {"p": last, "pc": prev_close, "dc": day_change}}
CACHE = {"asOf": None, "source": None, "prices": {}}
CLIENTS: set[WebSocket] = set()


# ----------------------------- price sources -----------------------------
class SimulatedFeed:
    """GBM around last closes; annual vol from the covariance diagonal."""
    name = "simulated"
    interval = float(os.environ.get("FEED_INTERVAL", "2.0"))

    def __init__(self):
        self.prev_close = dict(zip(TICKERS, DATA["prices"]))
        self.prices = dict(self.prev_close)
        self.vols = {t: math.sqrt(max(DATA["cov"][i][i], 1e-6))
                     for i, t in enumerate(TICKERS)}
        self.rng = random.Random(7)
        # per-tick sd: annual vol scaled to a 2-second step of a 6.5h day
        self.step_sd = {t: v / math.sqrt(252 * 6.5 * 3600 / self.interval)
                        for t, v in self.vols.items()}

    async def tick(self):
        for t in TICKERS:
            z = self.rng.gauss(0, 1)
            self.prices[t] = max(0.01, self.prices[t]
                                 * math.exp(self.step_sd[t] * z))
        # raw (unrounded) prices — rounding happens at broadcast time,
        # otherwise quantization noise corrupts realized-vol estimates
        return {t: {"p": self.prices[t],
                    "pc": self.prev_close[t],
                    "dc": self.prices[t] / self.prev_close[t] - 1}
                for t in TICKERS}


class AlpacaFeed:
    """Polls Alpaca's snapshot endpoint (free tier friendly)."""
    name = "alpaca"
    interval = 3.0
    URL = "https://data.alpaca.markets/v2/stocks/snapshots"

    def __init__(self, key, secret):
        self.headers = {"APCA-API-KEY-ID": key,
                        "APCA-API-SECRET-KEY": secret}
        self.client = httpx.AsyncClient(timeout=10)

    async def tick(self):
        out = {}
        # snapshot endpoint accepts comma-separated symbols
        r = await self.client.get(
            self.URL, headers=self.headers,
            params={"symbols": ",".join(TICKERS)})
        r.raise_for_status()
        snaps = r.json()
        for t in TICKERS:
            s = snaps.get(t) or {}
            trade = (s.get("latestTrade") or {}).get("p")
            prev = (s.get("prevDailyBar") or {}).get("c")
            if trade and prev:
                out[t] = {"p": round(trade, 2), "pc": round(prev, 2),
                          "dc": round(trade / prev - 1, 5)}
        if not out:
            raise RuntimeError("Alpaca returned no usable snapshots")
        return out


# --- intraday realized volatility from the tick stream ---
from collections import deque
TICK_HISTORY = {t: deque(maxlen=300) for t in TICKERS}

def update_realized_vol(prices, interval):
    """Annualized realized vol from rolling log tick returns; None until
    enough ticks. ticks/year = 252 trading days * 6.5h * 3600s / interval."""
    ann = math.sqrt(252 * 6.5 * 3600 / interval)
    out = {}
    for t, q in prices.items():
        h = TICK_HISTORY[t]
        h.append(q["p"])
        if len(h) >= 30:
            lr = [math.log(h[i] / h[i - 1]) for i in range(1, len(h))]
            m = sum(lr) / len(lr)
            var = sum((x - m) ** 2 for x in lr) / (len(lr) - 1)
            out[t] = round(math.sqrt(var) * ann, 4)
        else:
            out[t] = None
    return out


# --- positions source (Step: brokerage read) ---
class SimulatedPositions:
    """Demo holdings so the import flow is testable without keys."""
    async def get(self):
        picks = [(TICKERS[i], qty) for i, qty in
                 [(0, 12), (7, 25), (13, 30), (42, 40)]]
        px = dict(zip(TICKERS, DATA["prices"]))
        return [{"symbol": t, "qty": q,
                 "avg_entry_price": round(px[t] * 0.92, 2)}
                for t, q in picks]


class AlpacaPositions:
    URL = "https://api.alpaca.markets/v2/positions"
    PAPER_URL = "https://paper-api.alpaca.markets/v2/positions"

    def __init__(self, key, secret, paper=True):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.url = self.PAPER_URL if paper else self.URL
        self.client = httpx.AsyncClient(timeout=10)

    async def get(self):
        r = await self.client.get(self.url, headers=self.headers)
        r.raise_for_status()
        return [{"symbol": p["symbol"], "qty": float(p["qty"]),
                 "avg_entry_price": float(p["avg_entry_price"])}
                for p in r.json()]


def choose_positions():
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        paper = os.environ.get("ALPACA_PAPER", "1") != "0"
        return AlpacaPositions(key, secret, paper)
    return SimulatedPositions()


POSITIONS = choose_positions()


def choose_feed():
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        print("[feed] Using Alpaca real-time snapshots")
        return AlpacaFeed(key, secret)
    print("[feed] No Alpaca keys in env; using simulated feed "
          "(set ALPACA_API_KEY / ALPACA_SECRET_KEY for real data)")
    return SimulatedFeed()


# ----------------------------- feed loop -----------------------------
async def feed_loop():
    feed = choose_feed()
    while True:
        try:
            raw = await feed.tick()
            rv = update_realized_vol(raw, feed.interval)
            prices = {t: {"p": round(q["p"], 2), "pc": round(q["pc"], 2),
                          "dc": round(q["dc"], 5), "rv": rv[t]}
                      for t, q in raw.items()}
            CACHE["prices"] = prices
            CACHE["asOf"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            CACHE["source"] = feed.name
            payload = json.dumps(CACHE)
            dead = set()
            for ws in CLIENTS:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            CLIENTS.difference_update(dead)
        except Exception as e:
            print(f"[feed] tick failed ({e.__class__.__name__}: {e}); retrying")
        await asyncio.sleep(feed.interval)


@app.on_event("startup")
async def startup():
    asyncio.create_task(feed_loop())


# ----------------------------- endpoints -----------------------------
@app.get("/api/prices")
async def get_prices():
    return CACHE


@app.websocket("/ws/prices")
async def ws_prices(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    try:
        if CACHE["prices"]:
            await ws.send_text(json.dumps(CACHE))
        while True:
            await ws.receive_text()  # keepalive; broadcasts come from feed_loop
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)


@app.get("/api/positions")
async def get_positions():
    """Current brokerage positions (Alpaca with env keys; simulated demo
    otherwise). The browser never sees API keys — only this response."""
    try:
        pos = await POSITIONS.get()
        src = "alpaca" if isinstance(POSITIONS, AlpacaPositions) else "simulated"
        return {"source": src, "positions": pos}
    except Exception as e:
        return {"source": "error", "error": e.__class__.__name__, "positions": []}


# Static mount last so /api and /ws take precedence
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="app")
