from datetime import datetime, timezone
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_NAME = "Binance Futures Validation Chain Engine"
BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
CANDLE_MS = 300_000

# Conservative research assumptions only; adjust to your actual fee tier.
DEFAULT_FEE_PER_SIDE = 0.0005
DEFAULT_SLIPPAGE_BPS_PER_SIDE = 2.0
MIN_OOS_TRADES = 30
MIN_OOS_PROFIT_FACTOR = 1.05
MAX_OOS_DRAWDOWN = 0.20

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Public Binance USDⓈ-M Futures market data. Signal-only. No order endpoints or API keys."
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class BacktestRequest(BaseModel):
    days: int = Field(default=30, ge=10, le=365)
    fee_per_side: float = Field(default=DEFAULT_FEE_PER_SIDE, ge=0, le=0.01)
    slippage_bps_per_side: float = Field(default=DEFAULT_SLIPPAGE_BPS_PER_SIDE, ge=0, le=100)


class ValidateRequest(BacktestRequest):
    train_days: int = Field(default=14, ge=7, le=180)
    test_days: int = Field(default=5, ge=3, le=60)
    monte_carlo_paths: int = Field(default=1000, ge=100, le=10000)


def api_get(path: str, params: Optional[dict] = None):
    try:
        response = requests.get(BASE_URL + path, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Binance API request failed: {exc}") from exc


def fetch_candles(days: int = 3) -> pd.DataFrame:
    """Fetch real closed 5m candles via official public Futures API. No synthetic fallback."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = api_get("/fapi/v1/klines", {
            "symbol": SYMBOL, "interval": INTERVAL,
            "startTime": cursor, "endTime": end_ms, "limit": 1500
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + CANDLE_MS
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1500:
            break
        time.sleep(0.08)

    if not rows:
        raise HTTPException(status_code=502, detail="Binance returned no candles. No demo data is used.")
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    numeric = ["open_time", "close_time", "open", "high", "low", "close", "volume",
               "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "close_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    server_time = int(api_get("/fapi/v1/time")["serverTime"])
    df = df[df["close_time"] < server_time].copy()
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=502, detail="No fully closed candles received from Binance.")
    return df


def quality_report(df: pd.DataFrame) -> dict:
    if len(df) < 100:
        return {"valid": False, "status": "INSUFFICIENT_DATA", "reason": "Kurang dari 100 candle tertutup.", "gaps": None}
    gaps = int((np.diff(df["open_time"].to_numpy()) != CANDLE_MS).sum())
    last_close = int(df["close_time"].iloc[-1])
    server_now = int(api_get("/fapi/v1/time")["serverTime"])
    age_s = max(0, (server_now - last_close) / 1000)
    stale = age_s > 12 * 60
    reasons = []
    if gaps:
        reasons.append(f"Ada {gaps} gap candle")
    if stale:
        reasons.append("Data candle terakhir terlambat")
    valid = gaps == 0 and not stale
    return {
        "valid": valid, "status": "PASS" if valid else "FAIL",
        "reason": "Data dasar lolos pemeriksaan" if valid else "; ".join(reasons),
        "gaps": gaps, "latest_closed_candle_utc": df["datetime"].iloc[-1].isoformat(),
        "latest_candle_age_seconds": round(age_s, 1), "candle_count": int(len(df))
    }


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x.close
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi14"] = 100 - 100 / (1 + rs)
    prev = c.shift()
    tr = pd.concat([(x.high - x.low), (x.high - prev).abs(), (x.low - prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["vol_sma20"] = x.volume.rolling(20).mean()
    x["prior_high20"] = x.high.shift(1).rolling(20).max()
    x["prior_low20"] = x.low.shift(1).rolling(20).min()
    x["long_signal"] = (x.ema20 > x.ema50) & (x.rsi14 > 50) & (x.close > x.prior_high20) & (x.volume > x.vol_sma20)
    x["short_signal"] = (x.ema20 < x.ema50) & (x.rsi14 < 50) & (x.close < x.prior_low20) & (x.volume > x.vol_sma20)
    return x


def current_signal(df: pd.DataFrame) -> dict:
    q = quality_report(df)
    x = features(df)
    last = x.iloc[-1]
    base = {
        "symbol": SYMBOL, "interval": INTERVAL, "source": "Binance USDⓈ-M Futures public API",
        "signal": "WAIT", "reason": "", "validation_status": "NOT_RUN",
        "data_quality": q, "candle_utc": last.datetime.isoformat(),
        "price_reference": float(last.close), "indicators": {},
        "execution": "DISABLED", "api_keys_required": False
    }
    for col in ["ema20", "ema50", "rsi14", "atr14", "vol_sma20"]:
        val = last[col]
        base["indicators"][col] = None if pd.isna(val) else float(val)
    if not q["valid"]:
        base["reason"] = "WAIT: pemeriksaan kualitas data gagal."
    else:
        # Safety-first: this scaffold never unlocks directional signals automatically.
        base["reason"] = "WAIT: validasi strategi belum disetujui. Jalankan backtest dan validasi terlebih dahulu."
    return base


def simulate(df: pd.DataFrame, fee: float, slippage_bps: float):
    x = features(df).reset_index(drop=True)
    round_trip_cost = 2 * fee + 2 * slippage_bps / 10_000
    trades, position = [], None
    for i in range(60, len(x) - 1):
        row, nxt = x.iloc[i], x.iloc[i + 1]
        if position:
            exit_price, reason = None, None
            if position["side"] == "LONG":
                if row.low <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif row.high >= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
                elif bool(row.short_signal):
                    exit_price, reason = float(nxt.open), "OPPOSITE_SIGNAL"
            else:
                if row.high >= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif row.low <= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
                elif bool(row.long_signal):
                    exit_price, reason = float(nxt.open), "OPPOSITE_SIGNAL"
            if exit_price is not None:
                raw = ((exit_price - position["entry"]) / position["entry"] if position["side"] == "LONG"
                       else (position["entry"] - exit_price) / position["entry"])
                trades.append({"side": position["side"], "entry_time": position["entry_time"],
                               "exit_time": row.datetime.isoformat(), "entry": position["entry"],
                               "exit": float(exit_price), "net_return": float(raw - round_trip_cost),
                               "exit_reason": reason})
                position = None
        if position is None:
            side = "LONG" if bool(row.long_signal) else "SHORT" if bool(row.short_signal) else None
            if side and pd.notna(row.atr14) and row.atr14 > 0:
                entry = float(nxt.open)
                dist = max(1.5 * float(row.atr14), entry * 0.001)
                position = {"side": side, "entry": entry, "entry_time": nxt.datetime.isoformat(),
                            "sl": entry - dist if side == "LONG" else entry + dist,
                            "tp": entry + 2 * dist if side == "LONG" else entry - 2 * dist}
    if position:
        final = float(x.iloc[-1].close)
        raw = ((final - position["entry"]) / position["entry"] if position["side"] == "LONG"
               else (position["entry"] - final) / position["entry"])
        trades.append({"side": position["side"], "entry_time": position["entry_time"],
                       "exit_time": x.iloc[-1].datetime.isoformat(), "entry": position["entry"],
                       "exit": final, "net_return": float(raw - round_trip_cost), "exit_reason": "END_OF_DATA"})
    returns = np.array([t["net_return"] for t in trades], dtype=float)
    wins, losses = returns[returns > 0], returns[returns < 0]
    eq = np.cumprod(1 + returns) if len(returns) else np.array([])
    if len(eq):
        peaks = np.maximum.accumulate(np.r_[1.0, eq])[1:]
        max_dd = float(np.max((peaks - eq) / peaks))
    else:
        max_dd = 0.0
    gp, gl = float(wins.sum()) if len(wins) else 0.0, float(abs(losses.sum())) if len(losses) else 0.0
    metrics = {
        "trade_count": len(trades),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else None,
        "expectancy_net_pct_per_trade": round(float(returns.mean() * 100), 5) if len(returns) else None,
        "profit_factor": round(gp / gl, 4) if gl else (None if gp == 0 else "infinite"),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "compounded_return_pct": round((float(eq[-1]) - 1) * 100, 3) if len(eq) else None,
        "fee_per_side": fee, "slippage_bps_per_side": slippage_bps,
        "funding_included": False,
        "caveat": "Belum memasukkan funding historis, dampak pasar, likuidasi, atau urutan intra-candle yang pasti."
    }
    return trades, metrics


def mc_bootstrap(returns, paths):
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 5:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "Kurang dari 5 transaksi OOS"}
    rng = np.random.default_rng(21)
    final, dds = [], []
    for _ in range(paths):
        sample = rng.choice(arr, size=len(arr), replace=True)
        equity = np.cumprod(1 + sample)
        peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        final.append(float(equity[-1] - 1))
        dds.append(float(np.max((peaks - equity) / peaks)))
    return {"status": "DIAGNOSTIC_ONLY", "paths": paths,
            "p05_return_pct": round(float(np.quantile(final, .05) * 100), 3),
            "median_return_pct": round(float(np.median(final) * 100), 3),
            "p95_drawdown_pct": round(float(np.quantile(dds, .95) * 100), 3),
            "method": "Bootstrap return independen; tidak mempertahankan dependensi urutan pasar."}


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mode": "SIGNAL_ONLY", "orders_enabled": False,
            "data_source": BASE_URL, "validation": "NOT_RUN"}


@app.get("/api/market")
def market():
    df = fetch_candles(3)
    q = quality_report(df)
    x = features(df).tail(120)
    candles = []
    for _, r in x.iterrows():
        candles.append({"time": int(r.open_time // 1000), "open": float(r.open), "high": float(r.high),
                        "low": float(r.low), "close": float(r.close), "volume": float(r.volume),
                        "ema20": None if pd.isna(r.ema20) else float(r.ema20),
                        "ema50": None if pd.isna(r.ema50) else float(r.ema50)})
    return {"symbol": SYMBOL, "interval": INTERVAL, "source": "Binance USDⓈ-M Futures",
            "quality": q, "last_price": float(df.close.iloc[-1]), "candles": candles,
            "signal": current_signal(df)}


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    df = fetch_candles(req.days)
    if len(df) < 500:
        raise HTTPException(status_code=422, detail=f"Data historis asli tidak cukup: {len(df)} candle.")
    trades, metrics = simulate(df, req.fee_per_side, req.slippage_bps_per_side)
    return {"status": "COMPLETED_RESEARCH_ONLY", "source": "Binance USDⓈ-M Futures",
            "period_start_utc": df.datetime.iloc[0].isoformat(), "period_end_utc": df.datetime.iloc[-1].isoformat(),
            "candles": len(df), "quality": quality_report(df), "metrics": metrics, "trades": trades[-100:]}


@app.post("/api/validate")
def validate(req: ValidateRequest):
    df = fetch_candles(req.days)
    if len(df) < 1000:
        raise HTTPException(status_code=422, detail=f"Data asli belum cukup untuk validasi: {len(df)} candle.")
    # Reserve the final 20% as OOS; earlier data forms rolling chronological test windows.
    oos_n = max(req.test_days * 288, int(len(df) * .20))
    pre_oos = df.iloc[:-oos_n]
    oos = df.iloc[-oos_n:]
    train_n, test_n = req.train_days * 288, req.test_days * 288
    wf_results = []
    cursor = 0
    while cursor + train_n + test_n <= len(pre_oos) and len(wf_results) < 10:
        train = pre_oos.iloc[cursor:cursor + train_n]
        test = pre_oos.iloc[cursor + train_n:cursor + train_n + test_n]
        _, train_metrics = simulate(train, req.fee_per_side, req.slippage_bps_per_side)
        _, test_metrics = simulate(test, req.fee_per_side, req.slippage_bps_per_side)
        wf_results.append({"train_start": train.datetime.iloc[0].isoformat(),
                           "train_end": train.datetime.iloc[-1].isoformat(),
                           "test_start": test.datetime.iloc[0].isoformat(),
                           "test_end": test.datetime.iloc[-1].isoformat(),
                           "train": train_metrics, "test": test_metrics})
        cursor += test_n
    oos_trades, oos_metrics = simulate(oos, req.fee_per_side, req.slippage_bps_per_side)
    mc = mc_bootstrap([t["net_return"] for t in oos_trades], req.monte_carlo_paths)
    failures = []
    pf = oos_metrics["profit_factor"]
    if oos_metrics["trade_count"] < MIN_OOS_TRADES: failures.append(f"Transaksi OOS kurang dari {MIN_OOS_TRADES}")
    if oos_metrics["expectancy_net_pct_per_trade"] is None or oos_metrics["expectancy_net_pct_per_trade"] <= 0:
        failures.append("Expectancy bersih OOS tidak positif")
    if not isinstance(pf, (int, float)) or pf < MIN_OOS_PROFIT_FACTOR:
        failures.append("Profit factor OOS tidak melewati batas")
    if oos_metrics["max_drawdown_pct"] > MAX_OOS_DRAWDOWN * 100:
        failures.append("Drawdown OOS melebihi 20%")
    if mc["status"] != "DIAGNOSTIC_ONLY": failures.append("Monte Carlo kekurangan transaksi")
    return {"status": "FAIL_WAIT" if failures else "RESEARCH_GATES_PASSED_NOT_PROOF_OF_PROFIT",
            "directional_signals_enabled": False, "source": "Binance USDⓈ-M Futures",
            "data_period_start_utc": df.datetime.iloc[0].isoformat(),
            "data_period_end_utc": df.datetime.iloc[-1].isoformat(),
            "walk_forward": wf_results,
            "final_oos": {"start_utc": oos.datetime.iloc[0].isoformat(),
                          "end_utc": oos.datetime.iloc[-1].isoformat(),
                          "metrics": oos_metrics, "trades": oos_trades[-100:]},
            "monte_carlo": mc, "failed_gates": failures,
            "important": "Funding historis belum dimasukkan. Hasil bukan klaim profit. Sinyal LONG/SHORT tetap dikunci; sistem ini mode sinyal saja."}
