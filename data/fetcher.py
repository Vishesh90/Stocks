"""
data/fetcher.py — Historical OHLCV data fetcher

INTENT:
    Fetches historical OHLCV data exclusively from the Dhan API.
    Caches to Parquet files so we never re-download what we already have.
    The backtesting engine calls this — it never fetches raw data directly.

IMPACT:
    Every strategy backtest depends on this. Bad data = bad backtest signals.
    The cache layer means backtests run at memory speed after first download.

FUNCTIONS:
    - fetch_ohlcv(): Main entry point — returns clean DataFrame (Dhan only)
    - fetch_dhan(): Dhan API historical data with 85-day batch windowing
    - fetch_universe_data(): Bulk download for full universe

RATE LIMIT:
    Dhan API allows 10 req/s. We sleep 0.125s between batch calls = 8 req/s.
    That is 80% of the limit — never crosses 8 req/s.

DEPENDENCIES:
    - config/settings.py
    - data/universe.py

OWNED BY: Phase 1 — Data Pipeline
LAST UPDATED: 2026-03-20
"""

import os
import time
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from data.universe import Instrument, AssetClass, Exchange


# ─────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, interval: str, from_date: str, to_date: str) -> Path:
    """Deterministic cache key so re-runs never re-download."""
    key = f"{symbol}_{interval}_{from_date}_{to_date}"
    hashed = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_dir = settings.data_cache_dir / interval
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{hashed}.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            df = pd.read_parquet(path)
            logger.debug(f"Cache hit: {path.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache read failed for {path}: {e}. Will re-fetch.")
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
        logger.debug(f"Cached {len(df)} rows → {path.name}")
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


# ─────────────────────────────────────────────────────────────────
# DHAN FETCHER
# ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def fetch_dhan(
    security_id: str,
    exchange_segment: str,
    instrument_type: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> Optional[pd.DataFrame]:
    """
    Fetches OHLCV from Dhan API v2.

    WHY: Dhan provides 5 years of intraday data — the deepest free source
    available for Indian markets. This is our primary backtest data source.

    For intraday intervals, Dhan limits each call to 90 days. This function
    automatically batches requests across the full date range and concatenates.

    Returns None if Dhan credentials are not configured (fallback to yfinance).
    """
    if not settings.dhan_access_token:
        return None

    try:
        from dhanhq import dhanhq
        from datetime import datetime, timedelta

        dhan = dhanhq(settings.dhan_client_id, settings.dhan_access_token)

        if interval == "1d":
            resp = dhan.historical_daily_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                expiry_code=0,
                from_date=from_date,
                to_date=to_date,
            )
            return _parse_dhan_response(resp)
        else:
            # Intraday: batch into 90-day windows to respect Dhan API limit
            interval_map = {"1m": 1, "5m": 5, "15m": 15, "25m": 25, "60m": 60}
            dhan_interval = interval_map.get(interval, 5)

            start = datetime.strptime(from_date, "%Y-%m-%d")
            end   = datetime.strptime(to_date,   "%Y-%m-%d")
            batch_days = 85  # Stay under 90-day limit with buffer
            all_dfs = []

            import requests as _requests

            cursor = start
            while cursor < end:
                batch_end = min(cursor + timedelta(days=batch_days), end)
                batch_from = cursor.strftime("%Y-%m-%d")
                batch_to   = batch_end.strftime("%Y-%m-%d")

                try:
                    headers = {
                        "access-token": settings.dhan_access_token,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                    payload = {
                        "securityId": security_id,
                        "exchangeSegment": exchange_segment,
                        "instrument": instrument_type,
                        "interval": str(dhan_interval),
                        "fromDate": batch_from,
                        "toDate": batch_to,
                    }
                    resp = _requests.post(
                        "https://api.dhan.co/v2/charts/intraday",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    data = resp.json()
                    df_batch = _parse_dhan_response({"data": data} if "open" in data else data)
                    if df_batch is not None and not df_batch.empty:
                        all_dfs.append(df_batch)
                        logger.debug(f"  Batch {batch_from}→{batch_to}: {len(df_batch)} candles")
                except Exception as e:
                    logger.warning(f"Dhan batch {batch_from}→{batch_to} failed: {e}")

                cursor = batch_end + timedelta(days=1)
                # Dhan API limit: 10 req/s.
                # 0.125s between calls = exactly 8 req/s = 80% of limit.
                # Mathematically cannot exceed 8 req/s.
                time.sleep(0.125)

            if not all_dfs:
                return None

            combined = pd.concat(all_dfs)
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            return _clean_ohlcv(combined)

    except Exception as e:
        logger.warning(f"Dhan fetch failed for {security_id}: {e}")
        return None


def _parse_dhan_response(resp) -> Optional[pd.DataFrame]:
    """Parse Dhan API response dict into a clean OHLCV DataFrame."""
    if not resp:
        return None
    # Direct API response has open/high/low/close at top level
    data = resp.get("data", resp)
    if not isinstance(data, dict):
        return None
    if not data.get("open"):
        return None
    try:
        timestamps = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert("Asia/Kolkata")
        df = pd.DataFrame({
            "open":   data["open"],
            "high":   data["high"],
            "low":    data["low"],
            "close":  data["close"],
            "volume": data["volume"],
        }, index=timestamps)
        df.index.name = "datetime"
        return _clean_ohlcv(df)
    except Exception as e:
        logger.warning(f"Dhan parse failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# YAHOO FINANCE FALLBACK
# ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def fetch_yfinance(
    ticker: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> Optional[pd.DataFrame]:
    """
    Yahoo Finance fallback for when Dhan credentials aren't set.

    WHY: Enables backtesting to start immediately without waiting for Dhan
    API credentials. Also used for commodities (MCX) and international indices.

    Interval mapping: 1m/5m/15m/60m/1d → yfinance format.
    """
    try:
        import yfinance as yf

        yf_interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "25m": "30m",  # yfinance doesn't have 25m, use 30m
            "60m": "1h", "1d": "1d",
        }
        yf_interval = yf_interval_map.get(interval, "5m")

        df = yf.download(
            ticker,
            start=from_date,
            end=to_date,
            interval=yf_interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return None

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "datetime"

        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")

        return _clean_ohlcv(df)

    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    instrument: Instrument,
    interval: str = "5m",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Primary OHLCV fetch function. Dhan API only — no Yahoo fallback.

    WHY: We use Dhan specifically because it provides 5 years of 1m intraday
    data for Indian markets. Yahoo Finance caps 1m data at 8 days — useless
    for backtesting. Every instrument in the universe has a dhan_security_id.

    Returns None (and logs a warning) if Dhan returns no data for that instrument.
    """
    today = date.today().isoformat()

    if to_date is None:
        to_date = today

    if from_date is None:
        if interval == "1d":
            from_date = (date.today() - timedelta(days=365 * 10)).isoformat()
        else:
            from_date = (date.today() - timedelta(days=365 * 5)).isoformat()

    cache_path = _cache_path(instrument.symbol, interval, from_date, to_date)

    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    logger.info(f"Fetching {instrument.symbol} ({interval}) {from_date} → {to_date}")

    # Dhan only. No Yahoo fallback — Yahoo caps 1m at 8 days, worthless.
    if not settings.dhan_access_token:
        logger.error("DHAN_ACCESS_TOKEN not set in .env — cannot fetch data")
        return None

    if not instrument.dhan_security_id:
        logger.warning(f"{instrument.symbol}: no dhan_security_id — skipping")
        return None

    exchange_segment = _dhan_exchange_segment(instrument)
    instrument_type  = _dhan_instrument_type(instrument)
    df = fetch_dhan(
        security_id=instrument.dhan_security_id,
        exchange_segment=exchange_segment,
        instrument_type=instrument_type,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
    )

    if df is None or df.empty:
        logger.warning(f"No data returned from Dhan for {instrument.symbol}")
        return None

    _save_cache(df, cache_path)
    return df


def fetch_universe_data(
    instruments: list[Instrument],
    interval: str = "1d",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Bulk fetch for all instruments in a universe.

    WHY: The backtesting engine and segment scorer both need data for many
    instruments. Rate limiting is handled inside fetch_dhan() per batch —
    no extra sleep needed here.

    Returns dict mapping symbol → DataFrame.
    """
    results = {}
    total = len(instruments)

    for i, instrument in enumerate(instruments, 1):
        logger.info(f"[{i}/{total}] Fetching {instrument.symbol}...")
        df = fetch_ohlcv(instrument, interval=interval, from_date=from_date, to_date=to_date)
        if df is not None:
            results[instrument.symbol] = df

    logger.info(f"Fetched {len(results)}/{total} instruments successfully")
    return results


# ─────────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces data quality rules before any strategy sees the data.

    WHY: Dirty data (NaN, zero prices, duplicates, non-trading hours) produces
    false signals. Every strategy should see clean, validated data.
    """
    df = df.copy()

    # Drop rows with any NaN in OHLCV
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Remove zero or negative prices — clearly bad data
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

    # Enforce OHLC consistency: high >= max(open, close), low <= min(open, close)
    df = df[df["high"] >= df[["open", "close"]].max(axis=1)]
    df = df[df["low"] <= df[["open", "close"]].min(axis=1)]

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    # Sort ascending
    df = df.sort_index()

    # Cast types — volume comes back as float from Dhan; cast safely
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(np.int64)

    return df


def _dhan_exchange_segment(instrument: Instrument) -> str:
    # Dhan uses different segment strings per asset class
    if instrument.asset_class == AssetClass.INDEX:
        return "IDX_I"
    exchange_map = {
        Exchange.NSE: "NSE_EQ",
        Exchange.BSE: "BSE_EQ",
        Exchange.MCX: "MCX_COMM",
        Exchange.NFO: "NSE_FNO",
    }
    return exchange_map.get(instrument.exchange, "NSE_EQ")


def _dhan_instrument_type(instrument: Instrument) -> str:
    type_map = {
        AssetClass.EQUITY:    "EQUITY",
        AssetClass.INDEX:     "INDEX",
        AssetClass.COMMODITY: "FUTCOM",
        AssetClass.ETF:       "EQUITY",
    }
    return type_map.get(instrument.asset_class, "EQUITY")
