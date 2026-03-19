"""
intelligence/segment_scorer.py — Daily market intelligence

INTENT:
    Scores all market segments every morning and identifies which segment,
    instrument, and direction (long/short) has the highest probability of
    a profitable trade today. This is what the agent reads at 9:00 AM.

IMPACT:
    Without this, the agent trades every instrument every day. With it,
    the agent concentrates capital in the 1-2 highest-conviction opportunities.

FUNCTIONS:
    - score_segments(): Rank all segments by relative strength + momentum
    - identify_top_candidates(): Select top N instruments from top segment
    - market_context(): VIX-based market health check

OWNED BY: Phase 2 — Market Intelligence
LAST UPDATED: 2026-03-18
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    import ta as _ta_lib  # fallback to 'ta' library — used only for basic indicators in scorer
from loguru import logger
from datetime import date, timedelta

from data.universe import (
    Instrument, get_full_universe, get_equity_universe,
    get_index_universe, get_commodity_universe, ALL_SEGMENTS,
    BROAD_INDICES
)
from data.fetcher import fetch_ohlcv


@dataclass
class SegmentScore:
    segment:        str
    score:          float       # 0–100
    momentum_1d:    float       # 1-day return
    momentum_5d:    float       # 5-day return
    momentum_20d:   float       # 20-day return
    relative_str:   float       # vs Nifty 50
    volatility:     float       # 20-day ATR %
    direction:      str         # bull | bear | neutral
    instruments:    list[str]   # Top instruments in this segment


@dataclass
class InstrumentCandidate:
    symbol:         str
    segment:        str
    score:          float
    direction:      str         # long | short
    entry_zone:     float       # Approximate entry level
    stop_estimate:  float
    target_estimate:float
    reason:         str


def market_context(vix_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Reads India VIX to determine market health.

    Returns:
        regime: 'low_vol' | 'normal' | 'elevated' | 'fear'
        tradeable: bool (False during extreme fear)
        vix: current VIX value
    """
    if vix_df is None or vix_df.empty:
        return {"regime": "normal", "tradeable": True, "vix": None}

    current_vix = float(vix_df["close"].iloc[-1])

    if current_vix < 13:
        regime = "low_vol"
    elif current_vix < 18:
        regime = "normal"
    elif current_vix < 25:
        regime = "elevated"
    else:
        regime = "fear"

    return {
        "regime": regime,
        "tradeable": current_vix < 30,  # Above 30 = do not trade
        "vix": current_vix,
        "vix_note": f"India VIX at {current_vix:.1f} — {regime} volatility environment"
    }


def _momentum_score(returns_1d: float, returns_5d: float, returns_20d: float) -> float:
    """Composite momentum score across 3 timeframes."""
    # Weight: recent > short > medium
    score = (returns_1d * 50) + (returns_5d * 30) + (returns_20d * 20)
    return float(np.tanh(score * 10) * 50 + 50)  # Scale 0–100


def score_segments(
    lookback_days: int = 30,
    reference_symbol: str = "NIFTY",
) -> list[SegmentScore]:
    """
    Scores each segment based on relative momentum vs Nifty.

    WHY: Money rotates between sectors. On any given day, 2-3 sectors are
    outperforming the index. Concentrating trades in those sectors captures
    institutional flows that drive sustained moves.
    """
    from_date = (date.today() - timedelta(days=lookback_days + 10)).isoformat()
    to_date   = date.today().isoformat()

    # Fetch Nifty as benchmark
    nifty_instr = next((i for i in BROAD_INDICES if i.symbol == reference_symbol), None)
    nifty_df = None
    if nifty_instr:
        nifty_df = fetch_ohlcv(nifty_instr, interval="1d", from_date=from_date, to_date=to_date)

    segment_instruments: dict[str, list[Instrument]] = {}
    for instr in get_equity_universe():
        if instr.segment:
            segment_instruments.setdefault(instr.segment, []).append(instr)

    scores = []
    for segment, instruments in segment_instruments.items():
        segment_returns_1d  = []
        segment_returns_5d  = []
        segment_returns_20d = []
        segment_volatilities = []
        top_symbols = []

        for instr in instruments[:5]:  # Sample up to 5 per segment
            df = fetch_ohlcv(instr, interval="1d", from_date=from_date, to_date=to_date)
            if df is None or len(df) < 22:
                continue

            close = df["close"]
            r1d  = float((close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2  else 0)
            r5d  = float((close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6  else 0)
            r20d = float((close.iloc[-1] / close.iloc[-21]- 1) if len(close) >= 21 else 0)
            atr  = float(ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1] or 0)
            vol  = atr / close.iloc[-1] * 100

            segment_returns_1d.append(r1d)
            segment_returns_5d.append(r5d)
            segment_returns_20d.append(r20d)
            segment_volatilities.append(vol)
            top_symbols.append(instr.symbol)

        if not segment_returns_1d:
            continue

        avg_r1d  = float(np.mean(segment_returns_1d))
        avg_r5d  = float(np.mean(segment_returns_5d))
        avg_r20d = float(np.mean(segment_returns_20d))
        avg_vol  = float(np.mean(segment_volatilities))

        # Relative strength vs Nifty
        rel_str = avg_r5d
        if nifty_df is not None and len(nifty_df) >= 6:
            nifty_5d = float(nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-6] - 1)
            rel_str = avg_r5d - nifty_5d

        score = _momentum_score(avg_r1d, avg_r5d, avg_r20d)

        direction = "bull" if avg_r5d > 0.005 else ("bear" if avg_r5d < -0.005 else "neutral")

        scores.append(SegmentScore(
            segment=segment,
            score=score,
            momentum_1d=avg_r1d,
            momentum_5d=avg_r5d,
            momentum_20d=avg_r20d,
            relative_str=rel_str,
            volatility=avg_vol,
            direction=direction,
            instruments=top_symbols,
        ))

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


def identify_top_candidates(
    segment_scores: list[SegmentScore],
    top_segments: int = 2,
    candidates_per_segment: int = 3,
) -> list[InstrumentCandidate]:
    """
    From the top-scored segments, identify specific instruments to trade.

    WHY: Segment momentum tells us WHERE to look; instrument-level analysis
    tells us WHICH stock within that segment has the cleanest setup.
    We look for stocks near support (for longs) or resistance (for shorts).
    """
    universe = {i.symbol: i for i in get_equity_universe()}
    candidates = []

    for seg_score in segment_scores[:top_segments]:
        for symbol in seg_score.instruments[:candidates_per_segment]:
            instr = universe.get(symbol)
            if instr is None:
                continue

            from_date = (date.today() - timedelta(days=30)).isoformat()
            df = fetch_ohlcv(instr, interval="1d", from_date=from_date, to_date=date.today().isoformat())
            if df is None or len(df) < 10:
                continue

            close = df["close"]
            atr   = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
            rsi   = ta.rsi(close, length=14).iloc[-1]
            ema9  = ta.ema(close, length=9).iloc[-1]
            ema21 = ta.ema(close, length=21).iloc[-1]

            if any(pd.isna(v) for v in [atr, rsi, ema9, ema21]):
                continue

            current_price = float(close.iloc[-1])

            # Determine direction and setup quality
            if seg_score.direction == "bull" and ema9 > ema21 and rsi < 70:
                direction = "long"
                entry = current_price
                stop  = entry - atr * 1.5
                target = entry + atr * 3.0
                reason = f"Bull segment ({seg_score.segment}), RSI={rsi:.0f}, EMA bullish"
                score  = seg_score.score * (1 - rsi / 200)  # Penalise overbought
            elif seg_score.direction == "bear" and ema9 < ema21 and rsi > 30:
                direction = "short"
                entry = current_price
                stop  = entry + atr * 1.5
                target = entry - atr * 3.0
                reason = f"Bear segment ({seg_score.segment}), RSI={rsi:.0f}, EMA bearish"
                score  = seg_score.score * (rsi / 200)
            else:
                continue

            candidates.append(InstrumentCandidate(
                symbol=symbol,
                segment=seg_score.segment,
                score=score,
                direction=direction,
                entry_zone=entry,
                stop_estimate=stop,
                target_estimate=target,
                reason=reason,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
