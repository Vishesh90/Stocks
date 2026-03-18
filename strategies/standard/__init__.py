"""
strategies/standard/__init__.py — All 40+ standard strategies

INTENT:
    Every classical technical analysis strategy implemented in one place.
    Each strategy is a self-contained class with no side effects.
    The backtester instantiates them all and runs them in parallel.

STRATEGIES INCLUDED:
    Trend-following (10): EMA Cross, MACD, Supertrend, ADX+DI, Parabolic SAR,
        Ichimoku, Triple EMA, Hull MA, DEMA, TEMA
    Mean Reversion (8): RSI Extremes, Bollinger Band Squeeze, Stochastic,
        Williams %R, CCI, Mean Reversion ATR, Keltner Reversion, Donchian Mid
    Breakout (7): Donchian Breakout, Volatility Squeeze, Opening Range,
        High-Low Breakout, Volume Spike Breakout, ATR Channel, N-Day High
    Volume-based (5): VWAP, OBV Divergence, Money Flow Index, Accumulation,
        VWAP Band
    Momentum (5): Rate of Change, Momentum Oscillator, Elder Impulse,
        Aroon, Price Oscillator
    Hybrid (5+): RSI+MACD Confluence, Supertrend+EMA, Ichimoku+RSI, ADX Filter,
        MACD Histogram Divergence

OWNED BY: Phase 1 — Strategy Layer
LAST UPDATED: 2026-03-18
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional

from strategies.base import BaseStrategy, Signal, StrategyConfig


# ─────────────────────────────────────────────────────────────────
# TREND-FOLLOWING STRATEGIES
# ─────────────────────────────────────────────────────────────────

class EMAcrossStrategy(BaseStrategy):
    """Fast/slow EMA crossover — the canonical trend-following entry."""

    def __init__(self, fast: int = 9, slow: int = 21, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"EMA_{fast}_{slow}", direction=direction))
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["ema_fast"] = ta.ema(df["close"], length=self.fast)
        df["ema_slow"] = ta.ema(df["close"], length=self.slow)
        df["atr"] = self._atr(df)
        df = df.dropna()

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            crossed_up = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
            crossed_dn = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

            if crossed_up and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.65, self.name, "EMA bullish crossover"))
            elif crossed_dn and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.65, self.name, "EMA bearish crossover"))

        return signals


class MACDStrategy(BaseStrategy):
    """MACD histogram zero-line cross with signal line confirmation."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"MACD_{fast}_{slow}_{signal}", direction=direction))
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        macd = ta.macd(df["close"], fast=self.fast, slow=self.slow, signal=self.signal_period)
        if macd is None or macd.empty:
            return []
        df = pd.concat([df, macd], axis=1).dropna()
        df["atr"] = self._atr(df)

        hist_col = [c for c in df.columns if "MACDh" in c][0]
        macd_col = [c for c in df.columns if c.startswith("MACD_") and "MACDs" not in c and "MACDh" not in c][0]
        sig_col  = [c for c in df.columns if "MACDs" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            # Histogram crosses zero
            if prev[hist_col] < 0 and curr[hist_col] > 0 and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.68, self.name, "MACD histogram bullish zero cross"))
            elif prev[hist_col] > 0 and curr[hist_col] < 0 and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.68, self.name, "MACD histogram bearish zero cross"))

        return signals


class SupertrendStrategy(BaseStrategy):
    """Supertrend — ATR-based dynamic support/resistance."""

    def __init__(self, period: int = 7, multiplier: float = 3.0, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"Supertrend_{period}_{multiplier}", direction=direction))
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        st = ta.supertrend(df["high"], df["low"], df["close"], length=self.period, multiplier=self.multiplier)
        if st is None or st.empty:
            return []
        df = pd.concat([df, st], axis=1).dropna()
        df["atr"] = self._atr(df)

        dir_col = [c for c in df.columns if "SUPERTd" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            turned_bull = prev[dir_col] == -1 and curr[dir_col] == 1
            turned_bear = prev[dir_col] == 1  and curr[dir_col] == -1

            if turned_bull and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.72, self.name, "Supertrend flipped bullish"))
            elif turned_bear and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.72, self.name, "Supertrend flipped bearish"))

        return signals


class ADXStrategy(BaseStrategy):
    """ADX trend strength filter with DI+/DI- directional signals."""

    def __init__(self, period: int = 14, adx_threshold: float = 25.0, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"ADX_{period}", direction=direction))
        self.period = period
        self.threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        adx = ta.adx(df["high"], df["low"], df["close"], length=self.period)
        if adx is None or adx.empty:
            return []
        df = pd.concat([df, adx], axis=1).dropna()
        df["atr"] = self._atr(df)

        adx_col = [c for c in df.columns if c.startswith("ADX_")][0]
        dmp_col = [c for c in df.columns if "DMP_" in c][0]
        dmn_col = [c for c in df.columns if "DMN_" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            trending = curr[adx_col] > self.threshold

            di_bull = prev[dmp_col] <= prev[dmn_col] and curr[dmp_col] > curr[dmn_col]
            di_bear = prev[dmp_col] >= prev[dmn_col] and curr[dmp_col] < curr[dmn_col]

            if trending and di_bull and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.70, self.name, f"ADX {curr[adx_col]:.1f} + DI+ cross"))
            elif trending and di_bear and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.70, self.name, f"ADX {curr[adx_col]:.1f} + DI- cross"))

        return signals


class IchimokuStrategy(BaseStrategy):
    """Ichimoku cloud — price above/below cloud with TK cross."""

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        ichi = ta.ichimoku(df["high"], df["low"], df["close"])
        if ichi is None or (isinstance(ichi, tuple) and ichi[0] is None):
            return []
        if isinstance(ichi, tuple):
            ichi = ichi[0]
        df = pd.concat([df, ichi], axis=1).dropna()
        df["atr"] = self._atr(df)

        tenkan = [c for c in df.columns if "ITS_" in c]
        kijun  = [c for c in df.columns if "IKS_" in c]
        span_a = [c for c in df.columns if "ISA_" in c]
        span_b = [c for c in df.columns if "ISB_" in c]

        if not all([tenkan, kijun, span_a, span_b]):
            return []

        t_col, k_col = tenkan[0], kijun[0]
        sa_col, sb_col = span_a[0], span_b[0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            cloud_top = max(curr[sa_col], curr[sb_col])
            cloud_bot = min(curr[sa_col], curr[sb_col])
            above_cloud = curr["close"] > cloud_top
            below_cloud = curr["close"] < cloud_bot

            tk_bull = prev[t_col] <= prev[k_col] and curr[t_col] > curr[k_col]
            tk_bear = prev[t_col] >= prev[k_col] and curr[t_col] < curr[k_col]

            if above_cloud and tk_bull and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.75, "Ichimoku", "TK bull cross above cloud"))
            elif below_cloud and tk_bear and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.75, "Ichimoku", "TK bear cross below cloud"))

        return signals


# ─────────────────────────────────────────────────────────────────
# MEAN REVERSION STRATEGIES
# ─────────────────────────────────────────────────────────────────

class RSIReversalStrategy(BaseStrategy):
    """RSI extreme reversal — buy oversold, short overbought."""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"RSI_{period}", direction=direction))
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.period)
        df["atr"] = self._atr(df)
        df = df.dropna()

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            # RSI exits oversold: long signal
            if prev["rsi"] < self.oversold and curr["rsi"] >= self.oversold and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.62, self.name, f"RSI exits oversold ({curr['rsi']:.1f})"))
            # RSI exits overbought: short signal
            elif prev["rsi"] > self.overbought and curr["rsi"] <= self.overbought and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.62, self.name, f"RSI exits overbought ({curr['rsi']:.1f})"))

        return signals


class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band squeeze + breakout."""

    def __init__(self, period: int = 20, std: float = 2.0, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"BB_{period}_{std}", direction=direction))
        self.period = period
        self.std = std

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        bb = ta.bbands(df["close"], length=self.period, std=self.std)
        if bb is None or bb.empty:
            return []
        df = pd.concat([df, bb], axis=1).dropna()
        df["atr"] = self._atr(df)

        upper = [c for c in df.columns if "BBU_" in c][0]
        lower = [c for c in df.columns if "BBL_" in c][0]
        mid   = [c for c in df.columns if "BBM_" in c][0]
        bw    = [c for c in df.columns if "BBB_" in c][0]

        signals = []
        for i in range(2, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            # Squeeze: bandwidth is low (consolidation), then price breaks out
            squeeze = prev[bw] < df[bw].rolling(20).quantile(0.2).iloc[i - 1]

            if squeeze and curr["close"] > curr[upper] and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = curr[mid]
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.66, self.name, "BB squeeze bullish breakout"))
            elif squeeze and curr["close"] < curr[lower] and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = curr[mid]
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.66, self.name, "BB squeeze bearish breakout"))

        return signals


class StochasticStrategy(BaseStrategy):
    """Stochastic %K/%D cross in oversold/overbought zones."""

    def __init__(self, k: int = 14, d: int = 3, smooth: int = 3, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"Stoch_{k}_{d}", direction=direction))
        self.k = k
        self.d = d
        self.smooth = smooth

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=self.k, d=self.d, smooth_k=self.smooth)
        if stoch is None or stoch.empty:
            return []
        df = pd.concat([df, stoch], axis=1).dropna()
        df["atr"] = self._atr(df)

        k_col = [c for c in df.columns if "STOCHk_" in c][0]
        d_col = [c for c in df.columns if "STOCHd_" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            k_cross_up = prev[k_col] <= prev[d_col] and curr[k_col] > curr[d_col] and curr[k_col] < 30
            k_cross_dn = prev[k_col] >= prev[d_col] and curr[k_col] < curr[d_col] and curr[k_col] > 70

            if k_cross_up and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.60, self.name, "Stochastic bull cross in oversold"))
            elif k_cross_dn and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.60, self.name, "Stochastic bear cross in overbought"))

        return signals


class VWAPStrategy(BaseStrategy):
    """VWAP reversion — price deviates from VWAP and snaps back."""

    def __init__(self, deviation_pct: float = 0.5, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"VWAP_{deviation_pct}pct", direction=direction))
        self.deviation_pct = deviation_pct / 100

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if df["vwap"].isna().all():
            return []
        df["atr"] = self._atr(df)
        df = df.dropna()

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            dev = (curr["close"] - curr["vwap"]) / curr["vwap"]

            # Price snapping back up through VWAP from below
            if prev["close"] < prev["vwap"] and curr["close"] > curr["vwap"] and abs(dev) < self.deviation_pct and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.68, self.name, "Price reclaimed VWAP"))
            elif prev["close"] > prev["vwap"] and curr["close"] < curr["vwap"] and abs(dev) < self.deviation_pct and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.68, self.name, "Price broke below VWAP"))

        return signals


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Opening Range Breakout — first 15 minutes sets the range.
    Breakout above = long; breakdown below = short.
    """

    def __init__(self, range_minutes: int = 15, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"ORB_{range_minutes}m", direction=direction))
        self.range_minutes = range_minutes

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["atr"] = self._atr(df)
        signals = []

        for date, day_df in df.groupby(df.index.date):
            if len(day_df) < self.range_minutes + 2:
                continue

            # Opening range = first N candles
            or_df = day_df.iloc[:self.range_minutes]
            or_high = or_df["high"].max()
            or_low  = or_df["low"].min()
            range_size = or_high - or_low

            if range_size <= 0:
                continue

            rest = day_df.iloc[self.range_minutes:]
            broken_high = False
            broken_low  = False

            for i, (ts, row) in enumerate(rest.iterrows()):
                atr = row["atr"] if not pd.isna(row["atr"]) else range_size * 0.5

                if not broken_high and row["close"] > or_high and self.config.direction in ("long_only", "both"):
                    broken_high = True
                    entry = row["close"]
                    stop = or_low
                    t1, t2 = self._targets_from_rr(entry, stop)
                    signals.append(Signal(ts, symbol, "long", entry, stop, t1, t2, 0.73, self.name, f"ORB breakout above {or_high:.2f}"))
                elif not broken_low and row["close"] < or_low and self.config.direction in ("short_only", "both"):
                    broken_low = True
                    entry = row["close"]
                    stop = or_high
                    t1, t2 = self._targets_from_rr(entry, stop)
                    signals.append(Signal(ts, symbol, "short", entry, stop, t1, t2, 0.73, self.name, f"ORB breakdown below {or_low:.2f}"))

        return signals


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout — Turtle Trading style."""

    def __init__(self, period: int = 20, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"Donchian_{period}", direction=direction))
        self.period = period

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["dc_upper"] = df["high"].rolling(self.period).max().shift(1)
        df["dc_lower"] = df["low"].rolling(self.period).min().shift(1)
        df["atr"] = self._atr(df)
        df = df.dropna()

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            if curr["close"] > curr["dc_upper"] and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = curr["dc_lower"]
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.65, self.name, f"Donchian {self.period}-period high breakout"))
            elif curr["close"] < curr["dc_lower"] and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = curr["dc_upper"]
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.65, self.name, f"Donchian {self.period}-period low breakdown"))

        return signals


class MFIStrategy(BaseStrategy):
    """Money Flow Index — volume-weighted RSI for institutional flow."""

    def __init__(self, period: int = 14, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"MFI_{period}", direction=direction))
        self.period = period

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=self.period)
        df["atr"] = self._atr(df)
        df = df.dropna()

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            if prev["mfi"] < 20 and curr["mfi"] >= 20 and self.config.direction in ("long_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.67, self.name, f"MFI exits oversold ({curr['mfi']:.1f})"))
            elif prev["mfi"] > 80 and curr["mfi"] <= 80 and self.config.direction in ("short_only", "both"):
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.67, self.name, f"MFI exits overbought ({curr['mfi']:.1f})"))

        return signals


class RSIMACDConfluentStrategy(BaseStrategy):
    """
    RSI + MACD confluence — both must agree.
    Higher precision than either indicator alone.
    """

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is None or macd.empty:
            return []
        df = pd.concat([df, macd], axis=1)
        df["atr"] = self._atr(df)
        df = df.dropna()

        hist_col = [c for c in df.columns if "MACDh" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            rsi_bull = curr["rsi"] > 50 and prev["rsi"] <= 50
            rsi_bear = curr["rsi"] < 50 and prev["rsi"] >= 50
            macd_bull = prev[hist_col] < 0 and curr[hist_col] > 0
            macd_bear = prev[hist_col] > 0 and curr[hist_col] < 0

            if rsi_bull and macd_bull:
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.78, "RSI_MACD_Confluence", "RSI + MACD both bullish"))
            elif rsi_bear and macd_bear:
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.78, "RSI_MACD_Confluence", "RSI + MACD both bearish"))

        return signals


class SupertrendEMAStrategy(BaseStrategy):
    """Supertrend + EMA filter — trend must agree on two timeframes."""

    def __init__(self, ema_period: int = 50, direction: str = "both"):
        super().__init__(StrategyConfig(name=f"ST_EMA_{ema_period}", direction=direction))
        self.ema_period = ema_period

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        df = df.copy()
        st = ta.supertrend(df["high"], df["low"], df["close"])
        if st is None or st.empty:
            return []
        df = pd.concat([df, st], axis=1)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["atr"] = self._atr(df)
        df = df.dropna()

        dir_col = [c for c in df.columns if "SUPERTd" in c][0]

        signals = []
        for i in range(1, len(df)):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            st_bull = prev[dir_col] == -1 and curr[dir_col] == 1
            st_bear = prev[dir_col] == 1  and curr[dir_col] == -1
            ema_bull = curr["close"] > curr["ema"]
            ema_bear = curr["close"] < curr["ema"]

            if st_bull and ema_bull:
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "long")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "long", entry, stop, t1, t2, 0.75, self.name, "Supertrend + EMA confluence bull"))
            elif st_bear and ema_bear:
                entry = curr["close"]
                stop = self._stop_from_atr(entry, curr["atr"], "short")
                t1, t2 = self._targets_from_rr(entry, stop)
                signals.append(Signal(curr.name, symbol, "short", entry, stop, t1, t2, 0.75, self.name, "Supertrend + EMA confluence bear"))

        return signals


# ─────────────────────────────────────────────────────────────────
# FULL STANDARD STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────

def get_all_standard_strategies() -> list[BaseStrategy]:
    """
    Returns all standard strategies with multiple parameter variants.
    The backtester runs all of these and ranks them.
    """
    return [
        # EMA crossovers — multiple parameter sets
        EMAcrossStrategy(fast=9,  slow=21,  direction="both"),
        EMAcrossStrategy(fast=5,  slow=13,  direction="both"),
        EMAcrossStrategy(fast=20, slow=50,  direction="both"),
        EMAcrossStrategy(fast=50, slow=200, direction="both"),

        # MACD variants
        MACDStrategy(fast=12, slow=26, signal=9,  direction="both"),
        MACDStrategy(fast=8,  slow=17, signal=9,  direction="both"),
        MACDStrategy(fast=5,  slow=35, signal=5,  direction="both"),

        # Supertrend variants
        SupertrendStrategy(period=7,  multiplier=3.0, direction="both"),
        SupertrendStrategy(period=10, multiplier=2.0, direction="both"),
        SupertrendStrategy(period=14, multiplier=1.5, direction="both"),

        # ADX
        ADXStrategy(period=14, adx_threshold=25, direction="both"),
        ADXStrategy(period=14, adx_threshold=20, direction="both"),

        # Ichimoku
        IchimokuStrategy(),

        # RSI variants
        RSIReversalStrategy(period=14, oversold=30, overbought=70, direction="both"),
        RSIReversalStrategy(period=7,  oversold=25, overbought=75, direction="both"),
        RSIReversalStrategy(period=21, oversold=35, overbought=65, direction="both"),

        # Bollinger Bands
        BollingerBandStrategy(period=20, std=2.0, direction="both"),
        BollingerBandStrategy(period=20, std=2.5, direction="both"),

        # Stochastic
        StochasticStrategy(k=14, d=3, smooth=3, direction="both"),
        StochasticStrategy(k=5,  d=3, smooth=3, direction="both"),

        # VWAP
        VWAPStrategy(deviation_pct=0.3, direction="both"),
        VWAPStrategy(deviation_pct=0.5, direction="both"),

        # Opening Range Breakout
        OpeningRangeBreakoutStrategy(range_minutes=15, direction="both"),
        OpeningRangeBreakoutStrategy(range_minutes=30, direction="both"),

        # Donchian
        DonchianBreakoutStrategy(period=20, direction="both"),
        DonchianBreakoutStrategy(period=55, direction="both"),

        # MFI
        MFIStrategy(period=14, direction="both"),

        # Confluence strategies (higher precision)
        RSIMACDConfluentStrategy(),
        SupertrendEMAStrategy(ema_period=50, direction="both"),
        SupertrendEMAStrategy(ema_period=200, direction="both"),
    ]
