"""
strategies/base.py — Abstract base class for all strategies

INTENT:
    Every strategy — standard or mathematical — inherits from BaseStrategy.
    This enforces a uniform interface so the backtesting engine can run any
    strategy without knowing its internals.

IMPACT:
    Adding a new strategy means subclassing BaseStrategy. The backtester,
    the leaderboard, and the live agent all work automatically with no
    other changes.

OWNED BY: Phase 1 — Strategy Layer
LAST UPDATED: 2026-03-18
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    """Hyperparameters for a strategy. Subclasses extend this."""
    name: str = ""
    direction: str = "long_only"  # long_only | short_only | both
    timeframe: str = "5m"


@dataclass
class Signal:
    """Single trade signal emitted by a strategy."""
    timestamp: pd.Timestamp
    symbol: str
    direction: str           # long | short
    entry_price: float
    stop_loss: float
    target_1: float          # Partial profit at 1R
    target_2: float          # Remaining position target
    confidence: float        # 0.0 – 1.0
    strategy_name: str
    reason: str = ""         # Human-readable explanation


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    WHY: A uniform interface decouples strategy logic from execution and
    backtesting. The engine asks every strategy the same question:
    "Given this OHLCV data, what signals do you produce?"
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig(name=self.__class__.__name__)

    @property
    def name(self) -> str:
        return self.config.name or self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> list[Signal]:
        """
        Given clean OHLCV DataFrame, return a list of trade signals.

        The DataFrame always has columns: open, high, low, close, volume.
        Index is datetime (Asia/Kolkata timezone).
        """
        ...

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — used for stop loss placement across strategies."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _stop_from_atr(self, entry: float, atr: float, direction: str, multiplier: float = 1.5) -> float:
        """ATR-based stop loss. Never guesses — always sized to volatility."""
        if direction == "long":
            return entry - (atr * multiplier)
        return entry + (atr * multiplier)

    def _targets_from_rr(self, entry: float, stop: float, rr1: float = 1.5, rr2: float = 3.0) -> tuple[float, float]:
        """R-multiple targets. Book 50% at 1R, let rest run to 2R."""
        risk = abs(entry - stop)
        if entry > stop:  # long
            return entry + risk * rr1, entry + risk * rr2
        return entry - risk * rr1, entry - risk * rr2
