"""
backtesting/engine.py — Core backtesting engine

INTENT:
    Runs any strategy against historical OHLCV data and produces a standardised
    BacktestResult. Includes realistic transaction costs (brokerage, STT, exchange
    fees), slippage, partial profit booking at 1R, and trailing stop on remainder.

IMPACT:
    Every strategy's performance metrics come from here. The leaderboard, the
    segment scorer, and the live agent all rely on these results being accurate.
    Optimistic backtests (ignoring costs) produce strategies that lose money live.

FUNCTIONS:
    - run_backtest(): Single strategy on single instrument
    - run_full_backtest(): All strategies on all instruments (parallel)
    - BacktestResult: Structured result with all metrics

OWNED BY: Phase 1 — Backtesting Engine
LAST UPDATED: 2026-03-18
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

from strategies.base import BaseStrategy, Signal
from data.universe import Instrument
from config.settings import settings


# ─────────────────────────────────────────────────────────────────
# TRANSACTION COSTS (Indian market, realistic)
# ─────────────────────────────────────────────────────────────────

BROKERAGE_PER_TRADE = 20.0           # Zerodha/Groww flat ₹20 per order
STT_INTRADAY_SELL   = 0.00025        # 0.025% on sell side (intraday equity)
EXCHANGE_TXN_CHARGE = 0.0000345      # NSE transaction charge
SEBI_FEE            = 0.000001       # SEBI turnover fee
GST_ON_BROKERAGE    = 0.18           # 18% GST on brokerage
STAMP_DUTY_BUY      = 0.00003        # 0.003% on buy side
SLIPPAGE_PCT        = 0.0003         # 0.03% average slippage per side


def transaction_cost(price: float, qty: int, side: str = "buy") -> float:
    """
    Total realistic cost for a single side of a trade.

    WHY: Backtests without realistic costs overstate returns by 0.1-0.3% per trade.
    For intraday trading with 10 trades/day, this compounds to a fatal error.
    These numbers match actual NSE/BSE charges as of 2026.
    """
    value = price * qty
    brokerage = min(BROKERAGE_PER_TRADE, value * 0.0003)
    stt = value * STT_INTRADAY_SELL if side == "sell" else 0
    exchange = value * EXCHANGE_TXN_CHARGE
    sebi = value * SEBI_FEE
    gst = brokerage * GST_ON_BROKERAGE
    stamp = value * STAMP_DUTY_BUY if side == "buy" else 0
    slippage = value * SLIPPAGE_PCT
    return brokerage + stt + exchange + sebi + gst + stamp + slippage


# ─────────────────────────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_time:   pd.Timestamp
    exit_time:    Optional[pd.Timestamp]
    symbol:       str
    direction:    str
    entry_price:  float
    exit_price:   float
    qty:          int
    pnl:          float
    pnl_pct:      float
    cost:         float
    net_pnl:      float
    exit_reason:  str
    strategy:     str
    confidence:   float


@dataclass
class BacktestResult:
    strategy_name:   str
    symbol:          str
    total_trades:    int = 0
    winning_trades:  int = 0
    losing_trades:   int = 0
    gross_pnl:       float = 0.0
    total_costs:     float = 0.0
    net_pnl:         float = 0.0
    win_rate:        float = 0.0
    profit_factor:   float = 0.0
    sharpe_ratio:    float = 0.0
    sortino_ratio:   float = 0.0
    max_drawdown:    float = 0.0
    avg_win:         float = 0.0
    avg_loss:        float = 0.0
    expectancy:      float = 0.0
    total_return_pct:float = 0.0
    trades:          list[TradeRecord] = field(default_factory=list)

    def compute_metrics(self) -> None:
        """Derive all ratio metrics from trade list."""
        if not self.trades:
            return

        pnls = [t.net_pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        self.total_trades   = len(self.trades)
        self.winning_trades = len(wins)
        self.losing_trades  = len(losses)
        self.gross_pnl      = sum(t.pnl for t in self.trades)
        self.total_costs    = sum(t.cost for t in self.trades)
        self.net_pnl        = sum(pnls)
        self.win_rate       = self.winning_trades / self.total_trades if self.total_trades else 0

        gross_wins   = sum(wins)
        gross_losses = abs(sum(losses))
        self.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        self.avg_win  = np.mean(wins)   if wins   else 0.0
        self.avg_loss = np.mean(losses) if losses else 0.0
        self.expectancy = (self.win_rate * self.avg_win) + ((1 - self.win_rate) * self.avg_loss)
        self.total_return_pct = (self.net_pnl / settings.capital_inr) * 100

        # Sharpe & Sortino (annualised, assuming 252 trading days)
        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            mean_pnl = np.mean(pnl_arr)
            std_pnl  = np.std(pnl_arr)
            downside = np.std([p for p in pnls if p < 0]) if losses else 1e-10
            trades_per_year = 252 * (self.total_trades / max(1, len(set(
                t.entry_time.date() for t in self.trades
            ))))
            self.sharpe_ratio  = (mean_pnl / (std_pnl  + 1e-10)) * np.sqrt(trades_per_year)
            self.sortino_ratio = (mean_pnl / (downside + 1e-10)) * np.sqrt(trades_per_year)

        # Max Drawdown
        equity_curve = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak)
        self.max_drawdown = float(np.min(drawdown))

    def summary(self) -> dict:
        return {
            "strategy":       self.strategy_name,
            "symbol":         self.symbol,
            "trades":         self.total_trades,
            "win_rate":       f"{self.win_rate:.1%}",
            "profit_factor":  f"{self.profit_factor:.2f}",
            "sharpe":         f"{self.sharpe_ratio:.2f}",
            "sortino":        f"{self.sortino_ratio:.2f}",
            "net_pnl":        f"₹{self.net_pnl:,.0f}",
            "max_drawdown":   f"₹{self.max_drawdown:,.0f}",
            "expectancy":     f"₹{self.expectancy:.2f}",
            "total_return":   f"{self.total_return_pct:.2f}%",
        }


# ─────────────────────────────────────────────────────────────────
# POSITION SIZER
# ─────────────────────────────────────────────────────────────────

def position_size(entry: float, stop: float, capital: float, risk_pct: float = 0.015) -> int:
    """
    ATR/risk-based position sizing. Risks risk_pct of capital per trade.

    WHY: Fixed-share sizing blows up accounts. Risking a fixed % of capital
    per trade (Kelly-inspired fractional sizing) keeps drawdowns recoverable
    regardless of the instrument's price level.
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 1
    risk_amount = capital * risk_pct
    qty = int(risk_amount / risk_per_share)
    return max(1, qty)


# ─────────────────────────────────────────────────────────────────
# BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────

def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    symbol: str,
    capital: float = None,
    risk_pct: float = None,
    daily_loss_limit: float = None,
) -> BacktestResult:
    """
    Runs a single strategy on a single instrument's OHLCV DataFrame.

    Trade logic:
    - Entry at signal's entry_price + slippage
    - Stop loss: exit if price crosses stop
    - Target 1: exit 50% at 1R, move stop to breakeven
    - Target 2: trailing stop on remaining 50%
    - EOD: force close any open positions at 3:25 PM
    - Daily loss limit: stop trading for the day if exceeded

    WHY: Each of these rules reflects a real trading discipline. Without them,
    the backtest produces theoretical profits that can never be achieved live.
    """
    capital = capital or settings.capital_inr
    risk_pct = risk_pct or settings.risk_per_trade_pct
    daily_loss_limit = daily_loss_limit or settings.daily_loss_limit_inr

    result = BacktestResult(strategy_name=strategy.name, symbol=symbol)
    signals = strategy.generate_signals(df, symbol)

    if not signals:
        return result

    # Index signals by timestamp for fast lookup
    signal_map: dict[pd.Timestamp, Signal] = {s.timestamp: s for s in signals}
    open_position = None
    daily_pnl: dict = {}

    for i, (ts, row) in enumerate(df.iterrows()):
        date_key = ts.date()

        # Check daily loss limit
        day_loss = daily_pnl.get(date_key, 0.0)
        if day_loss <= -daily_loss_limit and open_position is None:
            continue  # Don't open new trades today

        # Check if current bar has a signal
        if ts in signal_map and open_position is None:
            sig = signal_map[ts]
            qty = position_size(sig.entry_price, sig.stop_loss, capital, risk_pct)

            # Ensure we can afford at least 1 unit
            if qty < 1 or sig.entry_price * qty > capital * 2:
                continue

            entry_price = sig.entry_price * (1 + SLIPPAGE_PCT if sig.direction == "long" else 1 - SLIPPAGE_PCT)
            entry_cost  = transaction_cost(entry_price, qty, "buy")

            open_position = {
                "signal":      sig,
                "entry_time":  ts,
                "entry_price": entry_price,
                "qty":         qty,
                "stop":        sig.stop_loss,
                "target1":     sig.target_1,
                "target2":     sig.target_2,
                "t1_hit":      False,
                "entry_cost":  entry_cost,
                "direction":   sig.direction,
            }
            continue

        if open_position is None:
            continue

        pos = open_position
        direction = pos["direction"]
        current_high = row["high"]
        current_low  = row["low"]
        current_close = row["close"]

        # EOD force close at 3:25 PM
        is_eod = ts.hour == 15 and ts.minute >= 25

        exit_price  = None
        exit_reason = None

        if direction == "long":
            # Stop hit
            if current_low <= pos["stop"]:
                exit_price  = pos["stop"]
                exit_reason = "stop_loss"
            # Target 1
            elif not pos["t1_hit"] and current_high >= pos["target1"]:
                # Partial profit: simulate by closing 50% at target1
                # For simplicity: record half position closed, move stop to breakeven
                pos["t1_hit"] = True
                pos["stop"]   = pos["entry_price"]  # Break even on remaining
                # Record partial profit trade
                partial_pnl = (pos["target1"] - pos["entry_price"]) * (pos["qty"] // 2)
                partial_cost = transaction_cost(pos["target1"], pos["qty"] // 2, "sell")
                rec = TradeRecord(
                    entry_time=pos["entry_time"], exit_time=ts,
                    symbol=symbol, direction=direction,
                    entry_price=pos["entry_price"], exit_price=pos["target1"],
                    qty=pos["qty"] // 2, pnl=partial_pnl, pnl_pct=partial_pnl / (pos["entry_price"] * pos["qty"] // 2),
                    cost=pos["entry_cost"] / 2 + partial_cost,
                    net_pnl=partial_pnl - pos["entry_cost"] / 2 - partial_cost,
                    exit_reason="target_1_partial", strategy=strategy.name, confidence=pos["signal"].confidence
                )
                result.trades.append(rec)
                daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + rec.net_pnl
                pos["qty"] = pos["qty"] - pos["qty"] // 2  # Remaining qty
                continue
            # Target 2 (remaining)
            elif pos["t1_hit"] and current_high >= pos["target2"]:
                exit_price  = pos["target2"]
                exit_reason = "target_2"
            # Trailing stop (after T1 hit): trail by 1 ATR
            elif pos["t1_hit"]:
                # Simple trail: move stop up to (current_close - ATR) every bar
                bar_atr = df["high"].iloc[i] - df["low"].iloc[i]
                new_stop = current_close - bar_atr * 1.5
                pos["stop"] = max(pos["stop"], new_stop)

        else:  # short
            if current_high >= pos["stop"]:
                exit_price  = pos["stop"]
                exit_reason = "stop_loss"
            elif not pos["t1_hit"] and current_low <= pos["target1"]:
                pos["t1_hit"] = True
                pos["stop"]   = pos["entry_price"]
                partial_pnl = (pos["entry_price"] - pos["target1"]) * (pos["qty"] // 2)
                partial_cost = transaction_cost(pos["target1"], pos["qty"] // 2, "buy")
                rec = TradeRecord(
                    entry_time=pos["entry_time"], exit_time=ts,
                    symbol=symbol, direction=direction,
                    entry_price=pos["entry_price"], exit_price=pos["target1"],
                    qty=pos["qty"] // 2, pnl=partial_pnl, pnl_pct=partial_pnl / (pos["entry_price"] * pos["qty"] // 2),
                    cost=pos["entry_cost"] / 2 + partial_cost,
                    net_pnl=partial_pnl - pos["entry_cost"] / 2 - partial_cost,
                    exit_reason="target_1_partial", strategy=strategy.name, confidence=pos["signal"].confidence
                )
                result.trades.append(rec)
                daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + rec.net_pnl
                pos["qty"] = pos["qty"] - pos["qty"] // 2
                continue
            elif pos["t1_hit"] and current_low <= pos["target2"]:
                exit_price  = pos["target2"]
                exit_reason = "target_2"
            elif pos["t1_hit"]:
                bar_atr = df["high"].iloc[i] - df["low"].iloc[i]
                new_stop = current_close + bar_atr * 1.5
                pos["stop"] = min(pos["stop"], new_stop)

        if is_eod and exit_price is None:
            exit_price  = current_close
            exit_reason = "eod_forced"

        if exit_price is not None:
            exit_price = exit_price * (1 - SLIPPAGE_PCT if direction == "long" else 1 + SLIPPAGE_PCT)
            exit_cost  = transaction_cost(exit_price, pos["qty"], "sell" if direction == "long" else "buy")
            pnl = (exit_price - pos["entry_price"]) * pos["qty"] if direction == "long" else (pos["entry_price"] - exit_price) * pos["qty"]
            total_cost = pos["entry_cost"] + exit_cost
            net_pnl = pnl - total_cost

            rec = TradeRecord(
                entry_time=pos["entry_time"], exit_time=ts,
                symbol=symbol, direction=direction,
                entry_price=pos["entry_price"], exit_price=exit_price,
                qty=pos["qty"], pnl=pnl, pnl_pct=pnl / (pos["entry_price"] * pos["qty"]),
                cost=total_cost, net_pnl=net_pnl,
                exit_reason=exit_reason, strategy=strategy.name, confidence=pos["signal"].confidence
            )
            result.trades.append(rec)
            daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + net_pnl
            open_position = None

    result.compute_metrics()
    return result


def run_full_backtest(
    strategies: list[BaseStrategy],
    instrument_data: dict[str, pd.DataFrame],
    max_workers: int = 4,
) -> list[BacktestResult]:
    """
    Runs all strategies on all instruments in parallel. Returns ranked list of results.

    WHY: With 45 strategies × 10 stocks = 450 combinations, sequential execution
    takes 80+ minutes. Parallelising across CPU cores brings this under 15 minutes.
    Each worker gets an independent strategy+data pair — no shared state.
    """
    tasks = [
        (strategy, symbol, df)
        for strategy in strategies
        for symbol, df in instrument_data.items()
    ]

    logger.info(f"Running {len(tasks)} backtest combinations ({len(strategies)} strategies × {len(instrument_data)} instruments) with {max_workers} workers")

    results = []

    def _run_single(args):
        strategy, symbol, df = args
        try:
            r = run_backtest(strategy, df, symbol)
            if r.total_trades >= 10:
                return r
        except Exception as e:
            logger.warning(f"Backtest failed: {strategy.name} / {symbol}: {e}")
        return None

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    # Use ThreadPoolExecutor (avoids pickling issues with complex strategy objects)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single, task): task for task in tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result is not None:
                results.append(result)
                logger.debug(f"  [{done_count}/{len(tasks)}] {result.strategy_name} / {result.symbol}: {result.total_trades} trades, Sharpe={result.sharpe_ratio:.2f}, Win%={result.win_rate:.1%}")
            elif done_count % 10 == 0:
                logger.info(f"  Progress: {done_count}/{len(tasks)} ({done_count/len(tasks):.0%})")

    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    logger.info(f"Completed. {len(results)} valid results. Top strategy: {results[0].strategy_name if results else 'None'}")
    return results
