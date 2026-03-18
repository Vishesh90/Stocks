"""
agent/paper_agent.py — Paper trading agent (no real money)

INTENT:
    The paper agent runs the full pipeline every day:
    1. Market context check (VIX)
    2. Segment scoring
    3. Candidate identification
    4. Signal generation from top strategies
    5. Position management (entry, partial profit, trailing stop, EOD close)
    6. Daily report

    Every action is logged but NO real orders are placed.
    Switch EXECUTION_MODE=live to connect to Groww API.

IMPACT:
    This is the validation layer. 2-4 weeks of paper trading must show
    consistent profits before live trading begins.

OWNED BY: Phase 3 — Paper Trading Agent
LAST UPDATED: 2026-03-18
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
from loguru import logger

from config.settings import settings
from data.universe import get_equity_universe, BROAD_INDICES
from data.fetcher import fetch_ohlcv
from intelligence.segment_scorer import score_segments, identify_top_candidates, market_context


@dataclass
class PaperTrade:
    id:             str
    symbol:         str
    direction:      str
    entry_price:    float
    qty:            int
    stop_loss:      float
    target_1:       float
    target_2:       float
    entry_time:     str
    strategy:       str
    reason:         str
    status:         str = "open"   # open | t1_hit | closed
    exit_price:     float = 0.0
    exit_time:      str = ""
    exit_reason:    str = ""
    pnl:            float = 0.0


class PaperAgent:
    """
    Simulates the live agent in paper mode.
    Reads live (or near-live) data, generates signals, manages positions,
    and writes a daily report.
    """

    def __init__(self):
        self.trades_file = settings.reports_dir / "paper_trades.json"
        self.reports_dir = settings.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.open_trades: list[PaperTrade] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self._load_open_trades()

    def _load_open_trades(self) -> None:
        if self.trades_file.exists():
            with open(self.trades_file) as f:
                raw = json.load(f)
            self.open_trades = [PaperTrade(**t) for t in raw.get("open_trades", [])]

    def _save_trades(self) -> None:
        with open(self.trades_file, "w") as f:
            json.dump({"open_trades": [asdict(t) for t in self.open_trades]}, f, indent=2)

    def morning_scan(self) -> None:
        """
        Run at 9:00 AM IST. Generates today's trade candidates.
        Respects VIX filter — does not trade in fear regime.
        """
        logger.info("=" * 60)
        logger.info(f"MORNING SCAN — {date.today().isoformat()}")
        logger.info("=" * 60)

        # VIX check
        vix_instr = next((i for i in BROAD_INDICES if i.symbol == "INDIA_VIX"), None)
        vix_df = fetch_ohlcv(vix_instr, interval="1d") if vix_instr else None
        ctx = market_context(vix_df)
        logger.info(f"Market context: {ctx.get('vix_note', 'VIX unavailable')}")

        if not ctx["tradeable"]:
            logger.warning("VIX too high — NO TRADES TODAY")
            self._write_daily_report(trades=[], context=ctx, reason="VIX circuit breaker")
            return

        # Segment scoring
        logger.info("Scoring market segments...")
        seg_scores = score_segments(lookback_days=20)
        top_segs = seg_scores[:3]
        logger.info(f"Top segments: {', '.join(f'{s.segment}({s.score:.0f})' for s in top_segs)}")

        # Candidate identification
        candidates = identify_top_candidates(seg_scores, top_segments=2, candidates_per_segment=3)
        if not candidates:
            logger.info("No high-quality candidates today — staying flat")
            self._write_daily_report(trades=[], context=ctx, reason="No candidates")
            return

        logger.info(f"Top candidates: {', '.join(f'{c.symbol}({c.direction})' for c in candidates[:5])}")

        # Save candidates for intraday monitoring
        candidates_file = self.reports_dir / f"candidates_{date.today().isoformat()}.json"
        with open(candidates_file, "w") as f:
            json.dump([{
                "symbol": c.symbol, "direction": c.direction,
                "entry_zone": c.entry_zone, "stop": c.stop_estimate,
                "target": c.target_estimate, "reason": c.reason, "score": c.score
            } for c in candidates], f, indent=2)

        logger.info(f"Candidates saved → {candidates_file}")

    def end_of_day_report(self) -> None:
        """
        Run at 3:30 PM IST. Force-close any open positions and write daily P&L report.
        """
        logger.info("=" * 60)
        logger.info(f"END-OF-DAY REPORT — {date.today().isoformat()}")

        closed = []
        for trade in list(self.open_trades):
            # Force close at last known price
            instr = next((i for i in get_equity_universe() if i.symbol == trade.symbol), None)
            if instr:
                df = fetch_ohlcv(instr, interval="5m", from_date=date.today().isoformat(), to_date=date.today().isoformat())
                exit_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else trade.entry_price
            else:
                exit_price = trade.entry_price

            pnl = ((exit_price - trade.entry_price) if trade.direction == "long" else (trade.entry_price - exit_price)) * trade.qty
            trade.exit_price  = exit_price
            trade.exit_time   = datetime.now().isoformat()
            trade.exit_reason = "eod_forced"
            trade.pnl         = pnl
            trade.status      = "closed"
            closed.append(trade)
            logger.info(f"  EOD close: {trade.symbol} {trade.direction} → ₹{pnl:+.0f}")

        self.open_trades = []
        self._save_trades()
        self._write_daily_report(trades=closed, context={}, reason="EOD close")

    def _write_daily_report(self, trades: list, context: dict, reason: str = "") -> None:
        total_pnl = sum(t.pnl for t in trades)
        report = {
            "date":        date.today().isoformat(),
            "reason":      reason,
            "context":     context,
            "total_pnl":   total_pnl,
            "trade_count": len(trades),
            "trades": [asdict(t) for t in trades] if trades and hasattr(trades[0], '__dataclass_fields__') else [],
        }
        report_file = self.reports_dir / f"daily_{date.today().isoformat()}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Daily P&L: ₹{total_pnl:+.0f} | Trades: {len(trades)}")
        logger.info(f"Report → {report_file}")
