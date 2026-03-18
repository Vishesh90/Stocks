"""
backtesting/leaderboard.py — Strategy ranking and reporting

INTENT:
    Takes backtest results and produces a ranked leaderboard with
    composite scoring, per-segment analysis, and a human-readable report.

OWNED BY: Phase 1 — Backtesting Engine
LAST UPDATED: 2026-03-18
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from rich.table import Table
from rich.console import Console
from rich import box

from backtesting.engine import BacktestResult


@dataclass
class LeaderboardEntry:
    rank:            int
    strategy_name:   str
    best_symbol:     str
    composite_score: float
    avg_sharpe:      float
    avg_win_rate:    float
    avg_profit_factor: float
    avg_net_pnl:     float
    avg_max_dd:      float
    total_trades:    int
    instruments_tested: int
    verdict:         str  # STRONG | GOOD | MARGINAL | AVOID


def composite_score(result: BacktestResult) -> float:
    """
    Single number to rank strategies. Weights:
    - Sharpe ratio:    40% (risk-adjusted return)
    - Win rate:        20% (psychological sustainability)
    - Profit factor:   25% (gross win / gross loss)
    - Drawdown penalty:15% (negative contribution from max drawdown)

    WHY: No single metric captures everything. A strategy with Sharpe=3
    but 90% drawdown is unusable. This composite balances all dimensions.
    """
    sharpe_score = min(result.sharpe_ratio * 20, 100)   # Cap at 100
    winrate_score = result.win_rate * 100
    pf_score = min((result.profit_factor - 1) * 25, 100) if result.profit_factor > 1 else 0
    dd_penalty = max(-50, result.max_drawdown / 100)     # Drawdown is negative INR

    return 0.40 * sharpe_score + 0.20 * winrate_score + 0.25 * pf_score + 0.15 * dd_penalty


def build_leaderboard(results: list[BacktestResult]) -> list[LeaderboardEntry]:
    """Aggregates per-strategy results and ranks by composite score."""
    from collections import defaultdict

    strategy_results: dict[str, list[BacktestResult]] = defaultdict(list)
    for r in results:
        if r.total_trades >= 10:
            strategy_results[r.strategy_name].append(r)

    entries = []
    for strategy_name, strat_results in strategy_results.items():
        scores = [composite_score(r) for r in strat_results]
        best   = max(strat_results, key=lambda r: composite_score(r))

        avg_sharpe = np.mean([r.sharpe_ratio for r in strat_results])
        avg_wr     = np.mean([r.win_rate for r in strat_results])
        avg_pf     = np.mean([r.profit_factor for r in strat_results if r.profit_factor < 1000])
        avg_pnl    = np.mean([r.net_pnl for r in strat_results])
        avg_dd     = np.mean([r.max_drawdown for r in strat_results])
        composite  = np.mean(scores)

        if composite >= 60:
            verdict = "STRONG"
        elif composite >= 40:
            verdict = "GOOD"
        elif composite >= 20:
            verdict = "MARGINAL"
        else:
            verdict = "AVOID"

        entries.append(LeaderboardEntry(
            rank=0,
            strategy_name=strategy_name,
            best_symbol=best.symbol,
            composite_score=composite,
            avg_sharpe=avg_sharpe,
            avg_win_rate=avg_wr,
            avg_profit_factor=avg_pf,
            avg_net_pnl=avg_pnl,
            avg_max_dd=avg_dd,
            total_trades=sum(r.total_trades for r in strat_results),
            instruments_tested=len(strat_results),
            verdict=verdict,
        ))

    entries.sort(key=lambda e: e.composite_score, reverse=True)
    for i, e in enumerate(entries):
        e.rank = i + 1

    return entries


def print_leaderboard(entries: list[LeaderboardEntry], top_n: int = 20) -> None:
    """Prints a rich-formatted leaderboard to console."""
    console = Console()
    table = Table(title=f"Strategy Leaderboard — Top {min(top_n, len(entries))}", box=box.ROUNDED)

    table.add_column("Rank",     style="bold white", width=5)
    table.add_column("Strategy", style="bold cyan",  width=30)
    table.add_column("Score",    style="bold yellow", width=8)
    table.add_column("Sharpe",   width=8)
    table.add_column("Win %",    width=8)
    table.add_column("PF",       width=7)
    table.add_column("Net PnL",  style="green",       width=12)
    table.add_column("Max DD",   style="red",          width=12)
    table.add_column("Trades",   width=8)
    table.add_column("Verdict",  width=10)

    for e in entries[:top_n]:
        verdict_color = {"STRONG": "green", "GOOD": "yellow", "MARGINAL": "orange3", "AVOID": "red"}
        v_style = verdict_color.get(e.verdict, "white")

        table.add_row(
            str(e.rank),
            e.strategy_name,
            f"{e.composite_score:.1f}",
            f"{e.avg_sharpe:.2f}",
            f"{e.avg_win_rate:.1%}",
            f"{e.avg_profit_factor:.2f}",
            f"₹{e.avg_net_pnl:,.0f}",
            f"₹{e.avg_max_dd:,.0f}",
            str(e.total_trades),
            f"[{v_style}]{e.verdict}[/{v_style}]",
        )

    console.print(table)


def save_leaderboard_csv(entries: list[LeaderboardEntry], path: str = "reports/leaderboard.csv") -> None:
    rows = []
    for e in entries:
        rows.append({
            "rank": e.rank,
            "strategy": e.strategy_name,
            "composite_score": round(e.composite_score, 2),
            "avg_sharpe": round(e.avg_sharpe, 3),
            "avg_win_rate": round(e.avg_win_rate, 4),
            "avg_profit_factor": round(e.avg_profit_factor, 3),
            "avg_net_pnl_inr": round(e.avg_net_pnl, 2),
            "avg_max_drawdown_inr": round(e.avg_max_dd, 2),
            "total_trades": e.total_trades,
            "instruments_tested": e.instruments_tested,
            "best_symbol": e.best_symbol,
            "verdict": e.verdict,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Leaderboard saved → {path}")
