#!/usr/bin/env python3
"""
scripts/run_backtest_fast.py — Fast focused backtest

Runs only the highest-signal strategies on intraday data.
Completes in ~10 minutes instead of 2 hours.
Use this for daily/iterative testing. Use run_backtest.py for full runs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console

from data.universe import NIFTY50_STOCKS
from data.fetcher import fetch_universe_data
from backtesting.engine import run_full_backtest
from backtesting.leaderboard import build_leaderboard, print_leaderboard, save_leaderboard_csv

# Import only the most promising strategies for intraday
from strategies.base import BaseStrategy
from strategies.standard import (
    EMAcrossStrategy, SupertrendStrategy, VWAPStrategy,
    OpeningRangeBreakoutStrategy, DonchianBreakoutStrategy,
    RSIMACDConfluentStrategy, SupertrendEMAStrategy, MACDStrategy,
)
from strategies.mathematical import (
    KalmanFilterMomentum, MultiTimeframeConvergence,
    VolumeWeightedMomentum, HurstExponentClassifier,
    VolatilityRegimeSwitching, PriceVelocityAcceleration,
    AutocorrelationMomentum, HalfLifeMeanReversion,
)

console = Console()

FOCUSED_STRATEGIES: list[BaseStrategy] = [
    # Best trend-following for intraday
    EMAcrossStrategy(fast=9,  slow=21,  direction="both"),
    EMAcrossStrategy(fast=20, slow=50,  direction="both"),
    SupertrendStrategy(period=7,  multiplier=3.0, direction="both"),
    SupertrendStrategy(period=10, multiplier=2.0, direction="both"),
    MACDStrategy(fast=12, slow=26, signal=9, direction="both"),
    # Best intraday-specific
    VWAPStrategy(deviation_pct=0.3, direction="both"),
    OpeningRangeBreakoutStrategy(range_minutes=15, direction="both"),
    OpeningRangeBreakoutStrategy(range_minutes=30, direction="both"),
    DonchianBreakoutStrategy(period=20, direction="both"),
    # Confluence (high precision)
    RSIMACDConfluentStrategy(),
    SupertrendEMAStrategy(ema_period=50, direction="both"),
    # Mathematical strategies (designed for 5m)
    KalmanFilterMomentum(),
    MultiTimeframeConvergence(ema_fast=9, ema_slow=21, mult=3),
    VolumeWeightedMomentum(momentum_period=10, volume_period=20, min_vol_ratio=1.5),
    HurstExponentClassifier(window=100),
    VolatilityRegimeSwitching(),
    PriceVelocityAcceleration(smooth=5),
    AutocorrelationMomentum(ac_window=30),
    HalfLifeMeanReversion(window=60, max_halflife_bars=20),
]

def main():
    console.rule("[bold cyan]Fast Intraday Backtest — 19 strategies × 10 stocks[/bold cyan]")
    console.print(f"[yellow]~{len(FOCUSED_STRATEGIES) * 10} combinations — est. 15-20 min[/yellow]\n")

    instruments = NIFTY50_STOCKS[:10]

    console.print("[cyan]Loading cached 5m data...[/cyan]")
    instrument_data = fetch_universe_data(instruments, interval="5m")
    console.print(f"[green]{len(instrument_data)} instruments loaded from cache[/green]\n")

    if not instrument_data:
        console.print("[red]No data. Run full backtest first to populate cache.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Running {len(FOCUSED_STRATEGIES) * len(instrument_data)} backtests...[/cyan]")
    results = run_full_backtest(FOCUSED_STRATEGIES, instrument_data, max_workers=6)

    entries = build_leaderboard(results)
    console.print(f"\n[bold green]DONE — {len(entries)} strategies ranked[/bold green]\n")
    print_leaderboard(entries, top_n=len(entries))

    Path("reports").mkdir(exist_ok=True)
    save_leaderboard_csv(entries, "reports/leaderboard_intraday.csv")

    if entries:
        best = entries[0]
        console.print(f"\n[bold green]WINNER: {best.strategy_name}[/bold green]")
        console.print(f"  Score: {best.composite_score:.1f} | Sharpe: {best.avg_sharpe:.2f} | Win Rate: {best.avg_win_rate:.1%} | Verdict: {best.verdict}")
        console.print(f"  Best instrument: {best.best_symbol}")

if __name__ == "__main__":
    main()
