#!/usr/bin/env python3
"""
scripts/run_backtest.py — Run the full backtest suite

Usage:
    python scripts/run_backtest.py                    # Full universe, all strategies
    python scripts/run_backtest.py --quick            # Nifty 50 only (fast test)
    python scripts/run_backtest.py --interval 1d      # Daily data
    python scripts/run_backtest.py --top 10           # Show top 10 strategies
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console

from data.universe import (
    NIFTY50_STOCKS, get_equity_universe, get_index_universe,
    get_commodity_universe, get_full_universe
)
from data.fetcher import fetch_universe_data
from strategies.standard import get_all_standard_strategies
from strategies.mathematical import get_all_mathematical_strategies
from backtesting.engine import run_full_backtest
from backtesting.leaderboard import build_leaderboard, print_leaderboard, save_leaderboard_csv

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Columnly Stocks Backtester")
    parser.add_argument("--quick",    action="store_true", help="Quick mode: Nifty 50 only")
    parser.add_argument("--interval", default="1d",        help="Data interval: 1d, 5m, 15m (default: 1d)")
    parser.add_argument("--top",      type=int, default=20,help="Number of top strategies to display")
    parser.add_argument("--math-only",action="store_true", help="Test only mathematical strategies")
    parser.add_argument("--std-only", action="store_true", help="Test only standard strategies")
    args = parser.parse_args()

    console.rule("[bold cyan]Columnly Stocks — Backtesting Engine[/bold cyan]")

    # Select universe
    if args.quick:
        instruments = NIFTY50_STOCKS[:10]
        console.print("[yellow]Quick mode: testing 10 Nifty 50 stocks[/yellow]")
    else:
        instruments = (
            get_equity_universe()    # All equity
            + get_index_universe()   # Indices
            + get_commodity_universe()  # Commodities
        )
        console.print(f"[green]Full universe: {len(instruments)} instruments[/green]")

    # Select strategies
    standard_strategies    = [] if args.math_only else get_all_standard_strategies()
    mathematical_strategies = [] if args.std_only  else get_all_mathematical_strategies()
    all_strategies = standard_strategies + mathematical_strategies
    console.print(f"[green]{len(all_strategies)} strategies loaded ({len(standard_strategies)} standard + {len(mathematical_strategies)} mathematical)[/green]")

    # Fetch data
    console.print(f"\n[cyan]Fetching {args.interval} data for {len(instruments)} instruments...[/cyan]")
    instrument_data = fetch_universe_data(
        instruments,
        interval=args.interval,
        delay_seconds=0.3,
    )
    console.print(f"[green]{len(instrument_data)} instruments with data[/green]")

    if not instrument_data:
        console.print("[red]No data available. Check your internet connection or add DHAN_ACCESS_TOKEN to .env[/red]")
        sys.exit(1)

    # Run backtests
    console.print(f"\n[cyan]Running {len(all_strategies) * len(instrument_data)} backtests...[/cyan]")
    results = run_full_backtest(all_strategies, instrument_data)
    console.print(f"[green]{len(results)} valid results[/green]")

    # Build leaderboard
    entries = build_leaderboard(results)

    console.print(f"\n[bold green]BACKTEST COMPLETE — {len(entries)} strategies ranked[/bold green]\n")
    print_leaderboard(entries, top_n=args.top)

    # Save outputs
    Path("reports").mkdir(exist_ok=True)
    save_leaderboard_csv(entries)

    # Print top recommendation
    if entries:
        best = entries[0]
        console.print(f"\n[bold green]TOP STRATEGY: {best.strategy_name}[/bold green]")
        console.print(f"  Composite score: {best.composite_score:.1f}")
        console.print(f"  Avg Sharpe: {best.avg_sharpe:.2f}")
        console.print(f"  Avg Win Rate: {best.avg_win_rate:.1%}")
        console.print(f"  Verdict: {best.verdict}")
        console.print(f"  Best on: {best.best_symbol}")
        console.print(f"\n[yellow]Results saved to reports/leaderboard.csv[/yellow]")


if __name__ == "__main__":
    main()
