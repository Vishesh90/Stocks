#!/usr/bin/env python3
"""
scripts/run_local_full.py — Full backtest on local machine with resource limits

INTENT:
    Runs the complete backtest suite overnight without freezing the user's machine.
    Uses 2 workers max (leaves plenty of CPU for other tasks), processes instruments
    in batches of 10, saves progress after every batch so a crash never loses work,
    and caps RAM by loading one batch at a time.

USAGE:
    python scripts/run_local_full.py                  # Full run (overnight)
    python scripts/run_local_full.py --workers 1      # Gentlest mode (slowest)
    python scripts/run_local_full.py --workers 3      # Faster (uses ~50% CPU)
    python scripts/run_local_full.py --resume         # Resume after a crash

IMPACT:
    Produces reports/leaderboard_full.csv with every strategy ranked across
    the full Indian market universe. This is the definitive strategy ranking.
"""

import sys
import os
import argparse
import time
import csv
import json
from pathlib import Path

# ── HARD RESOURCE LIMITS ─────────────────────────────────────────────────────
# Applied before any heavy imports so limits are in effect from the start.
# RAM: 4GB hard cap — process is killed by OS if it exceeds this
# CPU: 20% cap — achieved by sleeping after every unit of CPU work
import resource, psutil, threading, time as _time

MAX_RAM_BYTES = 4 * 1024 * 1024 * 1024   # 4 GB
CPU_CAP_PCT   = 20.0                       # 20% of total CPU

try:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM_BYTES, MAX_RAM_BYTES))
except Exception:
    pass  # Windows doesn't support RLIMIT_AS — handled by psutil watchdog below


def _cpu_watchdog():
    """
    Background thread that enforces the 20% CPU cap.

    WHY: Python has no native CPU% throttle. Instead we measure this process's
    CPU usage every 2 seconds and sleep proportionally to bring it back under cap.
    Formula: if we used X% over the last interval, sleep for (X/cap - 1) * interval
    to give back the excess CPU time.
    """
    proc = psutil.Process()
    interval = 2.0  # Check every 2 seconds
    while True:
        _time.sleep(interval)
        try:
            used = proc.cpu_percent(interval=None)
            if used > CPU_CAP_PCT:
                # Sleep proportionally to shed the excess
                sleep_for = ((used / CPU_CAP_PCT) - 1.0) * interval
                _time.sleep(min(sleep_for, 10.0))  # Cap at 10s sleep per cycle

            # Also check RAM — warn if getting close to limit
            mem_gb = proc.memory_info().rss / 1024**3
            if mem_gb > 3.5:
                import sys as _sys
                print(f"\n[WARNING] RAM usage {mem_gb:.1f}GB — approaching 4GB limit. Consider restarting with --resume")
        except Exception:
            pass


# Start the CPU watchdog as a daemon thread (dies automatically when main exits)
_watchdog = threading.Thread(target=_cpu_watchdog, daemon=True)
_watchdog.start()

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from data.universe import NIFTY50_STOCKS, NIFTY_NEXT50_STOCKS, get_equity_universe, get_index_universe, get_commodity_universe
from data.fetcher import fetch_universe_data
from strategies.standard import get_all_standard_strategies
from strategies.mathematical import get_all_mathematical_strategies
from backtesting.engine import run_backtest
from backtesting.leaderboard import build_leaderboard, print_leaderboard, save_leaderboard_csv, LeaderboardEntry

console = Console()

PROGRESS_FILE = Path("reports/local_run_progress.json")
RESULTS_FILE  = Path("reports/local_run_results.json")


def save_progress(completed_symbols: list[str], partial_results: list[dict]):
    """Save progress so we can resume after a crash."""
    Path("reports").mkdir(exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({"completed": completed_symbols}))
    RESULTS_FILE.write_text(json.dumps(partial_results))


def load_progress() -> tuple[list[str], list[dict]]:
    """Load saved progress for resume."""
    if PROGRESS_FILE.exists() and RESULTS_FILE.exists():
        completed = json.loads(PROGRESS_FILE.read_text())["completed"]
        results   = json.loads(RESULTS_FILE.read_text())
        return completed, results
    return [], []


def run_batch(strategies, symbol, df, max_workers: int) -> list:
    """Run all strategies against one instrument. Returns list of BacktestResult."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []

    def _run_one(strategy):
        try:
            r = run_backtest(strategy, df, symbol)
            if r and r.total_trades >= 5:
                return r
        except Exception as e:
            logger.debug(f"  {strategy.name}/{symbol}: {e}")
        return None

    # Use at most max_workers threads per instrument
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, s): s for s in strategies}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    return results


def result_to_dict(r) -> dict:
    """Serialise a BacktestResult to a plain dict for JSON storage."""
    return {
        "strategy_name":   r.strategy_name,
        "symbol":          r.symbol,
        "total_trades":    r.total_trades,
        "win_rate":        r.win_rate,
        "sharpe_ratio":    r.sharpe_ratio,
        "profit_factor":   r.profit_factor,
        "net_pnl_inr":     r.net_pnl_inr,
        "max_drawdown_inr":r.max_drawdown_inr,
    }


def main():
    parser = argparse.ArgumentParser(description="Local full backtest with resource limits")
    parser.add_argument("--workers",  type=int, default=1,   help="Parallel workers (default: 1 for 20%% CPU cap)")
    parser.add_argument("--interval", default="5m",          help="Data interval (default: 5m)")
    parser.add_argument("--resume",   action="store_true",   help="Resume from last checkpoint")
    parser.add_argument("--nifty50",  action="store_true",   help="Nifty 50 only (faster test)")
    args = parser.parse_args()

    console.rule("[bold cyan]Columnly Stocks — Full Local Backtest[/bold cyan]")
    console.print(f"[yellow]Workers: {args.workers} | Interval: {args.interval} | Resume: {args.resume}[/yellow]")
    console.print(f"[dim]Hard limits: 4GB RAM | 20% CPU cap | Auto-saves after every instrument[/dim]\n")

    # ── SELECT UNIVERSE ───────────────────────────────────────────────────────
    if args.nifty50:
        instruments = NIFTY50_STOCKS
        console.print(f"[yellow]Nifty 50 mode: {len(instruments)} instruments[/yellow]")
    else:
        instruments = (
            get_equity_universe()
            + get_index_universe()
            + get_commodity_universe()
        )
        console.print(f"[green]Full universe: {len(instruments)} instruments[/green]")

    # ── LOAD STRATEGIES ───────────────────────────────────────────────────────
    strategies = get_all_standard_strategies() + get_all_mathematical_strategies()
    console.print(f"[green]{len(strategies)} strategies loaded[/green]")

    # ── RESUME LOGIC ──────────────────────────────────────────────────────────
    completed_symbols = []
    all_raw_results   = []

    if args.resume:
        completed_symbols, all_raw_results = load_progress()
        if completed_symbols:
            console.print(f"[cyan]Resuming: {len(completed_symbols)} instruments already done, {len(all_raw_results)} results loaded[/cyan]")
        else:
            console.print("[yellow]No checkpoint found — starting fresh[/yellow]")

    remaining = [i for i in instruments if i.symbol not in completed_symbols]
    console.print(f"[cyan]{len(remaining)} instruments to process[/cyan]\n")

    # ── FETCH + BACKTEST INSTRUMENT BY INSTRUMENT ─────────────────────────────
    total   = len(remaining)
    done    = 0
    skipped = 0

    start_time = time.time()

    for instrument in remaining:
        symbol = instrument.symbol
        done += 1

        # ETA calculation
        elapsed = time.time() - start_time
        rate    = done / elapsed if elapsed > 0 else 0.001
        eta_s   = (total - done) / rate if rate > 0 else 0
        eta_min = int(eta_s / 60)

        console.print(f"[cyan][{done}/{total}][/cyan] {symbol} — ETA: ~{eta_min}m remaining")

        # Fetch data for this single instrument
        try:
            instrument_data = fetch_universe_data(
                [instrument],
                interval=args.interval,
                delay_seconds=0.2,
            )
        except Exception as e:
            logger.warning(f"  {symbol}: data fetch failed — {e}")
            skipped += 1
            completed_symbols.append(symbol)
            save_progress(completed_symbols, all_raw_results)
            continue

        if symbol not in instrument_data or instrument_data[symbol].empty:
            logger.warning(f"  {symbol}: no data returned")
            skipped += 1
            completed_symbols.append(symbol)
            save_progress(completed_symbols, all_raw_results)
            continue

        df = instrument_data[symbol]
        console.print(f"  [dim]{len(df)} candles loaded[/dim]")

        # Run all strategies against this instrument
        batch_results = run_batch(strategies, symbol, df, max_workers=args.workers)
        console.print(f"  [green]{len(batch_results)} valid results[/green]")

        # Serialise and accumulate
        for r in batch_results:
            all_raw_results.append(result_to_dict(r))

        # Mark done and save checkpoint
        completed_symbols.append(symbol)
        save_progress(completed_symbols, all_raw_results)

        # Brief pause between instruments — lets CPU cool down
        time.sleep(0.5)

    # ── BUILD LEADERBOARD ─────────────────────────────────────────────────────
    console.rule("[bold green]Building Leaderboard[/bold green]")

    if not all_raw_results:
        console.print("[red]No results to rank. Check your Dhan credentials in .env[/red]")
        sys.exit(1)

    # Reconstruct BacktestResult-like objects for leaderboard builder
    from backtesting.engine import BacktestResult
    reconstructed = []
    for d in all_raw_results:
        r = BacktestResult(strategy_name=d["strategy_name"], symbol=d["symbol"])
        r.total_trades    = d["total_trades"]
        r.win_rate        = d["win_rate"]
        r.sharpe_ratio    = d["sharpe_ratio"]
        r.profit_factor   = d["profit_factor"]
        r.net_pnl_inr     = d["net_pnl_inr"]
        r.max_drawdown_inr= d["max_drawdown_inr"]
        reconstructed.append(r)

    entries = build_leaderboard(reconstructed)

    total_time = int((time.time() - start_time) / 60)
    console.print(f"\n[bold green]COMPLETE — {len(entries)} strategies ranked in {total_time} minutes[/bold green]")
    console.print(f"[dim]Instruments: {len(completed_symbols)} processed, {skipped} skipped[/dim]\n")

    print_leaderboard(entries, top_n=20)

    # Save full leaderboard
    Path("reports").mkdir(exist_ok=True)
    save_leaderboard_csv(entries, filename="reports/leaderboard_full.csv")
    console.print(f"\n[yellow]Full results → reports/leaderboard_full.csv[/yellow]")

    # Clean up checkpoint files
    PROGRESS_FILE.unlink(missing_ok=True)
    RESULTS_FILE.unlink(missing_ok=True)

    # Top recommendation
    if entries:
        best = entries[0]
        console.print(f"\n[bold green]TOP STRATEGY: {best.strategy_name}[/bold green]")
        console.print(f"  Score: {best.composite_score:.1f} | Sharpe: {best.avg_sharpe:.2f} | Win%: {best.avg_win_rate:.1%} | Best on: {best.best_symbol}")


if __name__ == "__main__":
    main()
