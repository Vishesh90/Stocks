#!/usr/bin/env python3
"""
scripts/download_data.py — Bulk historical data downloader

INTENT:
    Downloads 5 years of 5m (or 1m) OHLCV data for every instrument in the
    universe from Dhan API and stores it as Parquet files in data/cache/.
    Once downloaded, all backtests run from cache — zero API calls, zero
    rate limit issues, works offline forever.

USAGE:
    python scripts/download_data.py                    # 5m data, all instruments
    python scripts/download_data.py --interval 1m      # 1m data (larger files)
    python scripts/download_data.py --resume           # Skip already-downloaded
    python scripts/download_data.py --nifty50          # Nifty 50 only (fast test)
    python scripts/download_data.py --delay 0.5        # Slower = safer rate limit

STORAGE ESTIMATE:
    5m data: ~50MB per instrument × 1000 instruments = ~50GB total
    1m data: ~250MB per instrument × 1000 instruments = ~250GB total

RATE LIMIT:
    Dhan allows 5 req/sec. We use 0.3s delay = ~3 req/sec (safe).
    Each instrument needs ~22 batches (5yr / 85 days).
    1000 instruments × 22 batches × 0.3s = ~1.8 hours total.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from data.universe import (
    NIFTY50_STOCKS, NIFTY_NEXT50_STOCKS,
    get_equity_universe, get_index_universe, get_commodity_universe
)
from data.fetcher import fetch_ohlcv, _cache_path

console = Console()

PROGRESS_FILE = Path("data/cache/download_progress.json")


def load_progress() -> set:
    """Load set of already-downloaded symbols."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("completed", []))
    return set()


def save_progress(completed: set):
    """Save progress so we can resume after interruption."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({"completed": list(completed)}))


def download_instrument(instrument, interval: str, years: int = 5) -> bool:
    """
    Download full history for one instrument and cache it.
    Returns True if successful, False if failed.
    """
    to_date   = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    try:
        df = fetch_ohlcv(instrument, interval=interval, from_date=from_date, to_date=to_date)
        if df is not None and not df.empty:
            logger.debug(f"  {instrument.symbol}: {len(df)} candles downloaded")
            return True
        else:
            logger.warning(f"  {instrument.symbol}: no data returned")
            return False
    except Exception as e:
        logger.warning(f"  {instrument.symbol}: failed — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk data downloader for backtesting")
    parser.add_argument("--interval", default="5m",      help="Data interval: 5m, 1m, 15m (default: 5m)")
    parser.add_argument("--delay",    type=float, default=0.35, help="Delay between instruments in seconds (default: 0.35)")
    parser.add_argument("--resume",   action="store_true", help="Skip already-downloaded instruments")
    parser.add_argument("--nifty50",  action="store_true", help="Nifty 50 only")
    parser.add_argument("--nifty100", action="store_true", help="Nifty 50 + Next 50 only")
    parser.add_argument("--years",    type=int, default=5, help="Years of history (default: 5)")
    args = parser.parse_args()

    console.rule(f"[bold cyan]Columnly Stocks — Data Downloader[/bold cyan]")
    console.print(f"[yellow]Interval: {args.interval} | Delay: {args.delay}s | Years: {args.years}[/yellow]")

    # ── SELECT UNIVERSE ───────────────────────────────────────────────────────
    if args.nifty50:
        instruments = NIFTY50_STOCKS
        console.print(f"[yellow]Mode: Nifty 50 ({len(instruments)} instruments)[/yellow]")
    elif args.nifty100:
        instruments = NIFTY50_STOCKS + NIFTY_NEXT50_STOCKS
        console.print(f"[yellow]Mode: Nifty 100 ({len(instruments)} instruments)[/yellow]")
    else:
        instruments = (
            get_equity_universe()
            + get_index_universe()
            + get_commodity_universe()
        )
        console.print(f"[green]Mode: Full universe ({len(instruments)} instruments)[/green]")

    # ── RESUME LOGIC ──────────────────────────────────────────────────────────
    completed = load_progress() if args.resume else set()
    if completed:
        console.print(f"[cyan]Resuming: {len(completed)} already downloaded, skipping them[/cyan]")

    remaining = [i for i in instruments if i.symbol not in completed]
    console.print(f"[cyan]{len(remaining)} instruments to download[/cyan]\n")

    # ── ESTIMATE ──────────────────────────────────────────────────────────────
    batches_per_instrument = 22 if args.years == 5 else int(args.years * 365 / 85) + 1
    est_seconds = len(remaining) * batches_per_instrument * args.delay
    est_hours   = est_seconds / 3600
    console.print(f"[dim]Estimated time: {est_hours:.1f} hours ({est_seconds/60:.0f} minutes)[/dim]")
    console.print(f"[dim]Storage estimate: ~{len(remaining) * 50}MB for 5m data[/dim]\n")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0
    failed_symbols = []
    start_time    = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(remaining))

        for i, instrument in enumerate(remaining):
            symbol = instrument.symbol
            progress.update(task, description=f"[cyan]{symbol}[/cyan] ({i+1}/{len(remaining)})")

            ok = download_instrument(instrument, interval=args.interval, years=args.years)

            if ok:
                success_count += 1
                completed.add(symbol)
            else:
                fail_count += 1
                failed_symbols.append(symbol)

            # Save progress after every instrument — crash-safe
            save_progress(completed)
            progress.advance(task)

            # Rate limit protection
            time.sleep(args.delay)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    console.print()
    console.rule("[bold green]Download Complete[/bold green]")

    table = Table(show_header=True)
    table.add_column("Metric",  style="cyan")
    table.add_column("Value",   style="green")
    table.add_row("Total instruments",  str(len(remaining)))
    table.add_row("Successfully downloaded", str(success_count))
    table.add_row("Failed",     str(fail_count))
    table.add_row("Time taken", f"{elapsed/60:.1f} minutes")
    table.add_row("Cache location", str(Path("data/cache") / args.interval))
    console.print(table)

    if failed_symbols:
        console.print(f"\n[yellow]Failed instruments ({len(failed_symbols)}):[/yellow]")
        console.print(", ".join(failed_symbols))
        console.print(f"\n[dim]Re-run with --resume to retry failed instruments[/dim]")

    if success_count > 0:
        console.print(f"\n[bold green]Data ready. Run backtests with:[/bold green]")
        console.print(f"  python scripts/run_backtest.py --interval {args.interval}")
        # Clean up progress file on full success
        if fail_count == 0:
            PROGRESS_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
