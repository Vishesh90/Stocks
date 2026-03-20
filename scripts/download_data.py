#!/usr/bin/env python3
"""
scripts/download_data.py — Bulk historical data downloader

INTENT:
    Downloads 5 years of 1m OHLCV data for all 536 instruments from the
    Dhan API and stores them as Parquet files. Once downloaded, all backtests
    run from cache — zero API calls, zero rate limit issues, works offline.

    536 instruments:
      498 Nifty 500 equities  (all with real Dhan security IDs)
       22 indices              (Nifty, Bank Nifty, sectoral — Dhan IDX_I)
       10 commodities          (Gold, Crude, etc. — Dhan MCX_COMM)
        6 ETFs                 (NiftyBeES, BankBeES, etc. — Dhan NSE_EQ)

DATA SOURCE:
    Dhan API only. No Yahoo Finance. Yahoo caps 1m data at 8 days — useless.

RATE LIMIT:
    Dhan API limit: 10 req/s.
    We sleep 0.125s between each batch call = 8 req/s = 80% of limit.
    Each instrument = ~22 batches (5yr ÷ 85-day windows).
    536 instruments × 22 batches × 0.125s = ~25 minutes total.

STORAGE:
    1m data: ~250MB per instrument × 536 = ~134GB total on disk.
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
    get_equity_universe, get_index_universe, get_commodity_universe, get_etf_universe
)
from data.fetcher import fetch_ohlcv

console = Console()


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
    parser.add_argument("--interval", default="1m",            help="Data interval: 1m, 5m, 15m (default: 1m)")
    parser.add_argument("--output",   default=None,            help='Custom folder e.g. "F:\\Stocks\\1m Nifty 500 stock data"')
    parser.add_argument("--resume",   action="store_true",     help="Skip already-downloaded instruments")
    parser.add_argument("--nifty50",  action="store_true",     help="Nifty 50 only (50 instruments, quick test)")
    parser.add_argument("--nifty100", action="store_true",     help="Nifty 50 + Next 50 (100 instruments)")
    parser.add_argument("--years",    type=int, default=5,     help="Years of history to download (default: 5)")
    args = parser.parse_args()

    # ── SET CUSTOM OUTPUT DIR ─────────────────────────────────────────────────
    # Must happen BEFORE any import of fetcher internals so the patched
    # settings.data_cache_dir is picked up by _cache_path().
    cache_root = Path(args.output) if args.output else Path("data/cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.output:
        import os
        os.environ["DATA_CACHE_DIR"] = str(cache_root)
        # Patch the already-instantiated settings singleton directly.
        # object.__setattr__ bypasses Pydantic's frozen-field guard.
        from config.settings import settings
        object.__setattr__(settings, "data_cache_dir", cache_root)

    # ── HEADER ────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]Columnly Stocks — Data Downloader[/bold cyan]")
    console.print(f"[yellow]Interval: {args.interval} | Rate: 8 req/s (80% of Dhan limit) | Years: {args.years}[/yellow]")
    console.print(f"[green]Output directory: {cache_root.resolve()}[/green]")

    # Progress file lives inside the output folder
    progress_file = cache_root / "download_progress.json"

    # ── SELECT UNIVERSE ───────────────────────────────────────────────────────
    if args.nifty50:
        instruments = NIFTY50_STOCKS
        console.print(f"[yellow]Mode: Nifty 50 ({len(instruments)} instruments)[/yellow]")
    elif args.nifty100:
        instruments = NIFTY50_STOCKS + NIFTY_NEXT50_STOCKS
        console.print(f"[yellow]Mode: Nifty 100 ({len(instruments)} instruments)[/yellow]")
    else:
        instruments = (
            get_equity_universe()      # 498 Nifty 500 stocks
            + get_index_universe()     # 22 indices
            + get_commodity_universe() # 10 commodities
            + get_etf_universe()       # 6 ETFs
        )                              # = 536 total
        console.print(f"[green]Mode: Full universe ({len(instruments)} instruments)[/green]")

    # ── RESUME LOGIC ──────────────────────────────────────────────────────────
    completed: set = set()
    if args.resume and progress_file.exists():
        try:
            data = json.loads(progress_file.read_text())
            completed = set(data.get("completed", []))
            console.print(f"[cyan]Resuming: {len(completed)} already downloaded, skipping them[/cyan]")
        except Exception:
            console.print("[yellow]Could not read progress file — starting fresh[/yellow]")

    remaining = [i for i in instruments if i.symbol not in completed]
    console.print(f"[cyan]{len(remaining)} instruments to download[/cyan]\n")

    if not remaining:
        console.print("[bold green]Nothing to download — all instruments already cached.[/bold green]")
        return

    # ── TIME / STORAGE ESTIMATE ───────────────────────────────────────────────
    batches_per_instrument = max(1, int(args.years * 365 / 85) + 1)
    BATCH_DELAY_S = 0.125  # 0.125s per batch = 8 req/s = 80% of Dhan's 10 req/s limit
    est_seconds = len(remaining) * batches_per_instrument * BATCH_DELAY_S
    mb_per_instrument = 250 if args.interval == "1m" else 50
    console.print(f"[dim]Estimated time  : {est_seconds / 3600:.1f} hours ({est_seconds / 60:.0f} minutes)[/dim]")
    console.print(f"[dim]Storage estimate: ~{len(remaining) * mb_per_instrument // 1024}GB "
                  f"({len(remaining) * mb_per_instrument}MB)[/dim]\n")

    # ── DOWNLOAD LOOP ─────────────────────────────────────────────────────────
    success_count  = 0
    fail_count     = 0
    failed_symbols = []
    start_time     = time.time()

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

        for idx, instrument in enumerate(remaining):
            symbol = instrument.symbol
            progress.update(task, description=f"[cyan]{symbol}[/cyan] ({idx + 1}/{len(remaining)})")

            ok = download_instrument(instrument, interval=args.interval, years=args.years)

            if ok:
                success_count += 1
                completed.add(symbol)
            else:
                fail_count += 1
                failed_symbols.append(symbol)

            # Persist progress after every instrument — crash-safe resume
            progress_file.write_text(json.dumps({"completed": list(completed)}))
            progress.advance(task)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    console.print()
    console.rule("[bold green]Download Complete[/bold green]")

    table = Table(show_header=True)
    table.add_column("Metric",  style="cyan")
    table.add_column("Value",   style="green")
    table.add_row("Total instruments",       str(len(remaining)))
    table.add_row("Successfully downloaded", str(success_count))
    table.add_row("Failed",                  str(fail_count))
    table.add_row("Time taken",              f"{elapsed / 60:.1f} minutes")
    table.add_row("Cache location",          str((cache_root / args.interval).resolve()))
    console.print(table)

    if failed_symbols:
        console.print(f"\n[yellow]Failed instruments ({len(failed_symbols)}):[/yellow]")
        console.print(", ".join(failed_symbols))
        console.print("\n[dim]Re-run with --resume to retry failed instruments[/dim]")

    if success_count > 0:
        console.print(f"\n[bold green]Data ready. Run backtests with:[/bold green]")
        console.print(f"  python scripts/run_backtest.py --interval {args.interval}")

    # Clean up progress file only when everything succeeded
    if fail_count == 0 and success_count > 0:
        progress_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
