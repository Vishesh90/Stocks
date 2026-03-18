#!/usr/bin/env python3
"""
scripts/setup.py — One-time environment setup

Run this first. Checks dependencies, creates .env, creates directories.
"""
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("Columnly Stocks — Setup")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Create directories
    for d in ["data/raw", "data/processed", "data/cache/1d", "data/cache/5m", "data/cache/15m", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

    # Create .env if not exists
    env_file = Path(".env")
    if not env_file.exists():
        Path(".env.example").read_text()
        env_file.write_text(Path(".env.example").read_text())
        print("✓ .env created from .env.example — ADD YOUR API KEYS")
    else:
        print("✓ .env already exists")

    # Install requirements
    print("\nInstalling requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)
    print("✓ Requirements installed")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Add DHAN_ACCESS_TOKEN to .env (get from dhan.co → API)")
    print("  2. Run: python scripts/run_backtest.py --quick  (fast test)")
    print("  3. Run: python scripts/run_backtest.py          (full backtest)")
    print("  4. Run: python scripts/morning_scan.py          (daily scan)")

if __name__ == "__main__":
    main()
