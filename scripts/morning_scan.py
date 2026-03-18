#!/usr/bin/env python3
"""
scripts/morning_scan.py — Run every day at 9:00 AM

Usage:
    python scripts/morning_scan.py

Schedule this with cron or Task Scheduler:
    cron: 0 9 * * 1-5 /usr/bin/python3 /path/to/scripts/morning_scan.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.paper_agent import PaperAgent

if __name__ == "__main__":
    agent = PaperAgent()
    agent.morning_scan()
