# Stocks — Autonomous AI Trading Agent

> **Status:** Paper trading mode. Live trading requires Dhan/Groww API credentials.

An intelligent trading agent that identifies the best market opportunities daily, backtests 55+ strategies across the entire Indian market universe, and executes trades with disciplined risk management.

---

## Core Philosophy

**Reduce risk first. Maximise profit second.**

The agent never trades:
- When India VIX > 30 (fear regime)
- After daily loss limit is hit (₹500)
- Without a clear stop loss defined before entry

---

## What's Inside

```
├── data/
│   ├── universe.py          # Complete Indian market universe (equity, indices, commodities, ETFs)
│   └── fetcher.py           # Dhan API + Yahoo Finance fallback, with local Parquet cache
│
├── strategies/
│   ├── base.py              # Abstract BaseStrategy — all strategies inherit this
│   ├── standard/            # 30+ classical TA strategies (EMA, MACD, Supertrend, RSI, BB, VWAP, ORB...)
│   └── mathematical/        # 15 mathematically derived strategies (see below)
│
├── backtesting/
│   ├── engine.py            # Realistic backtester (brokerage, STT, slippage, partial profits, trailing stop)
│   └── leaderboard.py       # Composite scoring + rich-formatted leaderboard
│
├── intelligence/
│   └── segment_scorer.py    # Daily segment momentum scoring + candidate identification
│
├── agent/
│   └── paper_agent.py       # Paper trading agent (switch to live via EXECUTION_MODE=live)
│
├── scripts/
│   ├── setup.py             # One-time setup
│   ├── run_backtest.py      # Full backtest suite CLI
│   └── morning_scan.py      # Daily 9:00 AM scan
│
└── config/
    └── settings.py          # All configuration via .env
```

---

## 15 Mathematical Strategies

| # | Strategy | Mathematical Basis |
|---|---|---|
| 1 | Kalman Filter Momentum | Bayesian optimal state estimation |
| 2 | Ornstein-Uhlenbeck Mean Reversion | Stochastic differential equation |
| 3 | Volatility Regime Switching | ATR percentile regime classification |
| 4 | Hurst Exponent Classifier | Rescaled range fractal analysis |
| 5 | Entropy-Based Breakout | Shannon information entropy |
| 6 | Polynomial Regression Channel | Degree-2 polynomial least squares |
| 7 | Adaptive RSI | Autocorrelation cycle detection |
| 8 | Spectral Momentum (FFT) | Fast Fourier Transform dominant frequency |
| 9 | Autocorrelation Momentum | Lag-1 serial correlation filter |
| 10 | FRAMA | Mandelbrot fractal dimension |
| 11 | Half-Life Mean Reversion | OU half-life = ln(2)/θ |
| 12 | Volume-Weighted Momentum | Volume-confirmed price velocity |
| 13 | Regime-Conditioned Stochastic | ADX regime + stochastic switching |
| 14 | Price Velocity + Acceleration | First and second price derivatives |
| 15 | Multi-Timeframe Convergence | 3-timeframe EMA agreement |

---

## Transaction Costs (Realistic)

The backtester includes ALL real-world costs:

| Cost | Value |
|---|---|
| Brokerage | ₹20 flat per order |
| STT (intraday sell) | 0.025% |
| Exchange txn charge | 0.00345% |
| SEBI fee | 0.0001% |
| GST on brokerage | 18% |
| Stamp duty | 0.003% |
| Slippage (per side) | 0.03% |

---

## Risk Management

- **Daily loss limit:** ₹500 — agent stops trading for the day
- **Max trades/day:** 10
- **Position sizing:** ATR-based, risk 1.5% of capital per trade
- **Stop loss:** Always placed before entry (never moved against you)
- **Partial profit:** 50% closed at 1R, stop moved to breakeven
- **Trailing stop:** On remaining 50% after 1R hit
- **EOD forced close:** All positions closed by 3:25 PM

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Vishesh90/Stocks.git
cd Stocks
python scripts/setup.py

# 2. Add API credentials to .env
#    DHAN_ACCESS_TOKEN=your_token_here

# 3. Quick backtest (Nifty 50, daily data)
python scripts/run_backtest.py --quick

# 4. Full backtest (all instruments, all strategies)
python scripts/run_backtest.py

# 5. Daily morning scan
python scripts/morning_scan.py
```

---

## Data Sources

| Source | Used For | Data Range |
|---|---|---|
| **Dhan API** (primary) | NSE equities, indices, F&O | 5 years intraday, daily to inception |
| **Yahoo Finance** (fallback) | All instruments | Varies |

No Dhan account? The system falls back to Yahoo Finance automatically. Add `DHAN_ACCESS_TOKEN` to `.env` for the full 5-year dataset.

---

## Backtest Output

```
Strategy Leaderboard — Top 20
┌──────┬──────────────────────────────┬───────┬────────┬───────┬──────┬────────────┬────────────┬────────┬──────────┐
│ Rank │ Strategy                     │ Score │ Sharpe │ Win % │ PF   │ Net PnL    │ Max DD     │ Trades │ Verdict  │
├──────┼──────────────────────────────┼───────┼────────┼───────┼──────┼────────────┼────────────┼────────┼──────────┤
│  1   │ MultiTimeframeConvergence    │  73.2 │  2.41  │ 68.2% │ 2.31 │ ₹18,420    │ ₹-2,100    │  847   │ STRONG   │
│  2   │ KalmanMomentum               │  68.7 │  2.18  │ 64.1% │ 2.08 │ ₹14,730    │ ₹-2,850    │  934   │ STRONG   │
│  ...                                                                                                              │
└──────┴──────────────────────────────┴───────┴────────┴───────┴──────┴────────────┴────────────┴────────┴──────────┘
```

---

## Roadmap

- [ ] Live execution via Groww API
- [ ] Options backtesting (Black-Scholes P&L simulation)
- [ ] Web dashboard (Flask + Chart.js)
- [ ] Multi-instrument correlation filter
- [ ] Walk-forward validation
- [ ] Telegram/WhatsApp daily report

---

## Disclaimer

This is a personal research project. Past backtest performance does not guarantee future returns. Trading involves substantial risk of loss. Always start with paper trading. Comply with SEBI regulations for algorithmic trading.
