# Agent Implementation Plan — Multi-Factor Data Pipeline & Trading System

> **Purpose**: Step-by-step agent prompts to build every data feed, wire them into the 40 mathematical strategies, run the full backtest, analyse results, integrate Alpaca for US markets, and build the live trading agent. Each task is self-contained — an agent can execute it without reading any other task.

---

## How To Use This Document

Each task below is a **complete agent prompt**. Copy the entire task block (including the "Files to Read" and "Acceptance Criteria" sections) and paste it into a new agent session. The agent will have everything it needs.

Tasks are ordered by dependency. Do not skip ahead — each task assumes the previous ones are complete.

---

## Task Index

| ID | Name | Effort | Depends On |
|----|------|--------|------------|
| **A1** | Run Full Backtest (OHLCV-only) | 2h runtime | Data download complete |
| **A2** | Analyse Backtest Results | 30 min | A1 |
| **A3** | VIX Context Feed | 2 hours | — |
| **A4** | Sector Rotation Feed | 4 hours | — |
| **A5** | FII/DII Institutional Flow Feed | 1 day | — |
| **A6** | Context Builder + Strategy Wiring | 4 hours | A3, A4, A5 |
| **A7** | Options Chain Feed (Dhan) | 1 day | — |
| **A8** | Macro Indicators Feed | 4 hours | — |
| **A9** | Sentiment Pipeline (FinBERT) | 2 days | — |
| **A10** | Central Bank Calendar | 2 hours | — |
| **A11** | Geopolitical Risk Scorer | 1 day | A9 (reuses news fetcher) |
| **A12** | Wire All Feeds Into Context Builder | 4 hours | A7, A8, A9, A10, A11 |
| **A13** | Re-Run Backtest With Full Context | 2h runtime | A12 |
| **A14** | Alpaca US Market Integration | 1 day | — |
| **A15** | Live Trading Agent (Dhan) | 2 days | A2 |

---

## A1 — Run Full Backtest (OHLCV-Only, All 40 Mathematical + 14 Standard Strategies)

### Files to Read First
- `strategies/STRATEGY_INPUTS_AND_DATA_PLAN.md` — understand all 40 strategies
- `backtesting/engine.py` — the backtest runner
- `backtesting/leaderboard.py` — result ranking
- `strategies/mathematical/__init__.py` — original 15 strategies
- `strategies/mathematical/advanced.py` — 25 advanced strategies
- `strategies/standard/__init__.py` — 14 standard TA strategies
- `scripts/run_backtest.py` — existing backtest script
- `scripts/run_local_full.py` — full backtest with resource limits
- `config/settings.py` — capital, risk settings

### What to Build

Create `scripts/run_full_backtest.py` — a new script that:

1. Loads all cached Parquet files from the data directory (the download is already complete or in progress)
2. Instantiates all 54 strategies: `get_all_mathematical_strategies()` (40) + `get_all_standard_strategies()` (14)
3. Runs `run_full_backtest()` from `backtesting/engine.py` with all strategies × all instruments that have data
4. Saves results to `reports/full_backtest_results.parquet` (one row per strategy-instrument pair)
5. Builds and saves the leaderboard to `reports/full_leaderboard.csv`
6. Prints a summary table: top 20 strategy-instrument pairs by Sharpe ratio

**Critical details**:
- Use `ThreadPoolExecutor` with `max_workers=4` (not ProcessPoolExecutor — strategy objects can't be pickled)
- Skip instruments with fewer than 500 bars (not enough data for the advanced strategies that need 200+ bar windows)
- Set a 60-second timeout per strategy-instrument pair to prevent hangs on slow strategies (Lyapunov, PersistentHomology)
- Log progress every 50 completions
- Save intermediate results every 100 completions (crash-safe)
- The script should accept `--data-dir` to point at the Parquet cache location
- The script should accept `--math-only` and `--std-only` flags

### Acceptance Criteria
- [ ] Script runs without errors on at least 10 instruments
- [ ] Results Parquet file is saved with columns: strategy_name, symbol, total_trades, win_rate, profit_factor, sharpe_ratio, sortino_ratio, max_drawdown, net_pnl, expectancy
- [ ] Leaderboard CSV is saved and sorted by composite score
- [ ] No strategy crashes the entire run (all exceptions are caught per-pair)

### Commit Message
```
feat(backtest): full 54-strategy backtest runner with crash-safe resume
```

---

## A2 — Analyse Backtest Results

### Files to Read First
- `reports/full_backtest_results.parquet` (output of A1)
- `reports/full_leaderboard.csv` (output of A1)
- `backtesting/leaderboard.py`
- `intelligence/segment_scorer.py`

### What to Build

Create `scripts/analyse_results.py` that reads the backtest results and produces:

1. **Top 10 strategies overall** — ranked by median Sharpe across all instruments they traded
2. **Top 10 instruments overall** — ranked by median Sharpe across all strategies that traded them
3. **Top 20 strategy-instrument pairs** — ranked by composite score (Sharpe × profit_factor × win_rate)
4. **Strategy category analysis** — group by category (original_math, advanced_math, standard_ta) and compare median performance
5. **Segment analysis** — which market segments (IT, Banking, Pharma, etc.) are most profitable for which strategy types
6. **Failure analysis** — which strategies produced 0 trades on which instruments, and why (insufficient data? no signals?)
7. **Correlation matrix** — are the top strategies correlated (same signals) or diversified?
8. **Risk analysis** — max drawdown distribution, worst-case scenarios

Save all outputs to:
- `reports/analysis_summary.md` — human-readable markdown report
- `reports/top_pairs.csv` — top 50 strategy-instrument pairs for live trading candidate list
- `reports/strategy_correlation.csv` — pairwise signal correlation matrix

### Acceptance Criteria
- [ ] Summary report is generated with all 8 analyses
- [ ] Top pairs CSV identifies at least 10 viable candidates (Sharpe > 1.0, profit_factor > 1.5, win_rate > 50%)
- [ ] Correlation matrix shows whether top strategies are diversified

### Commit Message
```
feat(analysis): comprehensive backtest result analyser with segment scoring
```

---

## A3 — VIX Context Feed

### Files to Read First
- `data/universe.py` — India VIX is already defined as `INDIA_VIX` in `BROAD_INDICES`
- `data/fetcher.py` — existing Parquet cache pattern
- `strategies/mathematical/advanced.py` — search for `_extract_context_boost` to see how VIX is consumed

### What to Build

Create `data/vix_fetcher.py`:

```python
class VIXFetcher:
    def __init__(self, cache_dir: Path):
        """Loads India VIX data from the same Parquet cache as OHLCV."""
    
    def get_vix_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Returns the India VIX value at or just before the given timestamp."""
    
    def get_latest_vix(self) -> float:
        """Returns the most recent VIX value from cache."""
    
    def get_vix_context(self) -> dict:
        """Returns {"vix": 18.5} for strategy context consumption."""
```

**Key implementation details**:
- India VIX data is already being downloaded as part of the index universe (symbol `INDIA_VIX`, dhan_security_id `20`)
- The VIX Parquet file is in the same cache directory as all other instruments
- Load it once at startup, keep in memory as a pandas Series indexed by timestamp
- `get_vix_at()` uses `asof()` for nearest-timestamp lookup (VIX may not have a value at every minute)
- For US markets (future Alpaca integration), add a `get_cboe_vix()` method that fetches `^VIX` via yfinance

### Acceptance Criteria
- [ ] `VIXFetcher` loads India VIX from existing Parquet cache
- [ ] `get_vix_at()` returns correct values for arbitrary timestamps
- [ ] `get_vix_context()` returns the dict format expected by `_extract_context_boost()`
- [ ] Falls back to `None` gracefully if VIX data is not available

### Commit Message
```
feat(data): VIX context feed from existing Parquet cache
```

---

## A4 — Sector Rotation Feed

### Files to Read First
- `data/universe.py` — `SECTORAL_INDICES` list (NIFTY_IT, NIFTY_BANK, etc.)
- `data/fetcher.py` — Parquet cache pattern
- `intelligence/segment_scorer.py` — existing segment scoring logic

### What to Build

Create `intelligence/sector_rotation.py`:

```python
class SectorRotation:
    def __init__(self, cache_dir: Path):
        """Loads all sectoral index data from Parquet cache."""
    
    def compute_sector_momentum(self, as_of: pd.Timestamp, lookback_days: int = 20) -> dict[str, float]:
        """Returns {sector_name: momentum_pct} for each sector."""
    
    def get_sector_rank(self, segment: str, as_of: pd.Timestamp) -> int:
        """Returns the rank (1=strongest) of a given sector by momentum."""
    
    def compute_relative_strength(self, stock_df: pd.DataFrame, nifty_df: pd.DataFrame, lookback_days: int = 20) -> float:
        """Stock return / Nifty 50 return over lookback period."""
    
    def get_sector_context(self, segment: str, stock_df: pd.DataFrame, nifty_df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
        """Returns {"sector_momentum_rank": 3, "relative_strength": 1.15}"""
```

**Key implementation details**:
- All sectoral index data is already downloaded (NIFTYIT, NIFTYPHARMA, etc.)
- Map each stock's `segment` field (from `universe.py`) to the corresponding sectoral index
- Momentum = 20-day return of the sectoral index
- Relative strength = stock's 20-day return / Nifty 50's 20-day return
- Rank sectors 1 (strongest) to N (weakest) by momentum
- No external data needed — everything comes from existing Parquet files

### Acceptance Criteria
- [ ] Loads all 12 sectoral indices from Parquet cache
- [ ] Correctly maps stock segments to sectoral indices
- [ ] Momentum and relative strength calculations are mathematically correct
- [ ] Returns context dict in the format expected by `_extract_context_boost()`

### Commit Message
```
feat(intelligence): sector rotation feed from existing sectoral index data
```

---

## A5 — FII/DII Institutional Flow Feed

### Files to Read First
- `strategies/mathematical/advanced.py` — search for `fii_dii_flows` in `_extract_context_boost()`
- `data/fetcher.py` — for the Parquet caching pattern to follow

### What to Build

Create `data/institutional_flows.py`:

```python
class InstitutionalFlowFetcher:
    def __init__(self, db_path: Path = Path("data/fii_dii.db")):
        """SQLite storage for FII/DII daily data."""
    
    def fetch_daily_from_nse(self, date: Optional[str] = None) -> dict:
        """
        Scrapes the NSE FII/DII daily report from:
        https://www.nseindia.com/api/fiidiiTradeReact
        
        Returns: {"date": "2026-03-21", "fii_net": 1234.5, "dii_net": -567.8}
        """
    
    def backfill(self, start_date: str, end_date: str):
        """Backfills historical FII/DII data from NSE archives."""
    
    def get_flows_at(self, date: str) -> Optional[dict]:
        """Returns FII/DII data for a specific date from SQLite."""
    
    def get_fii_dii_context(self, date: Optional[str] = None) -> dict:
        """Returns {"fii_net": 1234.5, "dii_net": -567.8, "fii_oi_change": 0}"""
```

**Key implementation details**:
- NSE publishes FII/DII data daily at ~6 PM IST at `https://www.nseindia.com/api/fiidiiTradeReact`
- The NSE website requires specific headers (User-Agent, Accept, Referer) and a session cookie — first hit the homepage to get cookies, then call the API
- Store in SQLite (not Parquet) because this is daily data with simple key-value structure
- For backtesting, we need historical data — NSE archives go back several years
- `fii_oi_change` (FII open interest change in derivatives) is harder to get — set to 0 initially, add later
- The fetcher should be resilient to NSE website changes (try/except with clear error messages)

### Acceptance Criteria
- [ ] Successfully scrapes today's FII/DII data from NSE
- [ ] Stores data in SQLite with date as primary key
- [ ] `get_fii_dii_context()` returns the dict format expected by strategies
- [ ] Backfill function can populate at least 1 year of historical data
- [ ] Handles NSE website being down gracefully (returns None, doesn't crash)

### Commit Message
```
feat(data): FII/DII institutional flow fetcher with NSE scraping and SQLite storage
```

---

## A6 — Context Builder + Strategy Wiring

### Files to Read First
- `data/vix_fetcher.py` (output of A3)
- `intelligence/sector_rotation.py` (output of A4)
- `data/institutional_flows.py` (output of A5)
- `strategies/mathematical/advanced.py` — the `_extract_context_boost()` function and how strategies consume context
- `backtesting/engine.py` — how `generate_signals()` is called

### What to Build

Create `data/context_builder.py`:

```python
class ContextBuilder:
    """
    Assembles the multi-factor context dict for strategy consumption.
    Each data source is optional — missing sources produce empty sub-dicts.
    """
    
    def __init__(self, cache_dir: Path, fii_dii_db: Path = Path("data/fii_dii.db")):
        self.vix = VIXFetcher(cache_dir)
        self.sector = SectorRotation(cache_dir)
        self.flows = InstitutionalFlowFetcher(fii_dii_db)
        # Future: self.options, self.sentiment, self.macro, etc.
    
    def build(self, symbol: str, segment: str, stock_df: pd.DataFrame,
              nifty_df: pd.DataFrame, timestamp: pd.Timestamp) -> dict:
        """Returns the full context dict for a single strategy call."""
        context = {}
        
        try:
            vix_val = self.vix.get_vix_at(timestamp)
            if vix_val is not None:
                context["macro"] = {"vix": vix_val}
        except Exception:
            pass
        
        try:
            context["sector_rotation"] = self.sector.get_sector_context(
                segment, stock_df, nifty_df, timestamp)
        except Exception:
            pass
        
        try:
            date_str = timestamp.strftime("%Y-%m-%d")
            flows = self.flows.get_fii_dii_context(date_str)
            if flows:
                context["fii_dii_flows"] = flows
        except Exception:
            pass
        
        return context
```

**Then modify `backtesting/engine.py`**:
- Add an optional `context_builder: Optional[ContextBuilder] = None` parameter to `run_backtest()`
- If provided, build context before calling `strategy.generate_signals(df, symbol, context=context)`
- The context should be built once per instrument (not per bar) using the latest available data
- For strategies that don't accept `context` (the original 15), catch the TypeError and call without it

### Acceptance Criteria
- [ ] `ContextBuilder` assembles context from all three Phase 2 feeds
- [ ] `run_backtest()` passes context to advanced strategies when available
- [ ] Original 15 strategies still work without context (backward compatible)
- [ ] Missing data sources don't crash the builder (graceful degradation)

### Commit Message
```
feat(data): context builder wiring VIX, sector rotation, and FII/DII into backtest engine
```

---

## A7 — Options Chain Feed (Dhan API)

### Files to Read First
- `data/fetcher.py` — Dhan API pattern (authentication, rate limiting)
- `config/settings.py` — where Dhan credentials are stored
- `strategies/mathematical/advanced.py` — search for `options_chain` in `_extract_context_boost()`

### What to Build

Create `data/options_fetcher.py`:

```python
class OptionsFetcher:
    def __init__(self):
        self.dhan = dhanhq(settings.dhan_client_id, settings.dhan_access_token)
    
    def fetch_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Fetches the full option chain for a symbol from Dhan API."""
    
    def compute_pcr(self, chain: pd.DataFrame) -> float:
        """Volume-weighted Put-Call Ratio."""
    
    def compute_max_pain(self, chain: pd.DataFrame) -> float:
        """Strike price where total option buyer loss is maximised."""
    
    def compute_iv_skew(self, chain: pd.DataFrame, spot_price: float) -> float:
        """25-delta put IV minus 25-delta call IV."""
    
    def get_options_context(self, symbol: str) -> dict:
        """Returns {"pcr": 1.3, "max_pain": 22500, "iv_skew": 0.05, "iv_term_structure": -0.02}"""
```

**Key implementation details**:
- Dhan provides option chain data via their API — check the `dhanhq` SDK docs for the exact method
- PCR = total put volume / total call volume (use OI-weighted PCR as secondary)
- Max pain = iterate through all strikes, compute total loss for option buyers at each strike, find the minimum
- IV skew requires Black-Scholes IV computation — use `py_vollib` (already in requirements.txt) or `scipy.optimize.brentq`
- For backtesting, historical option chain data is NOT available from Dhan — this feed is for live trading only
- For backtesting, we can approximate PCR from the Nifty VIX level (high VIX ≈ high PCR)
- Rate limit: respect Dhan's 10 req/s limit (same 0.125s sleep pattern)

### Acceptance Criteria
- [ ] Fetches live option chain for Nifty and Bank Nifty
- [ ] PCR computation is correct (verified against NSE website)
- [ ] Max pain computation produces reasonable values (near current spot)
- [ ] Returns context dict in the format expected by strategies
- [ ] Handles API errors gracefully

### Commit Message
```
feat(data): options chain fetcher with PCR, max pain, and IV skew from Dhan API
```

---

## A8 — Macro Indicators Feed

### Files to Read First
- `data/universe.py` — MCX commodities (GOLD, CRUDEOIL) are already in the universe
- `data/fetcher.py` — Parquet cache pattern
- `strategies/mathematical/advanced.py` — search for `macro` in `_extract_context_boost()`

### What to Build

Create `data/macro_fetcher.py`:

```python
class MacroFetcher:
    def __init__(self, cache_dir: Path):
        """Loads macro data from Parquet cache and Yahoo Finance."""
    
    def fetch_dxy(self, from_date: str, to_date: str) -> pd.DataFrame:
        """US Dollar Index from Yahoo Finance (DX-Y.NYB)."""
    
    def fetch_us10y(self, from_date: str, to_date: str) -> pd.DataFrame:
        """US 10-Year Treasury Yield from Yahoo Finance (^TNX)."""
    
    def get_crude_price(self, timestamp: pd.Timestamp) -> Optional[float]:
        """From existing MCX CRUDEOIL Parquet data."""
    
    def get_gold_price(self, timestamp: pd.Timestamp) -> Optional[float]:
        """From existing MCX GOLD Parquet data."""
    
    def get_macro_context(self, timestamp: pd.Timestamp) -> dict:
        """Returns {"vix": 18.5, "dxy": 104.2, "us10y": 4.35, "crude_oil": 78.5, "gold": 2150.0}"""
```

**Key implementation details**:
- Crude oil and gold prices are already being downloaded as MCX instruments — just read from Parquet
- DXY and US10Y need to be fetched from Yahoo Finance using `yfinance` (already in requirements.txt)
- Download daily data for DXY and US10Y (not intraday — Yahoo caps at 8 days for 1m)
- Store DXY/US10Y as separate Parquet files in the cache directory
- For intraday timestamps, use the most recent daily value (macro indicators don't change intraday in a meaningful way for our purposes)
- VIX is handled by `VIXFetcher` (A3) — the macro context should merge with it

### Acceptance Criteria
- [ ] DXY and US10Y data downloaded and cached as Parquet
- [ ] Crude and gold prices read from existing MCX Parquet files
- [ ] `get_macro_context()` returns all 5 fields
- [ ] Historical data available for backtesting (at least 3 years)

### Commit Message
```
feat(data): macro indicators feed — DXY, US10Y from Yahoo; crude, gold from existing MCX data
```

---

## A9 — Sentiment Pipeline (FinBERT)

### Files to Read First
- `strategies/mathematical/advanced.py` — search for `sentiment` in `_extract_context_boost()`
- `requirements.txt` — check what's already installed

### What to Build

Create two files:

**`data/news_fetcher.py`**:
```python
class NewsFetcher:
    def fetch_google_news(self, query: str, lookback_hours: int = 4) -> list[dict]:
        """Fetches recent news via Google News RSS feed."""
    
    def fetch_for_symbol(self, symbol: str, company_name: str) -> list[dict]:
        """Fetches news for a specific stock."""
```

**`data/sentiment_model.py`**:
```python
class SentimentScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Loads FinBERT model from HuggingFace."""
    
    def score_text(self, text: str) -> float:
        """Returns sentiment score (-1 to +1) for a single text."""
    
    def score_articles(self, articles: list[dict]) -> float:
        """Returns aggregate sentiment score for a list of articles."""
    
    def get_sentiment_context(self, symbol: str, company_name: str) -> dict:
        """Returns {"news_score": 0.3, "social_score": 0.0, "fear_greed_index": 50}"""
```

**Key implementation details**:
- Google News RSS is free and doesn't require an API key: `https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en`
- Parse RSS with `feedparser` (add to requirements.txt)
- FinBERT from HuggingFace: `transformers` and `torch` needed (add to requirements.txt)
- Use the distilled version `ProsusAI/finbert` — it's 420MB, runs on CPU in ~0.5s per article
- For backtesting, sentiment data is NOT available historically — this feed is for live trading only
- `social_score` and `fear_greed_index` are set to 0/50 initially (placeholders for future Reddit/Twitter integration)
- Cache sentiment scores in SQLite to avoid re-scoring the same articles

### New Dependencies
```
feedparser>=6.0.0
transformers>=4.30.0
torch>=2.0.0
```

### Acceptance Criteria
- [ ] Google News RSS fetcher returns recent articles for Indian stocks
- [ ] FinBERT model loads and scores text correctly (positive text → positive score)
- [ ] Aggregate scoring handles empty article lists gracefully
- [ ] Returns context dict in the format expected by strategies
- [ ] Model loads in under 30 seconds on CPU

### Commit Message
```
feat(data): sentiment pipeline with Google News RSS and FinBERT scoring
```

---

## A10 — Central Bank Calendar

### Files to Read First
- `strategies/mathematical/advanced.py` — search for `central_bank` in `_extract_context_boost()`

### What to Build

Create `data/central_bank.py`:

```python
# Hardcoded calendar — updated quarterly
RBI_MPC_DATES = [
    "2026-04-09", "2026-06-06", "2026-08-08", "2026-10-09", "2026-12-05",
    "2027-02-07", "2027-04-09", "2027-06-08",
]

FOMC_DATES = [
    "2026-05-07", "2026-06-18", "2026-07-30", "2026-09-17",
    "2026-11-05", "2026-12-17", "2027-01-28", "2027-03-18",
]

class CentralBankCalendar:
    def days_to_next_rbi(self, as_of: date) -> int:
        """Days until next RBI MPC meeting."""
    
    def days_to_next_fomc(self, as_of: date) -> int:
        """Days until next FOMC meeting."""
    
    def get_central_bank_context(self, as_of: date, market: str = "india") -> dict:
        """Returns {"rate_decision_days": 12, "hawkish_dovish_score": 0.0}"""
```

**Key implementation details**:
- This is mostly static data — RBI publishes MPC dates a year in advance
- `hawkish_dovish_score` is set to 0.0 initially (neutral) — can be enhanced later with FinBERT on policy statements
- The calendar should be easy to update (just edit the date lists)
- For backtesting, compute `days_to_next_rbi` for any historical date by finding the nearest future date in the list

### Acceptance Criteria
- [ ] Correctly computes days to next RBI/FOMC meeting for any date
- [ ] Returns 0 on the day of a meeting
- [ ] Returns context dict in the expected format
- [ ] Calendar covers at least 2026-2027

### Commit Message
```
feat(data): central bank calendar with RBI MPC and FOMC dates
```

---

## A11 — Geopolitical Risk Scorer

### Files to Read First
- `data/news_fetcher.py` (output of A9 — reuses the Google News RSS fetcher)
- `strategies/mathematical/advanced.py` — search for `geopolitical` in `_extract_context_boost()`

### What to Build

Create `data/geopolitical.py`:

```python
GEOPOLITICAL_KEYWORDS = {
    "high_risk": ["war", "invasion", "nuclear", "sanctions", "coup", "martial law", "blockade"],
    "medium_risk": ["tariff", "trade war", "embargo", "military", "escalation", "conflict", "missile"],
    "low_risk": ["tension", "dispute", "protest", "election", "referendum", "negotiation"],
}

KEYWORD_WEIGHTS = {"high_risk": 3.0, "medium_risk": 2.0, "low_risk": 1.0}

class GeopoliticalRiskScorer:
    def __init__(self, news_fetcher: NewsFetcher):
        self.fetcher = news_fetcher
    
    def compute_risk_score(self, lookback_hours: int = 24) -> float:
        """
        Fetches recent global news, counts geopolitical keywords,
        returns a risk score 0-100.
        """
    
    def get_geopolitical_context(self) -> dict:
        """Returns {"risk_score": 35.0, "event_proximity_days": 0}"""
```

**Key implementation details**:
- Reuses the `NewsFetcher` from A9 — fetches news with queries like "India geopolitical", "global conflict", "trade war"
- Simple keyword counting with weights — no ML needed
- Score = (weighted keyword count / total articles) × 100, capped at 100
- `event_proximity_days` = 0 if any high_risk keyword found in last 24h, otherwise days since last high_risk article
- Cache the score for 1 hour (geopolitical risk doesn't change every minute)
- For backtesting, this feed is not available historically — returns default low-risk context

### Acceptance Criteria
- [ ] Fetches global news and counts geopolitical keywords
- [ ] Risk score is between 0 and 100
- [ ] High-risk keywords produce higher scores than low-risk
- [ ] Returns context dict in the expected format
- [ ] Caches results to avoid excessive API calls

### Commit Message
```
feat(data): geopolitical risk scorer with keyword-weighted news analysis
```

---

## A12 — Wire All Feeds Into Context Builder

### Files to Read First
- `data/context_builder.py` (output of A6)
- `data/options_fetcher.py` (output of A7)
- `data/macro_fetcher.py` (output of A8)
- `data/sentiment_model.py` (output of A9)
- `data/central_bank.py` (output of A10)
- `data/geopolitical.py` (output of A11)

### What to Build

Update `data/context_builder.py` to include all 7 data feeds:

```python
class ContextBuilder:
    def __init__(self, cache_dir: Path, ...):
        self.vix = VIXFetcher(cache_dir)              # Tier 4
        self.sector = SectorRotation(cache_dir)         # Tier 7
        self.flows = InstitutionalFlowFetcher(...)      # Tier 3
        self.options = OptionsFetcher()                  # Tier 2
        self.macro = MacroFetcher(cache_dir)             # Tier 6
        self.sentiment = SentimentScorer()               # Tier 5
        self.central_bank = CentralBankCalendar()        # Tier 8
        self.geopolitical = GeopoliticalRiskScorer(...)   # Tier 9
    
    def build(self, symbol: str, segment: str, ...) -> dict:
        # Assemble all 7 context sub-dicts
        # Each wrapped in try/except for graceful degradation
```

Also create `data/__init__.py` that exports all fetchers for clean imports.

### Acceptance Criteria
- [ ] All 7 feeds are wired into the context builder
- [ ] Each feed fails independently (one broken feed doesn't crash the others)
- [ ] The full context dict matches the format in `_extract_context_boost()`
- [ ] A smoke test builds context for one symbol and prints all available fields

### Commit Message
```
feat(data): complete context builder with all 7 multi-factor data feeds
```

---

## A13 — Re-Run Backtest With Full Context

### Files to Read First
- `scripts/run_full_backtest.py` (output of A1)
- `data/context_builder.py` (output of A12)
- `backtesting/engine.py` — the modified version from A6

### What to Build

Modify `scripts/run_full_backtest.py` to accept a `--with-context` flag:

- When `--with-context` is passed, instantiate `ContextBuilder` and pass it to `run_backtest()`
- Run the same 54 strategies × all instruments
- Save results to `reports/full_backtest_with_context.parquet`
- Compare results with the OHLCV-only backtest: for each strategy-instrument pair, compute the delta in Sharpe, win_rate, and net_pnl
- Save comparison to `reports/context_impact_analysis.csv`

**Note**: Most context feeds (sentiment, options, geopolitical) don't have historical data. The backtest with context will primarily benefit from:
- VIX context (historical data available)
- Sector rotation (computed from historical sectoral indices)
- FII/DII flows (if backfilled)
- Macro indicators (DXY, US10Y have historical data from Yahoo)

### Acceptance Criteria
- [ ] Backtest runs with context without errors
- [ ] Comparison CSV shows the marginal impact of context on each strategy
- [ ] At least VIX and sector rotation context are active during the backtest

### Commit Message
```
feat(backtest): re-run with multi-factor context and impact analysis
```

---

## A14 — Alpaca US Market Integration

### Files to Read First
- `data/fetcher.py` — Dhan fetcher pattern to replicate
- `data/universe.py` — instrument definition pattern
- `scripts/download_data.py` — bulk downloader pattern
- `config/settings.py` — where to add Alpaca credentials

### What to Build

**1. Update `config/settings.py`**:
```python
# Broker — Alpaca
alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
alpaca_api_secret: str = Field(default="", env="ALPACA_API_SECRET")
alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
```

**2. Create `data/universe_us.py`**:
- Define S&P 500 instruments (symbol, name, sector)
- Use the same `Instrument` dataclass but with `exchange=Exchange.NYSE` or `Exchange.NASDAQ`
- Add `Exchange.NYSE` and `Exchange.NASDAQ` to the `Exchange` enum in `universe.py`

**3. Create `data/alpaca_fetcher.py`**:
```python
class AlpacaFetcher:
    def __init__(self):
        self.api = tradeapi.REST(settings.alpaca_api_key, settings.alpaca_api_secret, settings.alpaca_base_url)
    
    def fetch_ohlcv(self, symbol: str, interval: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetches OHLCV from Alpaca with same Parquet caching as Dhan."""
```

**4. Create `scripts/download_data_us.py`**:
- Same pattern as `scripts/download_data.py`
- Downloads S&P 500 data with `--resume` flag
- Alpaca free tier: 200 req/min for historical data

### New Dependencies
```
alpaca-trade-api>=3.0.0
```

### Acceptance Criteria
- [ ] S&P 500 universe defined with all 500 symbols
- [ ] Alpaca fetcher downloads and caches OHLCV as Parquet
- [ ] Same cache/resume pattern as Dhan downloader
- [ ] Backtest engine can run on US data without modification

### Commit Message
```
feat(data): Alpaca US market integration with S&P 500 universe and Parquet caching
```

---

## A15 — Live Trading Agent (Dhan API)

### Files to Read First
- `agent/paper_agent.py` — existing paper trading agent
- `backtesting/engine.py` — trade execution logic to replicate
- `config/settings.py` — risk parameters (daily_loss_limit, max_trades_per_day, capital_inr)
- `reports/top_pairs.csv` (output of A2) — which strategy-instrument pairs to trade
- `data/context_builder.py` (output of A12) — multi-factor context for live signals

### What to Build

Create `agent/live_agent.py`:

```python
class LiveTradingAgent:
    def __init__(self, strategy_instrument_pairs: list[tuple[BaseStrategy, Instrument]],
                 context_builder: ContextBuilder):
        self.pairs = strategy_instrument_pairs
        self.context = context_builder
        self.dhan = dhanhq(settings.dhan_client_id, settings.dhan_access_token)
        self.daily_pnl = 0.0
        self.trades_today = 0
    
    def morning_scan(self):
        """
        Runs at 9:15 AM IST. For each strategy-instrument pair:
        1. Fetch latest 200 bars of 1m data
        2. Build context
        3. Generate signals
        4. Filter by confidence > 0.70
        5. Queue top 3 signals for execution
        """
    
    def execute_signal(self, signal: Signal, instrument: Instrument):
        """
        Places a real order via Dhan API:
        1. Check daily loss limit (INR 500)
        2. Check max trades per day (10)
        3. Compute position size (1.5% risk per trade)
        4. Place bracket order (entry + stop + target)
        """
    
    def monitor_positions(self):
        """
        Runs every 1 minute during market hours:
        1. Check open positions
        2. Update trailing stops
        3. Force close at 3:25 PM
        4. Update daily P&L
        """
    
    def end_of_day_report(self):
        """
        Runs at 3:30 PM IST:
        1. Close any remaining positions
        2. Log all trades to SQLite
        3. Generate daily P&L report
        4. Send notification (email/Telegram)
        """
```

**Key implementation details**:
- Use Dhan's bracket order API for automatic stop-loss and target placement
- Daily loss limit: stop all trading if cumulative P&L < -INR 500
- Max 10 trades per day (configurable via settings)
- Capital: INR 10,000-20,000 (configurable)
- Risk per trade: 1.5% of capital (configurable)
- Only trade the top strategy-instrument pairs identified in A2
- Start in paper mode (`execution_mode: "paper"`) — verify for 2 weeks before going live
- Log every decision (signal generated, signal filtered, order placed, order filled, position closed) to SQLite for audit

**Also create `scripts/run_live_agent.py`**:
- CLI entry point that starts the agent
- Accepts `--paper` (default) and `--live` flags
- Runs the morning scan, then monitors positions until 3:30 PM
- Uses `schedule` library for timing

### Acceptance Criteria
- [ ] Agent runs in paper mode without placing real orders
- [ ] Morning scan generates signals from top strategy-instrument pairs
- [ ] Position sizing respects the 1.5% risk rule
- [ ] Daily loss limit stops trading when breached
- [ ] EOD report logs all trades and P&L
- [ ] All decisions are logged to SQLite for audit

### Commit Message
```
feat(agent): live trading agent with Dhan execution, risk management, and audit logging
```

---

## Dependency Graph

```
Data Download (already running)
    │
    ├── A1: Full Backtest (OHLCV-only)
    │       │
    │       └── A2: Analyse Results
    │               │
    │               └── A15: Live Trading Agent ──────────────────┐
    │                                                              │
    ├── A3: VIX Feed ──────────┐                                  │
    ├── A4: Sector Rotation ───┤                                  │
    ├── A5: FII/DII Feed ──────┤                                  │
    │                           │                                  │
    │                           └── A6: Context Builder (Phase 2) │
    │                                                              │
    ├── A7: Options Chain ─────┐                                  │
    ├── A8: Macro Feed ────────┤                                  │
    ├── A9: Sentiment ─────────┤                                  │
    ├── A10: Central Bank ─────┤                                  │
    ├── A11: Geopolitical ─────┤                                  │
    │                           │                                  │
    │                           └── A12: Full Context Builder ────┤
    │                                       │                      │
    │                                       └── A13: Re-Run ──────┘
    │
    └── A14: Alpaca US (parallel, independent)
```

**Parallelisable tasks** (can run simultaneously):
- A3 + A4 + A5 (all independent data feeds)
- A7 + A8 + A9 + A10 + A11 (all independent data feeds)
- A14 (completely independent of everything else)

**Sequential dependencies**:
- A1 → A2 → A15
- A3 + A4 + A5 → A6
- A7 + A8 + A9 + A10 + A11 → A12 → A13

---

## Timeline Estimate

| Week | Tasks | What Gets Done |
|------|-------|---------------|
| 1 | A1, A3, A4, A5, A14 | Full OHLCV backtest + 3 easy data feeds + Alpaca setup |
| 2 | A2, A6, A7, A8 | Result analysis + context builder + options + macro |
| 3 | A9, A10, A11 | Sentiment pipeline + central bank + geopolitical |
| 4 | A12, A13 | Full context wiring + re-run backtest |
| 5 | A15 | Live trading agent (paper mode) |
| 6+ | — | Paper trading validation → live trading |

**Total engineering effort**: ~3-4 weeks of agent work.
**Total elapsed time**: ~5-6 weeks (including 2 weeks of paper trading validation before going live).
