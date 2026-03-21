# Strategy Input Requirements & Data Acquisition Plan

> **Purpose**: This document specifies every input required by all 40 mathematical strategies, categorises them into data tiers, and provides a concrete engineering plan for acquiring each data source.

---

## Table of Contents

1. [Input Taxonomy](#1-input-taxonomy)
2. [Per-Strategy Input Matrix](#2-per-strategy-input-matrix)
3. [Data Tier Definitions](#3-data-tier-definitions)
4. [Tier 0 — OHLCV Core (DONE)](#4-tier-0--ohlcv-core)
5. [Tier 1 — Derived Computations (NO NEW DATA)](#5-tier-1--derived-computations)
6. [Tier 2 — Options Chain & Derivatives Data](#6-tier-2--options-chain--derivatives-data)
7. [Tier 3 — Institutional Flow Data (FII/DII)](#7-tier-3--institutional-flow-data)
8. [Tier 4 — Volatility Index (VIX / India VIX)](#8-tier-4--volatility-index)
9. [Tier 5 — Sentiment Data (News + Social)](#9-tier-5--sentiment-data)
10. [Tier 6 — Macroeconomic Indicators](#10-tier-6--macroeconomic-indicators)
11. [Tier 7 — Sector Rotation & Relative Strength](#11-tier-7--sector-rotation--relative-strength)
12. [Tier 8 — Central Bank & Policy Signals](#12-tier-8--central-bank--policy-signals)
13. [Tier 9 — Geopolitical Risk](#13-tier-9--geopolitical-risk)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Architecture: Data Pipeline Design](#15-architecture-data-pipeline-design)

---

## 1. Input Taxonomy

Every strategy input falls into one of two categories:

### A. Core Inputs (used directly in the mathematical model)

These are the columns the strategy's `generate_signals()` method reads from the DataFrame or computes internally. Without these, the strategy cannot produce signals.

| Input | Type | Source |
|-------|------|--------|
| `open` | float | OHLCV DataFrame |
| `high` | float | OHLCV DataFrame |
| `low` | float | OHLCV DataFrame |
| `close` | float | OHLCV DataFrame |
| `volume` | float | OHLCV DataFrame |

### B. Context Inputs (multi-factor confidence boosters)

These are passed via the optional `context: dict` parameter in the 25 advanced strategies. They do not change the mathematical model — they adjust the confidence score of signals by ±0.15 max. Without these, the strategy still works at full capability using OHLCV alone.

| Context Key | Sub-fields | Effect |
|-------------|-----------|--------|
| `options_chain` | `pcr`, `max_pain`, `iv_skew`, `iv_term_structure` | +0.03 confidence if PCR confirms direction |
| `fii_dii_flows` | `fii_net`, `dii_net`, `fii_oi_change` | +0.05 confidence if flow confirms direction |
| `sentiment` | `news_score`, `social_score`, `fear_greed_index` | +0.02 confidence if sentiment confirms direction |
| `macro` | `vix`, `dxy`, `us10y`, `crude_oil`, `gold` | -0.05 confidence if VIX > 30 and direction is long |
| `sector_rotation` | `sector_momentum_rank`, `relative_strength` | (hook defined, not yet wired) |
| `central_bank` | `rate_decision_days`, `hawkish_dovish_score` | (hook defined, not yet wired) |
| `geopolitical` | `risk_score`, `event_proximity_days` | (hook defined, not yet wired) |

---

## 2. Per-Strategy Input Matrix

### Legend

| Symbol | Meaning |
|--------|---------|
| **C** | `close` price series |
| **H** | `high` price series |
| **L** | `low` price series |
| **O** | `open` price series |
| **V** | `volume` series |
| **ATR** | Average True Range (derived from H, L, C) |
| **RET** | Returns (derived from C) |
| **LOG** | Log prices (derived from C) |
| **EMA** | Exponential Moving Average (derived from C) |
| **RSI** | Relative Strength Index (derived from C) |
| **CTX** | Multi-factor context dict (optional) |

### Original 15 Strategies

| # | Strategy | Core Inputs | Derived Internally | Min Bars | Context |
|---|----------|------------|-------------------|----------|---------|
| 1 | KalmanFilterMomentum | C, H, L | ATR, Kalman-filtered price, slope | 20 | — |
| 2 | OrnsteinUhlenbeck | C, H, L | ATR, LOG, OU params (θ, μ, σ), z-score | 61 | — |
| 3 | VolatilityRegimeSwitching | C, H, L | ATR, ATR percentile rank, EMA(9,21), RSI(14) | 114 | — |
| 4 | HurstExponentClassifier | C, H, L | ATR, Hurst exponent (R/S), EMA(9,21), RSI(14) | 100 | — |
| 5 | EntropyBreakout | C, H, L | ATR, RET, Shannon entropy, entropy percentile | 120 | — |
| 6 | PolynomialRegressionChannel | C, H, L | ATR, polynomial fit, residual z-score | 50 | — |
| 7 | AdaptiveRSI | C, H, L | ATR, autocorrelation cycle length, adaptive RSI | 60 | — |
| 8 | SpectralMomentum | C, H, L | ATR, FFT magnitudes/phases, dominant frequency | 69 | — |
| 9 | AutocorrelationMomentum | C, H, L | ATR, RET, lag-1 autocorrelation, EMA(9,21) | 31 | — |
| 10 | FRAMA | C, H, L | ATR, fractal dimension, adaptive EMA | 32 | — |
| 11 | HalfLifeMeanReversion | C, H, L | ATR, LOG, OU params, half-life, z-score | 61 | — |
| 12 | VolumeWeightedMomentum | C, H, L, **V** | ATR, momentum, volume MA, volume ratio | 20 | — |
| 13 | RegimeConditionedStochastic | C, H, L | ATR, Stochastic K/D, ADX | 20+ | — |
| 14 | PriceVelocityAcceleration | C, H, L | ATR, smoothed price, velocity (1st deriv), acceleration (2nd deriv) | 7 | — |
| 15 | MultiTimeframeConvergence | C, H, L, O, **V** | ATR, EMA(9,21) on 3 timeframes, resampled OHLCV | 30+ | — |

### Advanced 25 Strategies

| # | Strategy | Core Inputs | Derived Internally | Min Bars | Context |
|---|----------|------------|-------------------|----------|---------|
| 16 | FokkerPlanckDensity | C, H, L | ATR, LOG RET, Kramers-Moyal drift/diffusion, FP PDE solution, KDE | 110 | CTX |
| 17 | LévyFlightDetector | C, H, L | ATR, RET, Hill tail index α | 105 | CTX |
| 18 | RényiEntropyDivergence | C, H, L | ATR, RET, Rényi entropy (q=2), Rényi divergence | 205 | CTX |
| 19 | WaveletDenoisedMomentum | C, H, L | ATR, Haar DWT coefficients, denoised price, slope | 74 | CTX |
| 20 | KramersMoyalDrift | C, H, L | ATR, Epanechnikov kernel weights, drift, diffusion, SNR | 125 | CTX |
| 21 | FisherInfoRegimeDetector | C, H, L | ATR, RET, Fisher information trace, FI rate of change | 70 | CTX |
| 22 | IsingPhaseTransition | C, H, L | ATR, RET, spin chain, magnetisation, susceptibility, coupling J | 100 | CTX |
| 23 | OptimalTransportDivergence | C, H, L | ATR, RET, Wasserstein-1 distance (sorted quantiles) | 205 | CTX |
| 24 | HawkesProcessClustering | C, H, L | ATR, RET, event detection (2σ), inter-event times, Hawkes intensity (μ,α,β) | 210 | CTX |
| 25 | VariationalModeDecomposition | C, H, L | ATR, VMD modes (ADMM), mode energies, dominant mode slope | 138 | CTX |
| 26 | FractionalBrownianMotion | C, H, L | ATR, RET, variogram, Hurst H, EMA(9,21), RSI(14) | 125 | CTX |
| 27 | MaxEntropySpectrum | C, H, L | ATR, Burg AR coefficients, spectral density, dominant frequency, AR forecast | 74 | CTX |
| 28 | GrangerCausalityVolPrice | C, H, L, **V** | ATR, RET, volume changes, lagged regression matrices, F-statistic, forecast | 110 | CTX |
| 29 | TsallisEntropy | C, H, L | ATR, RET, Tsallis entropy (q=1.5), entropy rate of change | 80 | CTX |
| 30 | LangevinDynamics | C, H, L | ATR, empirical density histogram, potential U(x), force F(x), noise level | 105 | CTX |
| 31 | MutualInformationRate | C, H, L | ATR, fast returns (1-bar), slow returns (5-bar), 2D histogram, MI | 110 | CTX |
| 32 | EmpiricalModeDecomposition | C, H, L | ATR, IMFs via sifting, mode energies, dominant IMF slope | 110 | CTX |
| 33 | BayesianChangepoint | C, H, L | ATR, RET, Normal-Inverse-Gamma posterior, run-length distribution, Student-t predictive | 50 | CTX |
| 34 | HestonStochasticVolatility | C, H, L | ATR, RET, realised variance, Heston params (κ,θ,ξ,ρ), v/θ ratio | 130 | CTX |
| 35 | KolmogorovSmirnovShift | C, H, L | ATR, RET, KS statistic, p-value, mean shift | 205 | CTX |
| 36 | LyapunovChaosDetector | C, H, L | ATR, RET, time-delay embedding, nearest-neighbour divergence, λ₁ | 210 | CTX |
| 37 | CopulaTailDependence | C, H, L, **V** | ATR, RET, volume returns, rank transform, empirical copula, λ_U, λ_L | 105 | CTX |
| 38 | SpectralGapEstimator | C, H, L | ATR, RET, discretised states, transition matrix, eigenvalues, spectral gap γ, EMA(9,21), RSI(14) | 205 | CTX |
| 39 | InformationRatioAdaptive | C, H, L | ATR, RET, rolling volatility, Kelly fraction, information ratio, regime conditioning | 70 | CTX |
| 40 | PersistentHomologySignal | C, H, L | ATR, RET, time-delay embedding, pairwise distances, single-linkage persistence, max persistence | 110 | CTX |

---

## 3. Data Tier Definitions

| Tier | Name | Status | Strategies That Benefit | Effort |
|------|------|--------|------------------------|--------|
| 0 | OHLCV Core | **DONE** | All 40 (required) | — |
| 1 | Derived Computations | **DONE** | All 40 (computed internally) | — |
| 2 | Options Chain & Derivatives | NOT STARTED | All 25 advanced (+0.03 conf) | Medium |
| 3 | FII/DII Institutional Flows | NOT STARTED | All 25 advanced (+0.05 conf) | Medium |
| 4 | Volatility Index (VIX) | NOT STARTED | All 25 advanced (-0.05 conf filter) | Easy |
| 5 | Sentiment (News + Social) | NOT STARTED | All 25 advanced (+0.02 conf) | Hard |
| 6 | Macroeconomic Indicators | NOT STARTED | All 25 advanced (macro context) | Medium |
| 7 | Sector Rotation | NOT STARTED | All 25 advanced (hook only) | Medium |
| 8 | Central Bank Policy | NOT STARTED | All 25 advanced (hook only) | Easy |
| 9 | Geopolitical Risk | NOT STARTED | All 25 advanced (hook only) | Hard |

---

## 4. Tier 0 — OHLCV Core

### Status: DONE

### What It Is

1-minute OHLCV (Open, High, Low, Close, Volume) candle data for 536 Indian instruments:
- 498 Nifty 500 stocks
- 22 indices (Nifty 50, Bank Nifty, sectoral indices)
- 10 MCX commodities (Crude, Gold, Silver, Natural Gas, etc.)
- 6 liquid ETFs

### Data Format

| Column | Type | Description |
|--------|------|-------------|
| `open` | float64 | Opening price of the candle |
| `high` | float64 | Highest price during the candle |
| `low` | float64 | Lowest price during the candle |
| `close` | float64 | Closing price of the candle |
| `volume` | int64/float64 | Number of shares/contracts traded |
| index | DatetimeIndex | Timestamp in Asia/Kolkata timezone |

### Source

- **Dhan API** — `dhanhq` Python SDK
- 85-day intraday batches, 5 years of history
- Rate limit: 8 requests/second (80% of Dhan's 10 req/s limit)
- Cached as Parquet files in `F:\Stocks\1m Nifty 500 stock data\`
- Crash-safe resume with `--resume` flag

### What Every Strategy Reads From OHLCV

| Derived Value | Formula | Used By |
|---------------|---------|---------|
| ATR(14) | EMA of max(H-L, abs(H-C_prev), abs(L-C_prev)) | All 40 strategies |
| Returns | `close.pct_change()` | 30 strategies |
| Log returns | `log(close / close.shift(1))` | 12 strategies |
| Log prices | `log(close)` | 5 strategies |
| EMA(9), EMA(21) | Exponential moving average | 8 strategies |
| RSI(14) | Relative strength index | 6 strategies |
| Volume MA | `volume.rolling(20).mean()` | 3 strategies |
| Volume ratio | `volume / volume_ma` | 3 strategies |
| Volume returns | `volume.pct_change()` | 2 strategies |
| Resampled OHLCV | 15m, 45m aggregation | 1 strategy |

---

## 5. Tier 1 — Derived Computations

### Status: DONE (computed inside each strategy)

These are not external data sources — they are mathematical quantities computed from OHLCV inside each strategy's `generate_signals()` method. Listed here for completeness.

| Computation | Mathematical Basis | Strategies |
|-------------|-------------------|------------|
| Kalman-filtered price | Bayesian state estimation | #1 |
| OU parameters (θ, μ, σ) | Linear regression on log-price differences | #2, #11 |
| ATR percentile rank | Rolling rank of ATR | #3 |
| Hurst exponent (R/S) | Rescaled range analysis | #4 |
| Hurst exponent (variogram) | Second-order structure function | #26 |
| Shannon entropy | -Σ p·log₂(p) on binned returns | #5 |
| Rényi entropy (q=2) | (1/(1-q))·log(Σ pᵢᵠ) | #18 |
| Tsallis entropy (q=1.5) | (1-Σ pᵢᵠ)/(q-1) | #29 |
| Polynomial regression | np.polyfit degree-2 | #6 |
| Autocorrelation (lag-1) | np.corrcoef on lagged returns | #7, #9 |
| FFT dominant frequency | scipy.fft on detrended prices | #8 |
| Fractal dimension | Price range method (Hausdorff) | #10 |
| Half-life | ln(2)/θ from OU fit | #11 |
| Fokker-Planck PDE | Crank-Nicolson finite difference | #16 |
| Hill tail index α | Order statistics of absolute returns | #17 |
| Rényi divergence | Cross-entropy of two distributions | #18 |
| Haar DWT + VisuShrink | Lifting scheme + universal threshold | #19 |
| Kramers-Moyal coefficients | Nadaraya-Watson kernel regression | #20 |
| Fisher information trace | n/σ² + n/(2σ⁴) | #21 |
| Ising observables (M, χ, J) | Spin chain statistics | #22 |
| Wasserstein-1 distance | Sorted quantile L1 distance | #23 |
| Hawkes intensity λ(t) | Self-exciting kernel sum | #24 |
| VMD modes | ADMM spectral optimisation | #25 |
| Burg AR coefficients | Maximum entropy method | #27 |
| Granger F-statistic | Restricted vs unrestricted regression | #28 |
| Langevin force F(x) | -∂U/∂x from empirical density | #30 |
| Mutual information | 2D histogram joint entropy | #31 |
| EMD IMFs | Sifting process with envelope interpolation | #32 |
| Bayesian run-length posterior | Normal-Inverse-Gamma conjugate update | #33 |
| Heston parameters (κ,θ,ξ,ρ) | Method of moments on realised variance | #34 |
| KS statistic + p-value | scipy.stats.ks_2samp | #35 |
| Lyapunov exponent λ₁ | Rosenstein nearest-neighbour divergence | #36 |
| Copula tail dependence (λ_U, λ_L) | Rank transform + conditional probability | #37 |
| Spectral gap γ | 1 - |λ₂| of transition matrix | #38 |
| Kelly fraction f* | μ/σ² | #39 |
| Persistent homology | Single-linkage persistence on embedded point cloud | #40 |

---

## 6. Tier 2 — Options Chain & Derivatives Data

### Impact: +0.03 confidence boost on all 25 advanced strategies

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `pcr` | Put-Call Ratio (volume-based) | Every 1 minute |
| `max_pain` | Strike price where most options expire worthless | Every 5 minutes |
| `iv_skew` | Difference in IV between OTM puts and OTM calls | Every 1 minute |
| `iv_term_structure` | IV of near-month vs far-month ATM options | Every 5 minutes |

### How It Improves Strategies

- **PCR > 1.2** with a long signal → market is hedged, institutions expect upside → +0.03 confidence
- **PCR < 0.8** with a short signal → excessive call buying, likely to reverse → +0.03 confidence
- **Max pain** acts as a gravitational attractor for price near expiry
- **IV skew** measures fear asymmetry — steep put skew = crash fear = contrarian long opportunity

### Data Sources (India)

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **NSE Option Chain** (nse-india.com) | Web scraping / unofficial API | Free | 3-5 min delay | Nifty, Bank Nifty, top 50 F&O stocks |
| **Dhan Options API** | `dhanhq` SDK `get_option_chain()` | Free (with account) | Real-time | All F&O instruments |
| **Sensibull API** | REST API | ₹1,000/mo | Real-time | Full chain with Greeks |
| **Upstox Options** | REST API v2 | Free (with account) | Real-time | All F&O |

### Data Sources (US — for Alpaca integration)

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **Alpaca Options** | REST API | Free (paper), $99/mo (live) | Real-time | All US options |
| **Polygon.io** | REST + WebSocket | $29/mo (Starter) | Real-time | Full US options chain |
| **CBOE DataShop** | FTP/REST | $500+/mo | EOD to 15-min | Institutional grade |

### Implementation Plan

```
File: data/options_fetcher.py

class OptionsFetcher:
    def fetch_option_chain(symbol: str) -> OptionChain:
        """Fetches current option chain for a symbol."""
        
    def compute_pcr(chain: OptionChain) -> float:
        """Volume-weighted put-call ratio."""
        
    def compute_max_pain(chain: OptionChain) -> float:
        """Strike where total option buyer loss is maximised."""
        
    def compute_iv_skew(chain: OptionChain) -> float:
        """25-delta put IV minus 25-delta call IV."""
        
    def compute_iv_term_structure(chain: OptionChain) -> float:
        """Near-month ATM IV minus far-month ATM IV."""

Storage: Parquet files per symbol per day
Cache: In-memory LRU cache (options data changes every minute)
Schedule: Fetch every 60 seconds during market hours (9:15-15:30 IST)
```

---

## 7. Tier 3 — Institutional Flow Data

### Impact: +0.05 confidence boost on all 25 advanced strategies (highest single-factor boost)

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `fii_net` | FII net buy/sell in ₹ crores (cash segment) | Daily (EOD) |
| `dii_net` | DII net buy/sell in ₹ crores (cash segment) | Daily (EOD) |
| `fii_oi_change` | FII open interest change in index futures | Daily (EOD) |

### How It Improves Strategies

- **FII net positive + long signal** → smart money confirms direction → +0.05 confidence
- **FII net negative + short signal** → smart money selling → +0.05 confidence
- FII/DII divergence (FII selling, DII buying) → market at inflection point

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **NSDL/CDSL FII data** | Web scraping from nsdl.co.in | Free | T+1 day | Cash + derivatives |
| **NSE FII/DII daily** | nse-india.com/reports | Free | EOD (6 PM IST) | Cash segment |
| **MoneyControl FII data** | Web scraping | Free | EOD | Cash + derivatives |
| **Trendlyne API** | REST API | ₹2,000/mo | EOD | Full FII/DII + sector breakdown |

### Implementation Plan

```
File: data/institutional_flows.py

class InstitutionalFlowFetcher:
    def fetch_daily_fii_dii() -> FIIDIIData:
        """Scrapes NSE FII/DII daily report."""
        
    def fetch_fii_derivatives() -> FIIDerivativesData:
        """FII index futures/options OI changes."""
        
    def get_latest_flows() -> dict:
        """Returns context-ready dict for strategy consumption."""
        # Returns: {"fii_net": 1234.5, "dii_net": -567.8, "fii_oi_change": 45000}

Storage: SQLite table (date, fii_net, dii_net, fii_index_fut_oi, fii_index_opt_oi)
Schedule: Daily at 6:30 PM IST (after NSE publishes)
Backfill: 5 years of historical FII/DII data from NSE archives
```

---

## 8. Tier 4 — Volatility Index

### Impact: -0.05 confidence penalty when VIX > 30 and signal is long (risk filter)

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `vix` | India VIX (NSE) or CBOE VIX (US) | Every 1 minute |

### How It Improves Strategies

- **VIX > 30** → extreme fear → long signals get -0.05 confidence penalty (reduces false longs in panic)
- **VIX < 12** → extreme complacency → can be used as breakout compression signal
- VIX term structure (contango vs backwardation) indicates regime

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **NSE India VIX** | nse-india.com (web scrape) | Free | 1-min delay | India VIX |
| **Dhan API** | Fetch INDIAVIX as an index | Free | Real-time | India VIX |
| **Yahoo Finance** | `yfinance` (`^INDIAVIX`, `^VIX`) | Free | 15-min delay | India VIX, CBOE VIX |
| **CBOE VIX** (US) | Alpaca / Polygon | Free-$29/mo | Real-time | CBOE VIX |

### Implementation Plan

```
File: data/vix_fetcher.py

class VIXFetcher:
    def fetch_india_vix() -> float:
        """Current India VIX value."""
        
    def fetch_cboe_vix() -> float:
        """Current CBOE VIX value (for US strategies)."""
        
    def get_vix_context() -> dict:
        """Returns {"vix": 18.5} for strategy context."""

Storage: Append to OHLCV Parquet (add vix column) or separate time series
Schedule: Every 60 seconds during market hours
Note: India VIX is already in universe.py as an index — just needs to be
      loaded and passed as context rather than traded.
```

---

## 9. Tier 5 — Sentiment Data

### Impact: +0.02 confidence boost on all 25 advanced strategies

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `news_score` | Aggregate sentiment of recent news (-1 to +1) | Every 5 minutes |
| `social_score` | Social media sentiment (Twitter/Reddit/StockTwits) | Every 5 minutes |
| `fear_greed_index` | CNN Fear & Greed Index (0-100) | Daily |

### How It Improves Strategies

- **News score > 0.3 + long signal** → positive news confirms direction → +0.02 confidence
- **News score < -0.3 + short signal** → negative news confirms direction → +0.02 confidence
- Sentiment extremes are contrarian indicators at market tops/bottoms

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **FinBERT** (local model) | HuggingFace `ProsusAI/finbert` | Free (GPU needed) | 1-2s per article | Any text |
| **NewsAPI.org** | REST API | Free (100 req/day), $449/mo (unlimited) | Minutes | Global news |
| **Google News RSS** | RSS feed parsing | Free | Minutes | Global news |
| **Alpha Vantage News** | REST API | Free (5 req/min) | Minutes | Financial news |
| **Reddit API** (r/IndianStreetBets) | PRAW library | Free | Minutes | Social sentiment |
| **Twitter/X API** | REST API v2 | $100/mo (Basic) | Real-time | Social sentiment |
| **StockTwits API** | REST API | Free | Real-time | US stock sentiment |
| **CNN Fear & Greed** | Web scraping | Free | Daily | US market sentiment |
| **Trendlyne Sentiment** | REST API | ₹2,000/mo | Daily | India stock sentiment |

### Implementation Plan

```
File: data/sentiment_fetcher.py
File: data/sentiment_model.py

class SentimentFetcher:
    def fetch_news(symbol: str, lookback_hours: int = 4) -> list[Article]:
        """Fetches recent news articles for a symbol."""
        
    def fetch_social(symbol: str) -> list[Post]:
        """Fetches recent social media posts."""

class SentimentScorer:
    def __init__(self):
        self.model = FinBERT()  # or distilled version for speed
        
    def score_articles(articles: list[Article]) -> float:
        """Returns aggregate sentiment score (-1 to +1)."""
        
    def score_social(posts: list[Post]) -> float:
        """Returns aggregate social sentiment (-1 to +1)."""
        
    def get_sentiment_context(symbol: str) -> dict:
        """Returns {"news_score": 0.3, "social_score": -0.1, "fear_greed_index": 45}"""

Storage: SQLite table (timestamp, symbol, news_score, social_score, source)
Schedule: Every 5 minutes during market hours
Model: FinBERT (770M params) or distilled FinBERT (66M params) for speed
GPU: Optional — CPU inference takes ~0.5s per article with distilled model
```

---

## 10. Tier 6 — Macroeconomic Indicators

### Impact: Populates the `macro` context dict for all 25 advanced strategies

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `vix` | (covered in Tier 4) | 1 minute |
| `dxy` | US Dollar Index | 1 minute |
| `us10y` | US 10-Year Treasury Yield | 1 minute |
| `crude_oil` | WTI/Brent crude oil price | 1 minute |
| `gold` | Gold spot price (XAU/USD) | 1 minute |

### How It Improves Strategies

- **DXY rising** → emerging market pressure → bearish bias for Indian equities
- **US10Y rising** → risk-off → bearish for growth stocks
- **Crude rising** → inflationary → bearish for oil-importing economies (India)
- **Gold rising** → safe-haven demand → mixed signal (bearish equity, bullish gold strategies)

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **Yahoo Finance** | `yfinance` (`DX-Y.NYB`, `^TNX`, `CL=F`, `GC=F`) | Free | 15-min delay | All macro |
| **Alpha Vantage** | REST API | Free (5 req/min) | 1-min delay | Forex, commodities |
| **Twelve Data** | REST API | Free (800 req/day) | Real-time | All macro |
| **FRED** (Federal Reserve) | REST API | Free | Daily | US10Y, DXY |
| **Dhan API** | MCX commodities already in universe | Free | Real-time | Crude, Gold, Silver |

### Implementation Plan

```
File: data/macro_fetcher.py

class MacroFetcher:
    def fetch_dxy() -> float:
    def fetch_us10y() -> float:
    def fetch_crude() -> float:
    def fetch_gold() -> float:
    
    def get_macro_context() -> dict:
        """Returns {"vix": 18.5, "dxy": 104.2, "us10y": 4.35, "crude_oil": 78.5, "gold": 2150.0}"""

Storage: Time-series Parquet (1-minute resolution)
Schedule: Every 60 seconds during market hours
Note: Crude and Gold are already being downloaded as MCX instruments.
      DXY and US10Y need new data sources.
```

---

## 11. Tier 7 — Sector Rotation & Relative Strength

### Impact: Populates `sector_rotation` context (hook defined, logic not yet wired)

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `sector_momentum_rank` | Rank of the stock's sector by 20-day momentum (1=strongest) | Daily |
| `relative_strength` | Stock's return vs Nifty 50 return over 20 days | Daily |

### How It Improves Strategies

- Stocks in top-3 momentum sectors get a confidence boost
- Stocks with relative strength > 1.0 (outperforming Nifty) get a boost
- Sector rotation signals predict which sectors will lead next

### Data Sources

This is **entirely derivable from existing OHLCV data**. No new external data needed.

### Implementation Plan

```
File: intelligence/sector_rotation.py

class SectorRotation:
    def compute_sector_momentum(sector_indices: dict[str, pd.DataFrame]) -> dict[str, float]:
        """20-day momentum for each sectoral index."""
        # Uses: Nifty IT, Nifty Bank, Nifty Pharma, etc. (already in universe.py)
        
    def compute_relative_strength(stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> float:
        """Stock 20-day return / Nifty 50 20-day return."""
        
    def get_sector_context(symbol: str) -> dict:
        """Returns {"sector_momentum_rank": 3, "relative_strength": 1.15}"""

Storage: Computed daily from existing Parquet files
Schedule: Once daily after market close
Note: Sectoral indices are already in universe.py (NIFTY_IT, NIFTY_BANK, etc.)
```

---

## 12. Tier 8 — Central Bank & Policy Signals

### Impact: Populates `central_bank` context (hook defined, logic not yet wired)

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `rate_decision_days` | Days until next RBI/Fed rate decision | Daily |
| `hawkish_dovish_score` | Sentiment of last policy statement (-1 to +1) | Per event |

### How It Improves Strategies

- Markets behave differently in the 3 days before a rate decision (lower volatility, then expansion)
- Hawkish statements → bearish for equities; dovish → bullish
- Rate decision proximity can be used to widen stops or reduce position size

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **RBI Calendar** | rbi.org.in (web scrape) | Free | Manual/daily | RBI MPC dates |
| **Fed Calendar** | federalreserve.gov | Free | Manual/daily | FOMC dates |
| **Economic Calendar APIs** | Investing.com, ForexFactory | Free | Daily | Global central banks |

### Implementation Plan

```
File: data/central_bank.py

class CentralBankCalendar:
    # Hardcoded calendar of known RBI MPC and FOMC dates (updated quarterly)
    RBI_MPC_DATES = ["2026-04-09", "2026-06-06", "2026-08-08", ...]
    FOMC_DATES = ["2026-03-19", "2026-05-07", "2026-06-18", ...]
    
    def days_to_next_decision(market: str = "india") -> int:
    def last_statement_score(market: str = "india") -> float:
        """FinBERT on the last policy statement text."""
    
    def get_central_bank_context() -> dict:
        """Returns {"rate_decision_days": 12, "hawkish_dovish_score": -0.3}"""

Storage: Static calendar file + SQLite for statement scores
Schedule: Updated manually when new dates are announced (quarterly)
```

---

## 13. Tier 9 — Geopolitical Risk

### Impact: Populates `geopolitical` context (hook defined, logic not yet wired)

### Data Points Required

| Field | Description | Update Frequency |
|-------|-------------|-----------------|
| `risk_score` | Composite geopolitical risk score (0-100) | Daily |
| `event_proximity_days` | Days until/since nearest major geopolitical event | Daily |

### Data Sources

| Source | API | Cost | Latency | Coverage |
|--------|-----|------|---------|----------|
| **Caldara-Iacoviello GPR Index** | matteoiacoviello.com | Free | Monthly | Global GPR |
| **GDELT Project** | REST API | Free | 15 minutes | Global events |
| **BlackRock Geopolitical Risk Dashboard** | Web scraping | Free | Weekly | Top 10 risks |
| **Custom NLP on news** | FinBERT on geopolitical keywords | Free | Real-time | Custom |

### Implementation Plan

```
File: data/geopolitical.py

class GeopoliticalRisk:
    KEYWORDS = ["war", "sanctions", "tariff", "nuclear", "invasion", "coup", ...]
    
    def compute_risk_score(news_articles: list[Article]) -> float:
        """Count and weight geopolitical keywords in recent news."""
        
    def get_geopolitical_context() -> dict:
        """Returns {"risk_score": 35.0, "event_proximity_days": 7}"""

Storage: SQLite table (date, risk_score, top_events)
Schedule: Daily
```

---

## 14. Implementation Roadmap

### Phase 1 — OHLCV Backtesting (CURRENT)

**Timeline**: Now
**Status**: All 40 strategies work with OHLCV data today.

| Task | Status |
|------|--------|
| Download 5-year 1m data for 536 instruments | IN PROGRESS |
| Run full backtest (40 math + 14 standard strategies) | NEXT |
| Analyse results, identify top strategy-instrument pairs | NEXT |

### Phase 2 — Low-Hanging Fruit Context (Week 1-2)

**Timeline**: After backtest completes
**Priority**: Tier 4 (VIX) → Tier 7 (Sector Rotation) → Tier 3 (FII/DII)

These three tiers are the easiest to implement and provide the highest value:

| Task | Effort | Value | Dependencies |
|------|--------|-------|-------------|
| **Tier 4: VIX fetcher** | 2 hours | High (risk filter) | India VIX already in universe.py |
| **Tier 7: Sector rotation** | 4 hours | High (no new data needed) | Sectoral indices already downloaded |
| **Tier 3: FII/DII scraper** | 1 day | Highest (+0.05 boost) | NSE website scraping |

Deliverables:
- `data/vix_fetcher.py` — reads India VIX from existing Parquet data
- `intelligence/sector_rotation.py` — computes sector momentum from existing data
- `data/institutional_flows.py` — daily NSE FII/DII scraper with 5-year backfill
- Wire all three into the `context` dict passed to advanced strategies

### Phase 3 — Options Chain Integration (Week 2-3)

**Timeline**: After Phase 2
**Priority**: Tier 2 (Options Chain)

| Task | Effort | Value | Dependencies |
|------|--------|-------|-------------|
| **Dhan options chain fetcher** | 1 day | Medium (+0.03 boost) | Dhan API access |
| **PCR computation** | 2 hours | Medium | Options chain data |
| **Max pain computation** | 4 hours | Medium | Options chain data |
| **IV skew computation** | 4 hours | Medium | Options chain data + Black-Scholes |
| **Historical backfill** | 2 days | High (for backtesting) | NSE archives or Sensibull |

Deliverables:
- `data/options_fetcher.py` — real-time option chain via Dhan API
- `data/options_analytics.py` — PCR, max pain, IV skew, IV term structure
- Historical PCR/IV data for backtesting (if available)

### Phase 4 — Macroeconomic Data (Week 3-4)

**Timeline**: After Phase 3
**Priority**: Tier 6 (Macro)

| Task | Effort | Value | Dependencies |
|------|--------|-------|-------------|
| **DXY fetcher** (Yahoo Finance) | 2 hours | Medium | yfinance |
| **US 10Y fetcher** (Yahoo Finance) | 2 hours | Medium | yfinance |
| **Crude/Gold** (already downloading) | 0 hours | — | Already in MCX universe |
| **Macro context builder** | 2 hours | Medium | All above |

Deliverables:
- `data/macro_fetcher.py` — DXY, US10Y from Yahoo Finance; crude/gold from existing MCX data
- Historical macro data backfill (5 years from Yahoo Finance)

### Phase 5 — Sentiment Pipeline (Week 4-6)

**Timeline**: After Phase 4
**Priority**: Tier 5 (Sentiment)

This is the most complex tier — requires NLP infrastructure.

| Task | Effort | Value | Dependencies |
|------|--------|-------|-------------|
| **News fetcher** (NewsAPI / Google News RSS) | 1 day | Medium | API key |
| **FinBERT model setup** | 1 day | Medium | HuggingFace, PyTorch |
| **Sentiment scoring pipeline** | 2 days | Medium (+0.02 boost) | News fetcher + FinBERT |
| **Social media fetcher** (Reddit/Twitter) | 2 days | Low | API keys |
| **Fear & Greed Index scraper** | 2 hours | Low | CNN website |

Deliverables:
- `data/sentiment_fetcher.py` — news and social media fetchers
- `data/sentiment_model.py` — FinBERT-based sentiment scorer
- Scheduled pipeline: fetch → score → store → context

### Phase 6 — Policy & Geopolitical (Week 6-8)

**Timeline**: After Phase 5
**Priority**: Tier 8 (Central Bank) → Tier 9 (Geopolitical)

| Task | Effort | Value | Dependencies |
|------|--------|-------|-------------|
| **Central bank calendar** | 2 hours | Low | Static data |
| **Policy statement scorer** | 4 hours | Low | FinBERT |
| **Geopolitical risk scorer** | 1 day | Low | News fetcher + keyword NLP |

Deliverables:
- `data/central_bank.py` — calendar + statement scoring
- `data/geopolitical.py` — keyword-based risk scoring

### Phase 7 — Alpaca US Market Integration (Parallel with Phase 2-3)

**Timeline**: Starts Week 1, runs in parallel
**Priority**: High (expands universe to S&P 500)

| Task | Effort | Dependencies |
|------|--------|-------------|
| **Alpaca API integration** | 1 day | API key + secret from user |
| **S&P 500 universe definition** | 2 hours | Wikipedia/Alpaca symbol list |
| **Data downloader (same Parquet pattern)** | 4 hours | Alpaca SDK |
| **Backtest on US data** | Runs automatically | Downloaded data |

Deliverables:
- `data/alpaca_fetcher.py` — Alpaca OHLCV fetcher with same cache/resume pattern
- `data/universe_us.py` — S&P 500 instruments
- `scripts/download_data_us.py` — bulk downloader for US data

---

## 15. Architecture: Data Pipeline Design

### Current Architecture (Tier 0 only)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Dhan API    │────▶│  Parquet     │────▶│  Strategy    │
│  (OHLCV)    │     │  Cache       │     │  Engine      │
└─────────────┘     └──────────────┘     └──────────────┘
```

### Target Architecture (All Tiers)

```
                    ┌─────────────────────────────────────────────────┐
                    │              DATA LAKE (Parquet + SQLite)        │
                    │                                                  │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
                    │  │ OHLCV    │  │ Options  │  │ Macro    │      │
                    │  │ 1m bars  │  │ Chain    │  │ DXY/10Y  │      │
                    │  └──────────┘  └──────────┘  └──────────┘      │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
                    │  │ FII/DII  │  │ VIX      │  │ Sector   │      │
                    │  │ Flows    │  │ Series   │  │ Momentum │      │
                    │  └──────────┘  └──────────┘  └──────────┘      │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
                    │  │ Sentiment│  │ Central  │  │ Geopolit │      │
                    │  │ Scores   │  │ Bank Cal │  │ Risk     │      │
                    │  └──────────┘  └──────────┘  └──────────┘      │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │              CONTEXT BUILDER                     │
                    │                                                  │
                    │  Reads all data sources for a given symbol +     │
                    │  timestamp and assembles the context dict:       │
                    │                                                  │
                    │  context = {                                     │
                    │    "options_chain": {"pcr": 1.3, ...},          │
                    │    "fii_dii_flows": {"fii_net": 1200, ...},     │
                    │    "sentiment":     {"news_score": 0.2, ...},   │
                    │    "macro":         {"vix": 18, ...},           │
                    │    "sector_rotation": {"rank": 2, ...},         │
                    │    "central_bank":  {"days": 15, ...},          │
                    │    "geopolitical":  {"risk": 25, ...},          │
                    │  }                                               │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  OHLCV      │────▶│              STRATEGY ENGINE                     │
│  DataFrame  │     │                                                  │
└─────────────┘     │  for strategy in get_all_mathematical_strategies │
                    │      signals = strategy.generate_signals(        │
                    │          df, symbol, context=context             │
                    │      )                                           │
                    └─────────────────────┬────────────────────────────┘
                                          │
                                          ▼
                    ┌──────────────────────────────────────────────────┐
                    │              BACKTEST / LIVE ENGINE               │
                    │                                                  │
                    │  Processes signals → position sizing → execution │
                    │  → P&L tracking → risk management                │
                    └──────────────────────────────────────────────────┘
```

### Context Builder Implementation

```
File: data/context_builder.py

class ContextBuilder:
    """
    Assembles the multi-factor context dict for a given symbol and timestamp.
    Each data source is optional — missing sources return empty sub-dicts,
    which the strategies handle gracefully (0.0 confidence adjustment).
    """
    
    def __init__(self):
        self.options = OptionsFetcher()      # Tier 2
        self.flows = InstitutionalFlows()     # Tier 3
        self.vix = VIXFetcher()               # Tier 4
        self.sentiment = SentimentScorer()    # Tier 5
        self.macro = MacroFetcher()           # Tier 6
        self.sector = SectorRotation()        # Tier 7
        self.central_bank = CentralBank()     # Tier 8
        self.geopolitical = GeopoliticalRisk() # Tier 9
    
    def build(self, symbol: str, timestamp: pd.Timestamp) -> dict:
        context = {}
        
        # Each fetcher is wrapped in try/except — failure = empty dict
        try: context["options_chain"] = self.options.get_context(symbol)
        except: pass
        
        try: context["fii_dii_flows"] = self.flows.get_context()
        except: pass
        
        try: context["macro"] = self.macro.get_context()
        except: pass
        
        # ... etc for all tiers
        
        return context
```

---

## Summary: What Works Today vs What Needs Building

| Category | Status | Strategies Affected | Action |
|----------|--------|-------------------|--------|
| OHLCV (O, H, L, C, V) | **WORKING** | All 40 | None — downloading now |
| All internal computations | **WORKING** | All 40 | None — computed in code |
| Multi-factor context hooks | **CODE EXISTS** | 25 advanced | Need data feeds (Tiers 2-9) |
| VIX context | **EASY WIN** | 25 advanced | Load India VIX from existing data |
| Sector rotation | **EASY WIN** | 25 advanced | Compute from existing sectoral indices |
| FII/DII flows | **1 day work** | 25 advanced | Build NSE scraper |
| Options chain | **2-3 days work** | 25 advanced | Build Dhan options fetcher |
| Macro indicators | **1 day work** | 25 advanced | Yahoo Finance + existing MCX data |
| Sentiment | **1-2 weeks work** | 25 advanced | FinBERT + news/social fetchers |
| Central bank calendar | **2 hours work** | 25 advanced | Static calendar file |
| Geopolitical risk | **1-2 days work** | 25 advanced | Keyword NLP on news |

**The critical insight**: All 40 strategies are fully functional today with OHLCV data alone. The multi-factor context is a confidence booster (max ±0.15), not a requirement. The backtest should run first with OHLCV-only, and context feeds should be added incrementally to measure their marginal improvement on signal quality.
