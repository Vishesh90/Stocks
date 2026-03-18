"""
config/settings.py — Centralised configuration

INTENT:
    Single source of truth for all environment-driven settings.
    Every module imports from here — never reads .env directly.

IMPACT:
    Changing a risk parameter here affects the entire system immediately.
    Misconfigured values fail fast at startup, not mid-trade.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Broker — Dhan
    dhan_client_id: str = Field(default="", env="DHAN_CLIENT_ID")
    dhan_access_token: str = Field(default="", env="DHAN_ACCESS_TOKEN")

    # Broker — Groww
    groww_api_key: str = Field(default="", env="GROWW_API_KEY")
    groww_api_secret: str = Field(default="", env="GROWW_API_SECRET")

    # Risk
    daily_loss_limit_inr: float = Field(default=500.0, env="DAILY_LOSS_LIMIT_INR")
    max_trades_per_day: int = Field(default=10, env="MAX_TRADES_PER_DAY")
    capital_inr: float = Field(default=20000.0, env="CAPITAL_INR")
    risk_per_trade_pct: float = Field(default=0.015, env="RISK_PER_TRADE_PCT")

    # Execution
    execution_mode: str = Field(default="paper", env="EXECUTION_MODE")  # paper | live

    # Paths
    data_cache_dir: Path = Field(default=Path("data/cache"), env="DATA_CACHE_DIR")
    raw_data_dir: Path = Field(default=Path("data/raw"), env="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("data/processed"), env="PROCESSED_DATA_DIR")
    reports_dir: Path = Field(default=Path("reports"), env="REPORTS_DIR")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def risk_per_trade_inr(self) -> float:
        return self.capital_inr * self.risk_per_trade_pct

    @property
    def is_paper(self) -> bool:
        return self.execution_mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.execution_mode == "live"


settings = Settings()
