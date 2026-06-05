from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RESEARCH_START = "2018-06-01"
RESEARCH_END = "2026-04-30 23:59:59+00:00"
SECOND_SAMPLE_START = "2026-04-01"
SECOND_SAMPLE_END = "2026-04-30 23:59:59+00:00"

RESEARCH_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "BNB",
    "XRP",
    "ADA",
    "TRX",
    "LTC",
    "SPX",
    "NASDAQ100",
    "UDXUSD",
    "EURUSD",
    "USDJPY",
    "USDCHF",
    "GOLD",
    "BRENT",
)

CRYPTO_ASSETS: tuple[str, ...] = ("BTC", "ETH", "BNB", "XRP", "ADA", "TRX", "LTC")
EQUITY_ASSETS: tuple[str, ...] = ("SPX", "NASDAQ100")
FX_ASSETS: tuple[str, ...] = ("UDXUSD", "EURUSD", "USDJPY", "USDCHF")
COMMODITY_ASSETS: tuple[str, ...] = ("GOLD", "BRENT")

ASSET_BLOCKS: dict[str, str] = {
    **{asset: "crypto" for asset in CRYPTO_ASSETS},
    **{asset: "equity_index" for asset in EQUITY_ASSETS},
    **{asset: "fx_dollar" for asset in FX_ASSETS},
    **{asset: "commodity" for asset in COMMODITY_ASSETS},
}

BLOCK_ORDER: tuple[str, ...] = ("crypto", "equity_index", "fx_dollar", "commodity")

SOURCE_FREQUENCY_BY_SYMBOL: dict[str, str] = {}

PRICE_SANITY_RANGES: dict[str, tuple[float | None, float | None]] = {
    # HistData UDXUSD contains a wrong high-scale block before 2018-12-17
    # that looks like an equity index, not Dollar Index quotes.
    "UDXUSD": (50.0, 150.0),
    "USDC": (0.85, 1.15),
    "TUSD": (0.85, 1.15),
    "BUSD": (0.85, 1.15),
    "FDUSD": (0.85, 1.15),
}

PRICE_DISCOVERY_PAIRS: tuple[tuple[str, str], ...] = (
    ("BTC", "SPX"),
    ("BTC", "NASDAQ100"),
    ("ETH", "SPX"),
    ("ETH", "NASDAQ100"),
    ("BTC", "UDXUSD"),
    ("ETH", "UDXUSD"),
    ("BTC", "GOLD"),
    ("BTC", "BRENT"),
)


@dataclass(frozen=True)
class FrequencyConfig:
    name: str
    pandas_rule: str
    stale_limit_periods: int
    horizon: int
    max_lags: int
    max_obs: int | None
    rolling_window: str | None = None
    rolling_step: str | None = None
    rolling_max_obs: int | None = None
    build_full_period: bool = True


FREQUENCY_CONFIGS: dict[str, FrequencyConfig] = {
    "1s": FrequencyConfig(
        name="1s",
        pandas_rule="1s",
        stale_limit_periods=60,
        horizon=60,
        max_lags=2,
        max_obs=30_000,
        build_full_period=False,
    ),
    "1min": FrequencyConfig(
        name="1min",
        pandas_rule="1min",
        stale_limit_periods=15,
        horizon=60,
        max_lags=4,
        max_obs=30_000,
        build_full_period=True,
    ),
    "1h": FrequencyConfig(
        name="1h",
        pandas_rule="1h",
        stale_limit_periods=6,
        horizon=24,
        max_lags=4,
        max_obs=20_000,
        rolling_window="180D",
        rolling_step="60D",
        rolling_max_obs=5_000,
        build_full_period=True,
    ),
    "1d": FrequencyConfig(
        name="1d",
        pandas_rule="1D",
        stale_limit_periods=3,
        horizon=20,
        max_lags=4,
        max_obs=None,
        rolling_window="365D",
        rolling_step="60D",
        rolling_max_obs=None,
        build_full_period=True,
    ),
}

PERIOD_SPLITS: dict[str, tuple[str, str]] = {
    "full": (RESEARCH_START, RESEARCH_END),
    "pre_covid": (RESEARCH_START, "2020-02-29 23:59:59+00:00"),
    "covid_and_after": ("2020-03-01", RESEARCH_END),
}

DEFAULT_DATA_ROOT = Path("data") / "second"
DEFAULT_ARTIFACT_ROOT = Path("artifacts")
