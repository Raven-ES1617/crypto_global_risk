"""Data parsers and downloaders for second-level market data."""

from parsers.assets import (
    AssetSpec,
    DEFAULT_CRYPTO_ASSETS,
    DEFAULT_EQUITY_ASSETS,
    DEFAULT_HISTDATA_ASSETS,
    DEFAULT_HISTDATA_INDEX_ASSETS,
    DEFAULT_HISTDATA_MACRO_ASSETS,
    DEFAULT_MACRO_ASSETS,
    DEFAULT_STABLECOIN_ASSETS,
    get_default_assets,
)
from parsers.binance import (
    fetch_binance_klines,
    fetch_binance_monthly_zip_klines,
    read_binance_monthly_zip,
    save_binance_klines,
    save_binance_monthly_zip_to_parquet,
)
from parsers.binance_us import (
    fetch_binance_us_monthly_zip_klines,
    save_binance_us_monthly_zip_to_parquet,
)
from parsers.histdata import (
    download_histdata_tick_last_archive,
    read_histdata_tick_last_zip,
    save_histdata_tick_last_to_parquet,
)

__all__ = [
    "AssetSpec",
    "DEFAULT_CRYPTO_ASSETS",
    "DEFAULT_EQUITY_ASSETS",
    "DEFAULT_HISTDATA_ASSETS",
    "DEFAULT_HISTDATA_INDEX_ASSETS",
    "DEFAULT_HISTDATA_MACRO_ASSETS",
    "DEFAULT_MACRO_ASSETS",
    "DEFAULT_STABLECOIN_ASSETS",
    "download_histdata_tick_last_archive",
    "fetch_binance_klines",
    "fetch_binance_monthly_zip_klines",
    "fetch_binance_us_monthly_zip_klines",
    "read_binance_monthly_zip",
    "read_histdata_tick_last_zip",
    "get_default_assets",
    "save_binance_klines",
    "save_binance_monthly_zip_to_parquet",
    "save_binance_us_monthly_zip_to_parquet",
    "save_histdata_tick_last_to_parquet",
]
