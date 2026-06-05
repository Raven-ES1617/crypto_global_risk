from __future__ import annotations

from pathlib import Path

from parsers.binance import UrlOpener, fetch_binance_monthly_zip_klines
from parsers.storage import DEFAULT_SECOND_DATA_ROOT, DataFile, write_asset_parquet


BINANCE_US_ARCHIVE_BASE = "https://data.binance.us/public_data/spot"
BINANCE_US_MONTHLY_DATA_BASE = f"{BINANCE_US_ARCHIVE_BASE}/monthly/klines"


def fetch_binance_us_monthly_zip_klines(
    symbol: str,
    *,
    year: int,
    month: int,
    interval: str = "1m",
    opener: UrlOpener | None = None,
):
    return fetch_binance_monthly_zip_klines(
        symbol,
        year=year,
        month=month,
        interval=interval,
        base_url=BINANCE_US_MONTHLY_DATA_BASE,
        opener=opener,
    )


def save_binance_us_monthly_zip_to_parquet(
    symbol: str,
    *,
    year: int,
    month: int,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    interval: str = "1m",
    asset_symbol: str | None = None,
    opener: UrlOpener | None = None,
) -> DataFile:
    frame = fetch_binance_us_monthly_zip_klines(
        symbol,
        year=year,
        month=month,
        interval=interval,
        opener=opener,
    )
    output_symbol = asset_symbol or symbol.upper()
    return write_asset_parquet(
        frame,
        root=root,
        provider="binance_us",
        symbol=output_symbol,
        source_symbol=symbol.upper(),
        suffix=f"{interval}_{year}{month:02d}",
    )
