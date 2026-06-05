from __future__ import annotations

import zipfile
from io import BytesIO

import pandas as pd

from parsers.binance_us import (
    BINANCE_US_MONTHLY_DATA_BASE,
    fetch_binance_us_monthly_zip_klines,
    save_binance_us_monthly_zip_to_parquet,
)
from parsers.storage import read_asset_parquet


def test_fetch_binance_us_monthly_zip_uses_us_archive_url() -> None:
    def opener(url: str) -> bytes:
        assert url == f"{BINANCE_US_MONTHLY_DATA_BASE}/USDTUSD/1m/USDTUSD-1m-2020-01.zip"
        return _zip_payload()

    frame = fetch_binance_us_monthly_zip_klines("USDTUSD", year=2020, month=1, interval="1m", opener=opener)

    assert frame.index[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame["close"].iloc[0] == 1.001


def test_save_binance_us_monthly_zip_to_parquet_uses_provider(tmp_path) -> None:
    saved = save_binance_us_monthly_zip_to_parquet(
        "USDTUSD",
        year=2020,
        month=1,
        root=tmp_path,
        interval="1m",
        asset_symbol="USDT",
        opener=lambda _url: _zip_payload(),
    )
    loaded = read_asset_parquet(saved.path)

    assert saved.provider == "binance_us"
    assert saved.symbol == "USDT"
    assert saved.source_symbol == "USDTUSD"
    assert loaded["close"].iloc[0] == 1.001


def _zip_payload() -> bytes:
    row = [
        1577836800000,
        "1.0000",
        "1.0020",
        "0.9990",
        "1.0010",
        "10.0",
        1577836859999,
        "10.01",
        3,
        "5.0",
        "5.01",
        "0",
    ]
    csv = ",".join(str(value) for value in row)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("USDTUSD-1m-2020-01.csv", csv)
    return buffer.getvalue()
