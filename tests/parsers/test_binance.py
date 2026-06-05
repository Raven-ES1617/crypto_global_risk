from __future__ import annotations

import json
import zipfile
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from parsers.binance import (
    binance_klines_to_frame,
    fetch_binance_daily_zip_klines,
    fetch_binance_klines,
    fetch_binance_month_from_daily_zip_klines,
    list_binance_daily_zip_dates,
    read_binance_monthly_zip,
    save_binance_klines,
    save_binance_monthly_or_daily_to_parquet,
    save_binance_monthly_zip_to_parquet,
)
from parsers.storage import read_asset_parquet


def _row(ts_ms: int, close: str = "100.0") -> list[object]:
    return [
        ts_ms,
        "99.0",
        "101.0",
        "98.0",
        close,
        "1.5",
        ts_ms + 999,
        "150.0",
        10,
        "0.7",
        "70.0",
        "0",
    ]


def test_binance_klines_to_frame() -> None:
    frame = binance_klines_to_frame([_row(1577836800000, "101.5")])

    assert frame.index[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame.loc[pd.Timestamp("2020-01-01T00:00:00Z"), "close"] == 101.5
    assert frame.loc[pd.Timestamp("2020-01-01T00:00:00Z"), "number_of_trades"] == 10


def test_binance_klines_to_frame_accepts_microsecond_archive_timestamps() -> None:
    frame = binance_klines_to_frame([_row(1735689600000000, "101.5")])

    assert frame.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert frame["close"].iloc[0] == 101.5


def test_binance_klines_to_frame_drops_archive_header() -> None:
    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    frame = binance_klines_to_frame([header, _row(1577836800000, "101.5")])

    assert len(frame) == 1
    assert frame["close"].iloc[0] == 101.5


def test_fetch_binance_klines_paginates_with_fake_opener() -> None:
    calls: list[str] = []

    def opener(url: str) -> bytes:
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        start = int(query["startTime"][0])
        end = int(query["endTime"][0])
        rows = [_row(start), _row(start + 1000)]
        rows = [row for row in rows if row[0] <= end]
        return json.dumps(rows).encode("utf-8")

    frame = fetch_binance_klines(
        "BTCUSDT",
        start="2020-01-01T00:00:00Z",
        end="2020-01-01T00:00:03Z",
        limit=2,
        opener=opener,
    )

    assert len(calls) == 2
    assert len(frame) == 4
    assert frame.index.min() == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame.index.max() == pd.Timestamp("2020-01-01T00:00:03Z")


def test_save_binance_klines_writes_asset_file(tmp_path) -> None:
    def opener(url: str) -> bytes:
        query = parse_qs(urlparse(url).query)
        start = int(query["startTime"][0])
        return json.dumps([_row(start)]).encode("utf-8")

    saved = save_binance_klines(
        "ETHUSDT",
        start="2020-01-01T00:00:00Z",
        end="2020-01-01T00:00:00Z",
        root=tmp_path,
        asset_symbol="ETH",
        opener=opener,
    )

    assert saved.rows == 1
    assert saved.path.endswith(".parquet")
    assert "binance" in saved.path
    assert "ETH" in saved.path
    assert saved.source_symbol == "ETHUSDT"
    loaded = read_asset_parquet(saved.path)
    assert loaded["close"].iloc[0] == 100.0


def test_read_binance_monthly_zip_from_bytes() -> None:
    payload = _zip_payload([_row(1577836800000, "101.5")])
    frame = read_binance_monthly_zip(payload)

    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert frame["close"].iloc[0] == 101.5


def test_save_binance_monthly_zip_to_parquet(tmp_path) -> None:
    payload = _zip_payload([_row(1577836800000, "101.5")])

    def opener(url: str) -> bytes:
        assert "BTCUSDT-1s-2020-01.zip" in url
        return payload

    saved = save_binance_monthly_zip_to_parquet(
        "BTCUSDT",
        year=2020,
        month=1,
        root=tmp_path,
        asset_symbol="BTC",
        opener=opener,
    )
    loaded = read_asset_parquet(saved.path)

    assert saved.path.endswith(".parquet")
    assert saved.rows == 1
    assert loaded["close"].iloc[0] == 101.5


def test_fetch_binance_daily_zip_klines_uses_daily_archive_url() -> None:
    payload = _zip_payload([_row(1577923200000, "101.5")], name="BTCUSDT-1s-2020-01-02.csv")

    def opener(url: str) -> bytes:
        assert "data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2020-01-02.zip" in url
        return payload

    frame = fetch_binance_daily_zip_klines("BTCUSDT", date="2020-01-02", opener=opener)

    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2020-01-02T00:00:00Z")


def test_save_binance_monthly_or_daily_to_parquet_falls_back_to_daily_archives(tmp_path) -> None:
    calls: list[str] = []

    def opener(url: str) -> bytes:
        calls.append(url)
        if "/monthly/" in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        date = url.split("BTCUSDT-1s-", 1)[1].removesuffix(".zip")
        ts_ms = int(pd.Timestamp(f"{date}T00:00:00Z").timestamp() * 1000)
        return _zip_payload([_row(ts_ms, "101.5")], name=f"BTCUSDT-1s-{date}.csv")

    saved = save_binance_monthly_or_daily_to_parquet(
        "BTCUSDT",
        year=2020,
        month=1,
        root=tmp_path,
        asset_symbol="BTC",
        opener=opener,
    )
    loaded = read_asset_parquet(saved.path)

    assert any("/monthly/" in url for url in calls)
    assert sum("/daily/" in url for url in calls) == 31
    assert saved.rows == 31
    assert loaded.index.min() == pd.Timestamp("2020-01-01T00:00:00Z")
    assert loaded.index.max() == pd.Timestamp("2020-01-31T00:00:00Z")


def test_fetch_binance_month_from_daily_zip_requires_complete_month() -> None:
    def opener(url: str) -> bytes:
        if "2020-01-01" not in url:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return _zip_payload([_row(1577836800000)], name="BTCUSDT-1s-2020-01-01.csv")

    with pytest.raises(RuntimeError, match="missing daily Binance archives"):
        fetch_binance_month_from_daily_zip_klines("BTCUSDT", year=2020, month=1, opener=opener)


def test_list_binance_daily_zip_dates_parses_s3_listing(monkeypatch) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents>
    <Key>data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2020-01-01.zip</Key>
  </Contents>
  <Contents>
    <Key>data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2020-01-01.zip.CHECKSUM</Key>
  </Contents>
  <Contents>
    <Key>data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2020-01-02.zip</Key>
  </Contents>
</ListBucketResult>
""".encode()

    def fake_read_url(url: str, *, opener) -> bytes:
        assert "prefix=data%2Fspot%2Fdaily%2Fklines%2FBTCUSDT%2F1s%2FBTCUSDT-1s-2020-01" in url
        assert opener is None
        return xml

    monkeypatch.setattr("parsers.binance._read_url", fake_read_url)

    assert list_binance_daily_zip_dates("BTCUSDT", year=2020, month=1) == {"2020-01-01", "2020-01-02"}


def _zip_payload(rows: list[list[object]], *, name: str = "BTCUSDT-1s-2020-01.csv") -> bytes:
    csv = "\n".join(",".join(str(value) for value in row) for row in rows)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv)
    return buffer.getvalue()
