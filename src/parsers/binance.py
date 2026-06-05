from __future__ import annotations

import json
import time
import zipfile
from calendar import monthrange
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from parsers.storage import (
    DEFAULT_SECOND_DATA_ROOT,
    DataFile,
    write_asset_parquet,
)

BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_S3_BUCKET_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_ARCHIVE_BASE = f"{BINANCE_S3_BUCKET_BASE}/data/spot"
BINANCE_MONTHLY_DATA_BASE = f"{BINANCE_ARCHIVE_BASE}/monthly/klines"
BINANCE_DAILY_DATA_BASE = f"{BINANCE_ARCHIVE_BASE}/daily/klines"
BINANCE_DATA_BASE = BINANCE_MONTHLY_DATA_BASE
S3_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
KLINE_COLUMNS = [
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
INTERVAL_MS = {"1s": 1_000}
UrlOpener = Callable[[str], bytes]


def fetch_binance_klines(
    symbol: str,
    *,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    interval: str = "1s",
    limit: int = 1000,
    base_url: str = BINANCE_REST_BASE,
    opener: UrlOpener | None = None,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")
    if limit < 1 or limit > 1000:
        raise ValueError("Binance kline limit must be between 1 and 1000")

    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end)
    if end_ms < start_ms:
        raise ValueError("end must be greater than or equal to start")

    cursor = start_ms
    rows: list[list[object]] = []
    step_ms = INTERVAL_MS[interval]

    while cursor <= end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        url = f"{base_url.rstrip('/')}/api/v3/klines?{urlencode(params)}"
        payload = _read_url(url, opener=opener)
        batch = json.loads(payload.decode("utf-8"))
        if not batch:
            break
        if isinstance(batch, dict) and "code" in batch:
            raise RuntimeError(f"Binance error for {symbol}: {batch}")

        filtered = [row for row in batch if int(row[0]) <= end_ms]
        rows.extend(filtered)

        last_open = int(batch[-1][0])
        next_cursor = last_open + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if last_open >= end_ms:
            break

    frame = binance_klines_to_frame(rows)
    if frame.empty:
        return frame
    return frame[(frame.index >= pd.to_datetime(start, utc=True)) & (frame.index <= pd.to_datetime(end, utc=True))]


def save_binance_klines(
    symbol: str,
    *,
    start: str | datetime | pd.Timestamp,
    end: str | datetime | pd.Timestamp,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    interval: str = "1s",
    asset_symbol: str | None = None,
    opener: UrlOpener | None = None,
) -> DataFile:
    frame = fetch_binance_klines(symbol, start=start, end=end, interval=interval, opener=opener)
    output_symbol = asset_symbol or symbol.upper()
    return write_asset_parquet(
        frame,
        root=root,
        provider="binance",
        symbol=output_symbol,
        source_symbol=symbol.upper(),
        suffix=f"{interval}_{_stamp(start)}_{_stamp(end)}",
    )


def fetch_binance_monthly_zip_klines(
    symbol: str,
    *,
    year: int,
    month: int,
    interval: str = "1s",
    base_url: str = BINANCE_MONTHLY_DATA_BASE,
    opener: UrlOpener | None = None,
) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{year}-{month:02d}.zip"
    payload = _read_url(url, opener=opener)
    return read_binance_zip_klines(payload)


def fetch_binance_daily_zip_klines(
    symbol: str,
    *,
    date: str | datetime | pd.Timestamp,
    interval: str = "1s",
    base_url: str = BINANCE_DAILY_DATA_BASE,
    opener: UrlOpener | None = None,
) -> pd.DataFrame:
    date_label = pd.to_datetime(date, utc=True).strftime("%Y-%m-%d")
    url = f"{base_url.rstrip('/')}/{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{date_label}.zip"
    payload = _read_url(url, opener=opener)
    return read_binance_zip_klines(payload)


def fetch_binance_month_from_daily_zip_klines(
    symbol: str,
    *,
    year: int,
    month: int,
    interval: str = "1s",
    opener: UrlOpener | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    days = _month_days(year, month)
    missing_days: list[str] = []
    available_days = None if opener is not None else list_binance_daily_zip_dates(symbol, year=year, month=month, interval=interval)
    if available_days is not None:
        missing_days = [day for day in days if day not in available_days]
        if require_complete and missing_days:
            sample = ", ".join(missing_days[:5])
            extra = "" if len(missing_days) <= 5 else f", ... and {len(missing_days) - 5} more"
            raise RuntimeError(
                f"missing daily Binance archives for {symbol.upper()} {year}-{month:02d}: {sample}{extra}"
            )
        days = [day for day in days if day in available_days]

    for day in days:
        try:
            frames.append(fetch_binance_daily_zip_klines(symbol, date=day, interval=interval, opener=opener))
        except Exception as exc:
            if _is_not_found_error(exc):
                missing_days.append(day)
                continue
            raise

    if require_complete and missing_days:
        sample = ", ".join(missing_days[:5])
        extra = "" if len(missing_days) <= 5 else f", ... and {len(missing_days) - 5} more"
        raise RuntimeError(
            f"missing daily Binance archives for {symbol.upper()} {year}-{month:02d}: {sample}{extra}"
        )
    if not frames:
        raise RuntimeError(f"no daily Binance archives for {symbol.upper()} {year}-{month:02d}")

    frame = pd.concat(frames).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def list_binance_daily_zip_dates(
    symbol: str,
    *,
    year: int,
    month: int,
    interval: str = "1s",
) -> set[str] | None:
    prefix = f"data/spot/daily/klines/{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{year}-{month:02d}"
    try:
        keys = list_binance_s3_keys(prefix)
    except Exception:
        return None
    dates: set[str] = set()
    stem = f"{symbol.upper()}-{interval}-"
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        if not name.startswith(stem) or not name.endswith(".zip"):
            continue
        dates.add(name[len(stem) : -len(".zip")])
    return dates


def list_binance_s3_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    marker: str | None = None
    while True:
        params = {"prefix": prefix, "max-keys": "1000"}
        if marker is not None:
            params["marker"] = marker
        payload = _read_url(f"{BINANCE_S3_BUCKET_BASE}?{urlencode(params)}", opener=None)
        root = ET.fromstring(payload)
        batch = [node.text for node in root.findall("s3:Contents/s3:Key", S3_XML_NS) if node.text]
        keys.extend(batch)
        is_truncated = (root.findtext("s3:IsTruncated", namespaces=S3_XML_NS) or "").lower() == "true"
        if not is_truncated or not batch:
            break
        marker = batch[-1]
    return keys


def read_binance_monthly_zip(payload: bytes | Path | str) -> pd.DataFrame:
    return read_binance_zip_klines(payload)


def read_binance_zip_klines(payload: bytes | Path | str) -> pd.DataFrame:
    if isinstance(payload, bytes):
        archive = zipfile.ZipFile(BytesIO(payload))
    else:
        archive = zipfile.ZipFile(payload)

    with archive:
        csv_name = next((name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))), None)
        if csv_name is None:
            raise ValueError("Binance monthly archive does not contain a csv/txt file")
        with archive.open(csv_name) as file:
            raw = pd.read_csv(file, header=None, low_memory=False)

    rows = raw.values.tolist()
    return binance_klines_to_frame(rows)


def save_binance_monthly_zip_to_parquet(
    symbol: str,
    *,
    year: int,
    month: int,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    interval: str = "1s",
    asset_symbol: str | None = None,
    opener: UrlOpener | None = None,
) -> DataFile:
    frame = fetch_binance_monthly_zip_klines(symbol, year=year, month=month, interval=interval, opener=opener)
    output_symbol = asset_symbol or symbol.upper()
    return write_asset_parquet(
        frame,
        root=root,
        provider="binance",
        symbol=output_symbol,
        source_symbol=symbol.upper(),
        suffix=f"{interval}_{year}{month:02d}",
    )


def save_binance_monthly_or_daily_to_parquet(
    symbol: str,
    *,
    year: int,
    month: int,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    interval: str = "1s",
    asset_symbol: str | None = None,
    opener: UrlOpener | None = None,
    require_complete_daily: bool = True,
) -> DataFile:
    try:
        return save_binance_monthly_zip_to_parquet(
            symbol,
            year=year,
            month=month,
            root=root,
            interval=interval,
            asset_symbol=asset_symbol,
            opener=opener,
        )
    except Exception as monthly_error:
        if not _is_not_found_error(monthly_error):
            raise

    frame = fetch_binance_month_from_daily_zip_klines(
        symbol,
        year=year,
        month=month,
        interval=interval,
        opener=opener,
        require_complete=require_complete_daily,
    )
    output_symbol = asset_symbol or symbol.upper()
    return write_asset_parquet(
        frame,
        root=root,
        provider="binance",
        symbol=output_symbol,
        source_symbol=symbol.upper(),
        suffix=f"{interval}_{year}{month:02d}",
    )


def binance_klines_to_frame(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        return _empty_kline_frame()

    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS[: len(rows[0])])
    open_time = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.loc[open_time.notna()].copy()
    if frame.empty:
        return _empty_kline_frame()
    open_time_int = open_time.loc[frame.index].astype("int64")
    frame["timestamp"] = pd.to_datetime(open_time_int, unit=_timestamp_unit(open_time_int), utc=True)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    keep = [col for col in numeric_cols if col in frame]
    frame = frame.set_index("timestamp")[keep].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _read_url(url: str, *, opener: UrlOpener | None) -> bytes:
    if opener is not None:
        return opener(url)
    last_error: BaseException | None = None
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": "crypto-risk-metrics/0.1"})
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError:
            raise
        except (ConnectionError, OSError, TimeoutError, URLError) as exc:
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to read url: {url}") from last_error


def _to_utc_ms(value: str | datetime | pd.Timestamp) -> int:
    ts = pd.to_datetime(value, utc=True)
    return int(ts.timestamp() * 1000)


def _stamp(value: str | datetime | pd.Timestamp) -> str:
    ts = pd.to_datetime(value, utc=True)
    return ts.strftime("%Y%m%dT%H%M%S")


def _month_days(year: int, month: int) -> list[str]:
    days = monthrange(year, month)[1]
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, days + 1)]


def _is_not_found_error(exc: BaseException) -> bool:
    return isinstance(exc, HTTPError) and exc.code == 404


def _empty_kline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
    ).rename_axis("timestamp")


def _timestamp_unit(values: pd.Series) -> str:
    # Binance archives are historically millisecond-based, but newer 1s archives
    # can store open_time in microseconds.
    return "us" if values.max() >= 10_000_000_000_000 else "ms"


def utc_now_floor_second() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0)
