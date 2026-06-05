from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from parsers.storage import DEFAULT_SECOND_DATA_ROOT, DataFile, asset_dir, safe_symbol, write_asset_parquet

EST_NO_DST = timezone(timedelta(hours=-5))
HISTDATA_GET_URL = "https://www.histdata.com/get.php"
HISTDATA_TICK_LAST_REFERER = (
    "https://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-last-quotes"
)
HistDataDownloader = Callable[[str, int, int, Path], Path]


def download_histdata_tick_last_archive(
    pair: str,
    *,
    year: int,
    month: int,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    asset_symbol: str | None = None,
    verify_ssl: bool = True,
    downloader: HistDataDownloader | None = None,
) -> Path:
    output_symbol = asset_symbol or pair.upper()
    target_dir = asset_dir(Path(root), "histdata_raw", output_symbol)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"DAT_NT_{pair.upper()}_T_LAST_{year}{month:02d}.zip"
    if target.exists():
        return target

    if downloader is not None:
        downloaded = downloader(pair, year, month, target_dir)
        return Path(downloaded)

    return _download_histdata_tick_last_direct(pair, year, month, target_dir, verify_ssl=verify_ssl)


def read_histdata_tick_last_zip(
    payload: bytes | Path | str,
    *,
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if isinstance(payload, bytes):
        archive = zipfile.ZipFile(io.BytesIO(payload))
    else:
        archive = zipfile.ZipFile(payload)

    with archive:
        name = next((item for item in archive.namelist() if item.lower().endswith((".txt", ".csv"))), None)
        if name is None:
            raise ValueError("HistData archive does not contain a csv/txt file")
        raw = archive.read(name).decode("utf-8", errors="replace")

    ticks = pd.read_csv(
        io.StringIO(raw),
        sep=";",
        header=None,
        names=["dt", "price", "volume"],
        usecols=[0, 1, 2],
    )
    ticks["datetime"] = pd.to_datetime(ticks["dt"], format="%Y%m%d %H%M%S", errors="coerce")
    ticks["price"] = pd.to_numeric(ticks["price"], errors="coerce")
    ticks["volume"] = pd.to_numeric(ticks["volume"], errors="coerce").fillna(0.0)
    ticks = ticks.dropna(subset=["datetime", "price"])
    if ticks.empty:
        return _empty_frame()

    ticks["timestamp"] = ticks["datetime"].dt.tz_localize(EST_NO_DST).dt.tz_convert("UTC").dt.floor("s")
    if start is not None:
        ticks = ticks[ticks["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        ticks = ticks[ticks["timestamp"] <= pd.to_datetime(end, utc=True)]
    if ticks.empty:
        return _empty_frame()

    grouped = ticks.groupby("timestamp", sort=True)
    frame = grouped["price"].agg(open="first", high="max", low="min", close="last")
    frame["volume"] = grouped["volume"].sum()
    frame["tick_count"] = grouped["price"].size()
    frame.index.name = "timestamp"
    return frame


def save_histdata_tick_last_to_parquet(
    pair: str,
    *,
    year: int,
    month: int,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    asset_symbol: str | None = None,
    start: str | datetime | pd.Timestamp | None = None,
    end: str | datetime | pd.Timestamp | None = None,
    verify_ssl: bool = True,
    downloader: HistDataDownloader | None = None,
) -> DataFile:
    output_symbol = asset_symbol or pair.upper()
    archive = download_histdata_tick_last_archive(
        pair,
        year=year,
        month=month,
        root=root,
        asset_symbol=output_symbol,
        verify_ssl=verify_ssl,
        downloader=downloader,
    )
    frame = read_histdata_tick_last_zip(archive, start=start, end=end)
    return write_asset_parquet(
        frame,
        root=root,
        provider="histdata",
        symbol=output_symbol,
        source_symbol=pair.upper(),
        suffix=f"1s_{year}{month:02d}",
    )


def histdata_monthly_parquet_path(data_root: Path | str, symbol: str, year: int, month: int) -> Path:
    return asset_dir(Path(data_root), "histdata", symbol) / f"{safe_symbol(symbol)}_1s_{year}{month:02d}.parquet"


def _download_histdata_tick_last_direct(
    pair: str,
    year: int,
    month: int,
    output_directory: Path,
    *,
    verify_ssl: bool,
) -> Path:
    if not verify_ssl:
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    pair_lower = pair.lower()
    referer = f"{HISTDATA_TICK_LAST_REFERER}/{pair_lower}/{year}/{month}"
    headers = {
        "Host": "www.histdata.com",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
        "Origin": "https://www.histdata.com",
        "Upgrade-Insecure-Requests": "1",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": referer,
    }

    session = requests.Session()
    response = session.get(referer, allow_redirects=True, timeout=60, verify=verify_ssl)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    token_node = soup.find("input", {"id": "tk"})
    if token_node is None or not token_node.attrs.get("value"):
        raise RuntimeError(f"HistData token was not found for {pair} {year}-{month:02d}")

    data = {
        "tk": token_node.attrs["value"],
        "date": str(year),
        "datemonth": f"{year}{month:02d}",
        "platform": "NT",
        "timeframe": "T_LAST",
        "fxpair": pair.upper(),
    }
    download = session.post(HISTDATA_GET_URL, data=data, headers=headers, timeout=180, verify=verify_ssl)
    download.raise_for_status()
    if not download.content:
        raise RuntimeError(f"HistData returned an empty archive for {pair} {year}-{month:02d}")

    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / f"DAT_NT_{pair.upper()}_T_LAST_{year}{month:02d}.zip"
    target.write_bytes(download.content)
    return target


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "tick_count"]).rename_axis("timestamp")
