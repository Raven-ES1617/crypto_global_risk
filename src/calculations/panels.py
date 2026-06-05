from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parsers.storage import normalize_time_index


MONTH_PATTERN = re.compile(r"_(\d{6})\.parquet$")


@dataclass(frozen=True)
class AssetPanelStats:
    symbol: str
    provider: str
    files: int
    rows_read: int
    invalid_rows_removed: int
    resampled_rows: int
    first_timestamp: str | None
    last_timestamp: str | None


@dataclass(frozen=True)
class PanelBuildResult:
    prices: pd.DataFrame
    returns: pd.DataFrame
    raw_prices: pd.DataFrame
    stats: list[AssetPanelStats]

    def stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(item) for item in self.stats])


def discover_asset_files(
    symbol: str,
    *,
    data_root: Path | str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source_frequency: str = "1s",
) -> list[Path]:
    root = Path(data_root)
    start_period = _month_period(start)
    end_period = _month_period(end)
    candidates = sorted(root.glob(f"*/*/{symbol}_{source_frequency}_*.parquet"))
    files: list[Path] = []
    for path in candidates:
        match = MONTH_PATTERN.search(path.name)
        if not match:
            continue
        period = pd.Period(match.group(1), freq="M")
        if start_period <= period <= end_period:
            files.append(path)
    return files


def _month_period(value: str | pd.Timestamp) -> pd.Period:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_period("M")


def build_price_panel(
    symbols: tuple[str, ...] | list[str],
    *,
    data_root: Path | str,
    frequency: str,
    pandas_rule: str,
    start: str,
    end: str,
    stale_limit_periods: int,
    price_ranges: dict[str, tuple[float | None, float | None]] | None = None,
    source_frequency_by_symbol: dict[str, str] | None = None,
) -> PanelBuildResult:
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")

    series_by_symbol: list[pd.Series] = []
    stats: list[AssetPanelStats] = []
    source_frequency_by_symbol = source_frequency_by_symbol or {}
    for symbol in symbols:
        source_frequency = source_frequency_by_symbol.get(symbol, "1s")
        if frequency == "1s" and source_frequency != "1s":
            raise ValueError(f"{symbol} source frequency is {source_frequency}; refusing to synthesize 1s data")

        files = discover_asset_files(
            symbol,
            data_root=data_root,
            start=start_ts,
            end=end_ts,
            source_frequency=source_frequency,
        )
        if not files:
            raise FileNotFoundError(f"no parquet files found for {symbol} between {start_ts} and {end_ts}")

        series, asset_stats = _load_resampled_asset(
            symbol,
            files,
            pandas_rule=pandas_rule,
            start=start_ts,
            end=end_ts,
            price_range=None if price_ranges is None else price_ranges.get(symbol),
        )
        series_by_symbol.append(series.rename(symbol))
        stats.append(asset_stats)

    raw = pd.concat(series_by_symbol, axis=1, sort=True).sort_index()
    raw = raw.loc[(raw.index >= start_ts) & (raw.index <= end_ts)]
    aligned = raw.ffill(limit=int(stale_limit_periods)).dropna(how="any")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    aligned = aligned[(aligned > 0).all(axis=1)]

    if frequency == "1d":
        aligned = aligned[aligned.index.dayofweek < 5]

    returns = np.log(aligned).diff().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return PanelBuildResult(prices=aligned, returns=returns, raw_prices=raw, stats=stats)


def save_panel_artifacts(
    result: PanelBuildResult,
    *,
    frequency: str,
    panel_path: Path,
    returns_path: Path,
    stats_path: Path,
) -> None:
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    result.prices.to_parquet(panel_path, compression="zstd", index=True)
    result.returns.to_parquet(returns_path, compression="zstd", index=True)
    stats = result.stats_frame()
    stats.insert(0, "frequency", frequency)
    stats.to_csv(stats_path, index=False)


def _load_resampled_asset(
    symbol: str,
    files: list[Path],
    *,
    pandas_rule: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_range: tuple[float | None, float | None] | None,
) -> tuple[pd.Series, AssetPanelStats]:
    chunks: list[pd.Series] = []
    rows_read = 0
    invalid_rows_removed = 0
    provider = files[0].parent.parent.name
    for path in files:
        frame = pd.read_parquet(path, columns=["close"])
        frame = normalize_time_index(frame)
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
        rows_read += int(len(frame))
        if frame.empty:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").dropna()
        close = close[close > 0]
        if price_range is not None:
            before_filter = len(close)
            min_price, max_price = price_range
            if min_price is not None:
                close = close[close >= float(min_price)]
            if max_price is not None:
                close = close[close <= float(max_price)]
            invalid_rows_removed += before_filter - len(close)
        if close.empty:
            continue
        chunks.append(close.resample(pandas_rule).last().dropna())

    if not chunks:
        raise ValueError(f"no usable close prices found for {symbol}")

    series = pd.concat(chunks).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.loc[(series.index >= start) & (series.index <= end)]
    stats = AssetPanelStats(
        symbol=symbol,
        provider=provider,
        files=len(files),
        rows_read=rows_read,
        invalid_rows_removed=invalid_rows_removed,
        resampled_rows=int(series.notna().sum()),
        first_timestamp=None if series.empty else series.index.min().isoformat(),
        last_timestamp=None if series.empty else series.index.max().isoformat(),
    )
    return series, stats
