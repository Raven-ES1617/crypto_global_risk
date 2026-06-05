from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

DEFAULT_SECOND_DATA_ROOT = Path("data") / "second"


@dataclass(frozen=True)
class DataFile:
    provider: str
    symbol: str
    source_symbol: str
    path: str
    rows: int
    start: str | None
    end: str | None


def safe_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol).replace(":", "_")


def asset_dir(root: Path | str, provider: str, symbol: str) -> Path:
    return Path(root) / provider / safe_symbol(symbol)


def normalize_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out.index.name = "timestamp"
        return out

    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    return out


def write_asset_parquet(
    frame: pd.DataFrame,
    *,
    root: Path | str = DEFAULT_SECOND_DATA_ROOT,
    provider: str,
    symbol: str,
    source_symbol: str | None = None,
    suffix: str | None = None,
    compression: str = "zstd",
) -> DataFile:
    out = normalize_time_index(frame)
    target_dir = asset_dir(root, provider, symbol)
    target_dir.mkdir(parents=True, exist_ok=True)

    if suffix is None:
        if out.empty:
            suffix = "empty"
        else:
            start = out.index.min().strftime("%Y%m%dT%H%M%S")
            end = out.index.max().strftime("%Y%m%dT%H%M%S")
            suffix = f"{start}_{end}"

    path = target_dir / f"{safe_symbol(symbol)}_{suffix}.parquet"
    out.to_parquet(path, compression=compression, index=True)

    return DataFile(
        provider=provider,
        symbol=symbol,
        source_symbol=source_symbol or symbol,
        path=str(path),
        rows=int(len(out)),
        start=None if out.empty else out.index.min().isoformat(),
        end=None if out.empty else out.index.max().isoformat(),
    )


def read_asset_parquet(path: Path | str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return normalize_time_index(frame)


def write_manifest(files: list[DataFile], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in files]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_manifest(path: Path | str) -> list[DataFile]:
    source = Path(path)
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    return [DataFile(**item) for item in payload]


def merge_manifest(existing: list[DataFile], new: list[DataFile]) -> list[DataFile]:
    merged = {item.path: item for item in existing}
    for item in new:
        merged[item.path] = item
    return sorted(merged.values(), key=lambda item: (item.provider, item.symbol, item.path))
