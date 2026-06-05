from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from parsers.assets import AssetSpec, get_default_assets
from parsers.binance import save_binance_klines, save_binance_monthly_or_daily_to_parquet
from parsers.histdata import histdata_monthly_parquet_path, save_histdata_tick_last_to_parquet
from parsers.storage import (
    DEFAULT_SECOND_DATA_ROOT,
    DataFile,
    asset_dir,
    merge_manifest,
    read_manifest,
    safe_symbol,
    write_manifest,
)


DEFAULT_START = "2020-01-01T00:00:00Z"
DEFAULT_END = "2020-01-01T00:04:59Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download per-asset second-level market data.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--data-root", default=str(DEFAULT_SECOND_DATA_ROOT))
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["crypto"],
        help="Groups: crypto histdata histdata_macro histdata_fx histdata_index.",
    )
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional source symbols, e.g. BTCUSDT SPX.")
    parser.add_argument(
        "--mode",
        choices=["rest", "monthly-zip"],
        default="rest",
        help="Binance mode. monthly-zip writes monthly parquet from monthly archives or daily archive fallback.",
    )
    parser.add_argument("--year", type=int, default=None, help="Required for --mode monthly-zip.")
    parser.add_argument("--month", type=int, default=None, help="Required for --mode monthly-zip.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for Binance monthly zip mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-asset files.")
    parser.add_argument(
        "--histdata-no-ssl-verify",
        action="store_true",
        help="Disable TLS certificate verification for HistData downloads when histdata.com has certificate issues.",
    )
    parser.add_argument("--list-assets", action="store_true")
    args = parser.parse_args(argv)

    assets = get_default_assets(args.groups)
    if args.symbols:
        wanted = {item.upper() for item in args.symbols}
        assets = [asset for asset in assets if asset.source_symbol.upper() in wanted or asset.symbol.upper() in wanted]

    if args.list_assets:
        for asset in assets:
            print(asdict(asset))
        return 0

    files = download_assets(
        assets,
        start=args.start,
        end=args.end,
        data_root=Path(args.data_root),
        mode=args.mode,
        year=args.year,
        month=args.month,
        workers=args.workers,
        overwrite=args.overwrite,
        histdata_verify_ssl=not args.histdata_no_ssl_verify,
    )
    manifest_path = Path(args.data_root) / "manifest.json"
    merged = merge_manifest(read_manifest(manifest_path), files)
    manifest = write_manifest(merged, manifest_path)
    print(f"Saved {len(files)} files")
    print(f"Manifest: {manifest} ({len(merged)} total entries)")
    return 0


def download_assets(
    assets: list[AssetSpec],
    *,
    start: str,
    end: str,
    data_root: Path,
    mode: str = "rest",
    year: int | None = None,
    month: int | None = None,
    workers: int = 8,
    overwrite: bool = False,
    histdata_verify_ssl: bool = True,
) -> list[DataFile]:
    files: list[DataFile] = []

    histdata_assets = [asset for asset in assets if asset.provider == "histdata"]
    if histdata_assets:
        files.extend(
            download_histdata_tick_last_parallel(
                histdata_assets,
                start=start,
                end=end,
                data_root=data_root,
                year=year,
                month=month,
                workers=workers,
                overwrite=overwrite,
                verify_ssl=histdata_verify_ssl,
            )
        )
        assets = [asset for asset in assets if asset.provider != "histdata"]

    if mode == "monthly-zip":
        binance_assets = [asset for asset in assets if asset.provider == "binance"]
        other_assets = [asset for asset in assets if asset.provider != "binance"]
        files.extend(
            download_binance_monthly_zip_parallel(
                binance_assets,
                start=start,
                end=end,
                data_root=data_root,
                year=year,
                month=month,
                workers=workers,
                overwrite=overwrite,
            )
        )
        assets = other_assets

    failures: list[str] = []
    for asset in assets:
        print(f"[{asset.provider}] {asset.source_symbol} -> {asset.symbol}")
        try:
            if asset.provider == "binance":
                files.append(
                    save_binance_klines(
                        asset.source_symbol,
                        start=start,
                        end=end,
                        root=data_root,
                        interval="1s",
                        asset_symbol=asset.symbol,
                    )
                )
            else:
                raise ValueError(f"unsupported provider: {asset.provider}")
        except Exception as exc:
            print(f"  failed: {exc}")
            failures.append(f"{asset.provider}:{asset.source_symbol}: {exc}")
    if failures:
        detail = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... and {len(failures) - 5} more"
        raise RuntimeError(f"failed to download {len(failures)} asset(s): {detail}{extra}")
    return files


def download_histdata_tick_last_parallel(
    assets: list[AssetSpec],
    *,
    start: str,
    end: str,
    data_root: Path,
    year: int | None = None,
    month: int | None = None,
    workers: int = 8,
    overwrite: bool = False,
    verify_ssl: bool = True,
) -> list[DataFile]:
    if not assets:
        return []
    if (year is None) != (month is None):
        raise ValueError("--year and --month must be provided together")
    months = [(year, month)] if year is not None and month is not None else month_pairs(start, end)

    tasks = [(asset, int(y), int(m)) for asset in assets for y, m in months]
    max_workers = max(1, min(int(workers), len(tasks)))
    print(f"[histdata tick-last] tasks={len(tasks)} workers={max_workers}")

    files: list[DataFile] = []
    completed = 0
    skipped = 0
    failed = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _download_one_histdata_month,
                asset,
                y,
                m,
                start,
                end,
                data_root,
                overwrite,
                verify_ssl,
            ): (asset, y, m)
            for asset, y, m in tasks
        }
        for future in as_completed(future_to_task):
            asset, y, m = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                if result is None:
                    skipped += 1
                else:
                    files.append(result)
                    print(f"  ok {asset.symbol} {y}-{m:02d}: {result.rows:,} rows")
            except Exception as exc:
                failed += 1
                print(f"  failed {asset.symbol} {y}-{m:02d}: {exc}")
                failures.append(f"{asset.symbol} {y}-{m:02d}: {exc}")
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  progress {completed}/{len(tasks)} ok={len(files)} skipped={skipped} failed={failed}")

    if failures:
        detail = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... and {len(failures) - 5} more"
        raise RuntimeError(f"failed to download {len(failures)} HistData file(s): {detail}{extra}")

    return files


def download_binance_monthly_zip_parallel(
    assets: list[AssetSpec],
    *,
    start: str,
    end: str,
    data_root: Path,
    year: int | None = None,
    month: int | None = None,
    workers: int = 8,
    overwrite: bool = False,
) -> list[DataFile]:
    if not assets:
        return []
    if (year is None) != (month is None):
        raise ValueError("--year and --month must be provided together")
    months = [(year, month)] if year is not None and month is not None else month_pairs(start, end)

    tasks = [(asset, int(y), int(m)) for asset in assets for y, m in months]
    max_workers = max(1, min(int(workers), len(tasks)))
    print(f"[binance monthly-zip/daily-fallback] tasks={len(tasks)} workers={max_workers}")

    files: list[DataFile] = []
    completed = 0
    skipped = 0
    failed = 0
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _download_one_binance_month,
                asset,
                y,
                m,
                data_root,
                overwrite,
            ): (asset, y, m)
            for asset, y, m in tasks
        }
        for future in as_completed(future_to_task):
            asset, y, m = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                if result is None:
                    skipped += 1
                else:
                    files.append(result)
                    print(f"  ok {asset.symbol} {y}-{m:02d}: {result.rows:,} rows")
            except Exception as exc:
                failed += 1
                print(f"  failed {asset.symbol} {y}-{m:02d}: {exc}")
                failures.append(f"{asset.symbol} {y}-{m:02d}: {exc}")
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  progress {completed}/{len(tasks)} ok={len(files)} skipped={skipped} failed={failed}")

    if failures:
        detail = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... and {len(failures) - 5} more"
        raise RuntimeError(f"failed to download {len(failures)} monthly file(s): {detail}{extra}")

    return files


def month_pairs(start: str, end: str) -> list[tuple[int, int]]:
    start_ts = pd.to_datetime(start, utc=True).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_ts = pd.to_datetime(end, utc=True).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if end_ts < start_ts:
        raise ValueError("end must be greater than or equal to start")
    months = pd.date_range(start_ts, end_ts, freq="MS", tz="UTC")
    return [(int(ts.year), int(ts.month)) for ts in months]


def _download_one_binance_month(
    asset: AssetSpec,
    year: int,
    month: int,
    data_root: Path,
    overwrite: bool,
) -> DataFile | None:
    target = _binance_monthly_parquet_path(data_root, asset.symbol, year, month)
    if target.exists() and not overwrite:
        return None
    return save_binance_monthly_or_daily_to_parquet(
        asset.source_symbol,
        year=year,
        month=month,
        root=data_root,
        interval="1s",
        asset_symbol=asset.symbol,
    )


def _download_one_histdata_month(
    asset: AssetSpec,
    year: int,
    month: int,
    start: str,
    end: str,
    data_root: Path,
    overwrite: bool,
    verify_ssl: bool,
) -> DataFile | None:
    target = histdata_monthly_parquet_path(data_root, asset.symbol, year, month)
    if target.exists() and not overwrite:
        return None
    return save_histdata_tick_last_to_parquet(
        asset.source_symbol,
        year=year,
        month=month,
        root=data_root,
        asset_symbol=asset.symbol,
        start=start,
        end=end,
        verify_ssl=verify_ssl,
    )


def _binance_monthly_parquet_path(data_root: Path, symbol: str, year: int, month: int) -> Path:
    return asset_dir(data_root, "binance", symbol) / f"{safe_symbol(symbol)}_1s_{year}{month:02d}.parquet"


if __name__ == "__main__":
    raise SystemExit(main())
