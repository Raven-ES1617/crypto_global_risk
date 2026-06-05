from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from parsers.binance_us import save_binance_us_monthly_zip_to_parquet
from parsers.download_second_data import month_pairs
from parsers.storage import (
    DEFAULT_SECOND_DATA_ROOT,
    DataFile,
    asset_dir,
    merge_manifest,
    read_manifest,
    safe_symbol,
    write_manifest,
)


DEFAULT_START = "2019-09-01"
DEFAULT_END = "2026-04-30 23:59:59+00:00"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Binance.US USDT/USD monthly kline archives.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--data-root", default=str(DEFAULT_SECOND_DATA_ROOT))
    parser.add_argument("--interval", default="1m", choices=["1s", "1m", "1h", "1d"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    files = download_usdt_usd_range(
        start=args.start,
        end=args.end,
        data_root=Path(args.data_root),
        interval=args.interval,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    manifest_path = Path(args.data_root) / "manifest.json"
    merged = merge_manifest(read_manifest(manifest_path), files)
    manifest = write_manifest(merged, manifest_path)
    print(f"Saved {len(files)} files")
    print(f"Manifest: {manifest} ({len(merged)} total entries)")
    return 0


def download_usdt_usd_range(
    *,
    start: str,
    end: str,
    data_root: Path,
    interval: str = "1m",
    workers: int = 8,
    overwrite: bool = False,
) -> list[DataFile]:
    months = month_pairs(start, end)
    tasks = [(int(year), int(month)) for year, month in months]
    max_workers = max(1, min(int(workers), len(tasks)))
    files: list[DataFile] = []
    failures: list[str] = []
    skipped = 0

    print(f"[binance_us USDTUSD {interval}] tasks={len(tasks)} workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_download_one_month, year, month, data_root, interval, overwrite): (year, month)
            for year, month in tasks
        }
        for future in as_completed(future_to_task):
            year, month = future_to_task[future]
            try:
                result = future.result()
                if result is None:
                    skipped += 1
                else:
                    files.append(result)
                    print(f"  ok USDT {year}-{month:02d}: {result.rows:,} rows")
            except Exception as exc:
                failures.append(f"{year}-{month:02d}: {exc}")
                print(f"  failed USDT {year}-{month:02d}: {exc}")

    if failures:
        detail = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... and {len(failures) - 5} more"
        raise RuntimeError(f"failed to download {len(failures)} USDTUSD file(s): {detail}{extra}")
    print(f"  skipped={skipped}")
    return files


def _download_one_month(
    year: int,
    month: int,
    data_root: Path,
    interval: str,
    overwrite: bool,
) -> DataFile | None:
    target = asset_dir(data_root, "binance_us", "USDT") / f"{safe_symbol('USDT')}_{interval}_{year}{month:02d}.parquet"
    if target.exists() and not overwrite:
        return None
    return save_binance_us_monthly_zip_to_parquet(
        "USDTUSD",
        year=year,
        month=month,
        root=data_root,
        interval=interval,
        asset_symbol="USDT",
    )


if __name__ == "__main__":
    raise SystemExit(main())
