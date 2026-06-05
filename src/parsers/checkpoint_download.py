from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from parsers.assets import AssetSpec, get_default_assets
from parsers.binance import save_binance_monthly_or_daily_to_parquet
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


DEFAULT_GROUPS = ["crypto", "histdata"]
DEFAULT_START = "2018-06-01T00:00:00Z"
DEFAULT_END = "2026-04-30T23:59:59Z"
MIN_ARCHIVE_MONTHS = {
    "BTCUSDT": (2017, 8),
    "ETHUSDT": (2017, 8),
    "BNBUSDT": (2017, 11),
    "LTCUSDT": (2017, 12),
    "ADAUSDT": (2018, 4),
    "XRPUSDT": (2018, 5),
    "TRXUSDT": (2018, 6),
}


@dataclass(frozen=True)
class DownloadTask:
    provider: str
    symbol: str
    source_symbol: str
    year: int
    month: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpointed second-level downloader.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--data-root", default=str(DEFAULT_SECOND_DATA_ROOT))
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--include-current-month", action="store_true")
    parser.add_argument("--histdata-no-ssl-verify", action="store_true")
    parser.add_argument("--checkpoint-dir", default=str(Path("data") / "second" / "checkpoints"))
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    assets = get_default_assets(args.groups)
    if args.symbols:
        wanted = {item.upper() for item in args.symbols}
        assets = [asset for asset in assets if asset.symbol.upper() in wanted or asset.source_symbol.upper() in wanted]

    tasks = build_checkpoint_tasks(
        assets,
        start=args.start,
        end=args.end,
        data_root=data_root,
        include_current_month=args.include_current_month,
    )
    log_path = checkpoint_dir / "download_events.jsonl"
    state_path = checkpoint_dir / "download_state.json"

    runner = CheckpointRunner(
        tasks,
        start=args.start,
        end=args.end,
        data_root=data_root,
        workers=args.workers,
        duration_seconds=None if args.duration_hours is None else args.duration_hours * 3600,
        histdata_verify_ssl=not args.histdata_no_ssl_verify,
        log_path=log_path,
        state_path=state_path,
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


class CheckpointRunner:
    def __init__(
        self,
        tasks: list[DownloadTask],
        *,
        start: str,
        end: str,
        data_root: Path,
        workers: int,
        duration_seconds: float | None,
        histdata_verify_ssl: bool,
        log_path: Path,
        state_path: Path,
    ) -> None:
        self.tasks = tasks
        self.start = start
        self.end = end
        self.data_root = data_root
        self.workers = max(1, int(workers))
        self.duration_seconds = duration_seconds
        self.histdata_verify_ssl = histdata_verify_ssl
        self.log_path = log_path
        self.state_path = state_path
        self.manifest_path = data_root / "manifest.json"
        self.manifest_paths = {item.path for item in read_manifest(self.manifest_path)}
        self.stats = {
            "total_tasks": len(tasks),
            "submitted": 0,
            "ok": 0,
            "skipped_existing": 0,
            "failed": 0,
            "not_submitted": 0,
        }

    def run(self) -> dict[str, object]:
        started = time.time()
        pending_tasks = iter(self.tasks)
        futures = {}

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while len(futures) < self.workers and self._within_time(started):
                task = next(pending_tasks, None)
                if task is None:
                    break
                if self._skip_existing(task):
                    continue
                futures[executor.submit(self._download_one, task)] = task
                self.stats["submitted"] += 1

            while futures:
                done, _pending = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    self._write_state(started)
                    continue

                for future in done:
                    task = futures.pop(future)
                    try:
                        data_file = future.result()
                    except Exception as exc:
                        self.stats["failed"] += 1
                        self._write_event("failed", task, error=str(exc))
                    else:
                        self.stats["ok"] += 1
                        self._checkpoint_manifest(data_file)
                        self._write_event("ok", task, data_file=data_file)

                while len(futures) < self.workers and self._within_time(started):
                    task = next(pending_tasks, None)
                    if task is None:
                        break
                    if self._skip_existing(task):
                        continue
                    futures[executor.submit(self._download_one, task)] = task
                    self.stats["submitted"] += 1
                self._write_state(started)

        self.stats["not_submitted"] = self.stats["total_tasks"] - (
            self.stats["submitted"] + self.stats["skipped_existing"]
        )
        self._write_state(started, finished=True)
        return {
            **self.stats,
            "elapsed_seconds": round(time.time() - started, 3),
            "manifest": str(self.manifest_path),
            "events_log": str(self.log_path),
            "state": str(self.state_path),
        }

    def _download_one(self, task: DownloadTask) -> DataFile:
        if task.provider == "binance":
            return save_binance_monthly_or_daily_to_parquet(
                task.source_symbol,
                year=task.year,
                month=task.month,
                root=self.data_root,
                interval="1s",
                asset_symbol=task.symbol,
            )
        if task.provider == "histdata":
            return save_histdata_tick_last_to_parquet(
                task.source_symbol,
                year=task.year,
                month=task.month,
                root=self.data_root,
                asset_symbol=task.symbol,
                start=self.start,
                end=self.end,
                verify_ssl=self.histdata_verify_ssl,
            )
        raise ValueError(f"unsupported checkpoint provider: {task.provider}")

    def _skip_existing(self, task: DownloadTask) -> bool:
        path = task_output_path(self.data_root, task)
        if not path.exists():
            return False
        if str(path) not in self.manifest_paths:
            data_file = describe_existing_parquet(path, task)
            self._checkpoint_manifest(data_file)
            self._write_event("checkpoint_existing", task, data_file=data_file)
        self.stats["skipped_existing"] += 1
        self._write_event("skipped_existing", task, path=str(path))
        return True

    def _checkpoint_manifest(self, data_file: DataFile) -> None:
        merged = merge_manifest(read_manifest(self.manifest_path), [data_file])
        write_manifest(merged, self.manifest_path)
        self.manifest_paths.add(data_file.path)

    def _within_time(self, started: float) -> bool:
        return self.duration_seconds is None or (time.time() - started) < self.duration_seconds

    def _write_event(
        self,
        status: str,
        task: DownloadTask,
        *,
        data_file: DataFile | None = None,
        path: str | None = None,
        error: str | None = None,
    ) -> None:
        event = {
            "time_utc": pd.Timestamp.now("UTC").isoformat(),
            "status": status,
            **asdict(task),
        }
        if data_file is not None:
            event.update({"path": data_file.path, "rows": data_file.rows, "start": data_file.start, "end": data_file.end})
        if path is not None:
            event["path"] = path
        if error is not None:
            event["error"] = error
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_state(self, started: float, *, finished: bool = False) -> None:
        payload = {
            **self.stats,
            "finished": finished,
            "elapsed_seconds": round(time.time() - started, 3),
            "updated_utc": pd.Timestamp.now("UTC").isoformat(),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_checkpoint_tasks(
    assets: list[AssetSpec],
    *,
    start: str,
    end: str,
    data_root: Path,
    include_current_month: bool = False,
) -> list[DownloadTask]:
    months = month_pairs_for_archives(start, end, include_current_month=include_current_month)
    tasks: list[DownloadTask] = []
    manifest_paths = {item.path for item in read_manifest(data_root / "manifest.json")}
    for asset in assets:
        if asset.provider not in {"binance", "histdata"}:
            continue
        asset_months = months_after_asset_start(asset, months)
        for year, month in asset_months:
            task = DownloadTask(asset.provider, asset.symbol, asset.source_symbol, year, month)
            path = task_output_path(data_root, task)
            if (not path.exists()) or str(path) not in manifest_paths:
                tasks.append(task)
    return sorted(tasks, key=lambda item: (item.provider, item.year, item.month, item.symbol))


def months_after_asset_start(asset: AssetSpec, months: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if asset.provider != "binance":
        return months
    first_month = MIN_ARCHIVE_MONTHS.get(asset.source_symbol.upper())
    if first_month is None:
        return months
    return [month for month in months if month >= first_month]


def month_pairs_for_archives(
    start: str,
    end: str,
    *,
    include_current_month: bool = False,
    current_month: pd.Timestamp | None = None,
) -> list[tuple[int, int]]:
    start_ts = pd.to_datetime(start, utc=True).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_ts = pd.to_datetime(end, utc=True).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if current_month is None:
        current_month = pd.Timestamp.now("UTC")
    current_month = current_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not include_current_month and end_ts >= current_month:
        end_ts = current_month - pd.offsets.MonthBegin(1)
    if end_ts < start_ts:
        return []
    return [(int(ts.year), int(ts.month)) for ts in pd.date_range(start_ts, end_ts, freq="MS", tz="UTC")]


def task_output_path(data_root: Path, task: DownloadTask) -> Path:
    if task.provider == "binance":
        return asset_dir(data_root, "binance", task.symbol) / f"{safe_symbol(task.symbol)}_1s_{task.year}{task.month:02d}.parquet"
    if task.provider == "histdata":
        return asset_dir(data_root, "histdata", task.symbol) / f"{safe_symbol(task.symbol)}_1s_{task.year}{task.month:02d}.parquet"
    raise ValueError(f"unsupported checkpoint provider: {task.provider}")


def describe_existing_parquet(path: Path, task: DownloadTask) -> DataFile:
    parquet_file = pq.ParquetFile(path)
    rows = int(parquet_file.metadata.num_rows)
    start = None
    end = None
    if rows > 0 and "timestamp" in parquet_file.schema.names:
        table = pq.read_table(path, columns=["timestamp"])
        timestamps = table.column("timestamp").to_pandas()
        if len(timestamps) > 0:
            index = pd.to_datetime(pd.Series(timestamps), utc=True)
            start = index.min().isoformat()
            end = index.max().isoformat()
    return DataFile(
        provider=task.provider,
        symbol=task.symbol,
        source_symbol=task.source_symbol.upper(),
        path=str(path),
        rows=rows,
        start=start,
        end=end,
    )


if __name__ == "__main__":
    raise SystemExit(main())
