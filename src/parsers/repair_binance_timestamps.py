from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.types as pat
import pyarrow.parquet as pq

from parsers.storage import DEFAULT_SECOND_DATA_ROOT, normalize_time_index


MICROSECOND_THRESHOLD = 10_000_000_000_000


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.data_root)
    paths = sorted((root / "binance").glob("*/*_1s_*.parquet"))
    fixed = 0
    checked = 0
    for path in paths:
        checked += 1
        if repair_file(path, dry_run=args.dry_run):
            fixed += 1
            action = "would fix" if args.dry_run else "fixed"
            print(f"{action}: {path}", flush=True)
    print(f"checked={checked} fixed={fixed}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Binance parquet files written with microsecond timestamps as ms.")
    parser.add_argument("--data-root", default=str(DEFAULT_SECOND_DATA_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def repair_file(path: Path, *, dry_run: bool = False) -> bool:
    table = pq.read_table(path)
    if "timestamp" not in table.column_names:
        return False

    timestamp_type = table.schema.field("timestamp").type
    if not pat.is_timestamp(timestamp_type) or timestamp_type.unit != "ms":
        return False

    raw_timestamp = table["timestamp"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    if len(raw_timestamp) == 0 or int(raw_timestamp.max()) < MICROSECOND_THRESHOLD:
        return False
    if dry_run:
        return True

    data: dict[str, object] = {}
    for name in table.column_names:
        if name == "timestamp":
            continue
        data[name] = table[name].to_numpy(zero_copy_only=False)

    frame = pd.DataFrame(data)
    frame["timestamp"] = pd.to_datetime(raw_timestamp, unit="us", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = normalize_time_index(frame)
    frame.to_parquet(path, compression="zstd", index=True)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
