from __future__ import annotations

from pathlib import Path

import pandas as pd

from parsers.assets import AssetSpec
from parsers.checkpoint_download import (
    DownloadTask,
    build_checkpoint_tasks,
    month_pairs_for_archives,
    task_output_path,
)
from parsers.storage import write_asset_parquet, write_manifest


def test_month_pairs_for_archives_excludes_current_month_by_default() -> None:
    pairs = month_pairs_for_archives(
        "2026-03-01T00:00:00Z",
        "2026-05-22T00:00:00Z",
        current_month=pd.Timestamp("2026-05-01T00:00:00Z"),
    )

    assert pairs == [(2026, 3), (2026, 4)]


def test_task_output_path_for_supported_providers(tmp_path: Path) -> None:
    binance = DownloadTask("binance", "BTC", "BTCUSDT", 2020, 1)
    histdata = DownloadTask("histdata", "SPX", "spxusd", 2020, 1)

    assert task_output_path(tmp_path, binance) == tmp_path / "binance" / "BTC" / "BTC_1s_202001.parquet"
    assert task_output_path(tmp_path, histdata) == tmp_path / "histdata" / "SPX" / "SPX_1s_202001.parquet"


def test_build_checkpoint_tasks_skips_existing_files(tmp_path: Path) -> None:
    asset = AssetSpec("BTC", "binance", "BTCUSDT", "crypto", "spot_crypto")
    existing = write_asset_parquet(
        _one_row_frame(),
        root=tmp_path,
        provider="binance",
        symbol="BTC",
        source_symbol="BTCUSDT",
        suffix="1s_202001",
    )
    write_manifest([existing], tmp_path / "manifest.json")

    tasks = build_checkpoint_tasks(
        [asset],
        start="2020-01-01T00:00:00Z",
        end="2020-02-01T00:00:00Z",
        data_root=tmp_path,
    )

    assert tasks == [DownloadTask("binance", "BTC", "BTCUSDT", 2020, 2)]


def _one_row_frame():
    import pandas as pd

    return pd.DataFrame(
        {"close": [1.0]},
        index=pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True),
    )
