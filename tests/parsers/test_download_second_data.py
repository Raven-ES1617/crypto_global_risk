from __future__ import annotations

from pathlib import Path

import pytest

from parsers.assets import AssetSpec
from parsers.download_second_data import (
    _binance_monthly_parquet_path,
    download_histdata_tick_last_parallel,
    download_assets,
    download_binance_monthly_zip_parallel,
    month_pairs,
)
from parsers.storage import write_asset_parquet


def test_month_pairs_from_start_end() -> None:
    assert month_pairs("2020-01-15T00:00:00Z", "2020-03-01T00:00:00Z") == [
        (2020, 1),
        (2020, 2),
        (2020, 3),
    ]


def test_binance_monthly_parquet_path(tmp_path: Path) -> None:
    path = _binance_monthly_parquet_path(tmp_path, "BTC", 2020, 1)
    assert path == tmp_path / "binance" / "BTC" / "BTC_1s_202001.parquet"


def test_parallel_monthly_zip_skips_existing(monkeypatch, tmp_path: Path) -> None:
    asset = AssetSpec("BTC", "binance", "BTCUSDT", "crypto", "spot_crypto")
    existing = write_asset_parquet(
        frame=_one_row_frame(),
        root=tmp_path,
        provider="binance",
        symbol="BTC",
        source_symbol="BTCUSDT",
        suffix="1s_202001",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing file should have been skipped")

    monkeypatch.setattr(
        "parsers.download_second_data.save_binance_monthly_or_daily_to_parquet",
        fail_if_called,
    )

    files = download_binance_monthly_zip_parallel(
        [asset],
        start="2020-01-01T00:00:00Z",
        end="2020-01-31T00:00:00Z",
        data_root=tmp_path,
        workers=2,
        overwrite=False,
    )

    assert files == []
    assert Path(existing.path).exists()


def test_parallel_monthly_zip_downloads_missing(monkeypatch, tmp_path: Path) -> None:
    asset = AssetSpec("BTC", "binance", "BTCUSDT", "crypto", "spot_crypto")
    calls: list[tuple[str, int, int]] = []

    def fake_save(source_symbol, *, year, month, root, interval, asset_symbol):
        calls.append((source_symbol, year, month))
        return write_asset_parquet(
            frame=_one_row_frame(),
            root=root,
            provider="binance",
            symbol=asset_symbol,
            source_symbol=source_symbol,
            suffix=f"{interval}_{year}{month:02d}",
        )

    monkeypatch.setattr(
        "parsers.download_second_data.save_binance_monthly_or_daily_to_parquet",
        fake_save,
    )

    files = download_binance_monthly_zip_parallel(
        [asset],
        start="2020-01-01T00:00:00Z",
        end="2020-02-01T00:00:00Z",
        data_root=tmp_path,
        workers=2,
        overwrite=False,
    )

    assert sorted(calls) == [("BTCUSDT", 2020, 1), ("BTCUSDT", 2020, 2)]
    assert len(files) == 2


def test_parallel_monthly_zip_raises_on_failed_download(monkeypatch, tmp_path: Path) -> None:
    asset = AssetSpec("BTC", "binance", "BTCUSDT", "crypto", "spot_crypto")

    def fake_save(*args, **kwargs):
        raise RuntimeError("provider error")

    monkeypatch.setattr(
        "parsers.download_second_data.save_binance_monthly_or_daily_to_parquet",
        fake_save,
    )

    with pytest.raises(RuntimeError, match="failed to download 1 monthly file"):
        download_binance_monthly_zip_parallel(
            [asset],
            start="2020-01-01T00:00:00Z",
            end="2020-01-01T00:00:00Z",
            data_root=tmp_path,
            workers=1,
            overwrite=False,
        )


def test_parallel_histdata_skips_existing(monkeypatch, tmp_path: Path) -> None:
    asset = AssetSpec("SPX", "histdata", "spxusd", "histdata_index", "index_cfd")
    existing = write_asset_parquet(
        frame=_one_row_frame(),
        root=tmp_path,
        provider="histdata",
        symbol="SPX",
        source_symbol="SPXUSD",
        suffix="1s_202001",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing file should have been skipped")

    monkeypatch.setattr(
        "parsers.download_second_data.save_histdata_tick_last_to_parquet",
        fail_if_called,
    )

    files = download_histdata_tick_last_parallel(
        [asset],
        start="2020-01-01T00:00:00Z",
        end="2020-01-31T00:00:00Z",
        data_root=tmp_path,
        workers=1,
        overwrite=False,
    )

    assert files == []
    assert Path(existing.path).exists()


def test_parallel_histdata_downloads_missing(monkeypatch, tmp_path: Path) -> None:
    asset = AssetSpec("SPX", "histdata", "spxusd", "histdata_index", "index_cfd")
    calls: list[tuple[str, int, int]] = []

    def fake_save(source_symbol, *, year, month, root, asset_symbol, start, end, verify_ssl):
        calls.append((source_symbol, year, month))
        return write_asset_parquet(
            frame=_one_row_frame(),
            root=root,
            provider="histdata",
            symbol=asset_symbol,
            source_symbol=source_symbol,
            suffix=f"1s_{year}{month:02d}",
        )

    monkeypatch.setattr(
        "parsers.download_second_data.save_histdata_tick_last_to_parquet",
        fake_save,
    )

    files = download_histdata_tick_last_parallel(
        [asset],
        start="2020-01-01T00:00:00Z",
        end="2020-02-01T00:00:00Z",
        data_root=tmp_path,
        workers=2,
        overwrite=False,
    )

    assert sorted(calls) == [("spxusd", 2020, 1), ("spxusd", 2020, 2)]
    assert len(files) == 2


def test_download_assets_raises_on_asset_failure(tmp_path: Path) -> None:
    asset = AssetSpec("BAD", "unsupported", "BAD", "test", "test")

    with pytest.raises(RuntimeError, match="failed to download 1 asset"):
        download_assets(
            [asset],
            start="2024-01-02T14:30:00Z",
            end="2024-01-02T14:31:00Z",
            data_root=tmp_path,
        )


def _one_row_frame():
    import pandas as pd

    return pd.DataFrame(
        {"close": [1.0]},
        index=pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True),
    )
