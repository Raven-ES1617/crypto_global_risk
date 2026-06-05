from __future__ import annotations

import pandas as pd

from calculations.panels import build_price_panel
from parsers.storage import write_asset_parquet


def test_build_price_panel_resamples_and_aligns(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="s", tz="UTC")
    root = tmp_path / "second"
    write_asset_parquet(
        pd.DataFrame({"close": range(100, 220)}, index=idx),
        root=root,
        provider="binance",
        symbol="BTC",
        suffix="1s_202401",
    )
    write_asset_parquet(
        pd.DataFrame({"close": range(200, 260)}, index=idx[::2]),
        root=root,
        provider="histdata",
        symbol="SPX",
        suffix="1s_202401",
    )

    result = build_price_panel(
        ["BTC", "SPX"],
        data_root=root,
        frequency="1min",
        pandas_rule="1min",
        start="2024-01-01",
        end="2024-01-01 00:01:59+00:00",
        stale_limit_periods=1,
    )

    assert list(result.prices.columns) == ["BTC", "SPX"]
    assert len(result.prices) == 2
    assert len(result.returns) == 1


def test_build_price_panel_can_filter_symbol_price_range(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="s", tz="UTC")
    root = tmp_path / "second"
    write_asset_parquet(
        pd.DataFrame({"close": [100.0, 25_000.0, 101.0, 102.0]}, index=idx),
        root=root,
        provider="histdata",
        symbol="UDXUSD",
        suffix="1s_202401",
    )
    write_asset_parquet(
        pd.DataFrame({"close": [10.0, 10.1, 10.2, 10.3]}, index=idx),
        root=root,
        provider="binance",
        symbol="BTC",
        suffix="1s_202401",
    )

    result = build_price_panel(
        ["UDXUSD", "BTC"],
        data_root=root,
        frequency="1s",
        pandas_rule="1s",
        start="2024-01-01",
        end="2024-01-01 00:00:03+00:00",
        stale_limit_periods=1,
        price_ranges={"UDXUSD": (50.0, 150.0)},
    )

    assert result.prices["UDXUSD"].max() == 102.0
    assert result.stats_frame().loc[0, "invalid_rows_removed"] == 1


def test_build_price_panel_can_use_non_second_source_frequency(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    root = tmp_path / "second"
    write_asset_parquet(
        pd.DataFrame({"close": [1.0, 1.001, 0.999]}, index=idx),
        root=root,
        provider="binance_us",
        symbol="USDT",
        suffix="1m_202401",
    )
    write_asset_parquet(
        pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx),
        root=root,
        provider="binance",
        symbol="BTC",
        suffix="1s_202401",
    )

    result = build_price_panel(
        ["USDT", "BTC"],
        data_root=root,
        frequency="1min",
        pandas_rule="1min",
        start="2024-01-01",
        end="2024-01-01 00:02:00+00:00",
        stale_limit_periods=1,
        source_frequency_by_symbol={"USDT": "1m"},
    )

    assert list(result.prices.columns) == ["USDT", "BTC"]
    assert len(result.prices) == 3
    assert result.stats_frame().loc[0, "provider"] == "binance_us"
