from __future__ import annotations

import pandas as pd

from parsers.storage import (
    DataFile,
    merge_manifest,
    read_asset_parquet,
    read_manifest,
    safe_symbol,
    write_asset_parquet,
    write_manifest,
)


def test_safe_symbol_handles_provider_prefixes() -> None:
    assert safe_symbol("I:UDXUSD") == "I_UDXUSD"
    assert safe_symbol("BRK.B") == "BRK.B"


def test_write_and_read_asset_parquet(tmp_path) -> None:
    frame = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.to_datetime(["2020-01-01T00:00:01Z", "2020-01-01T00:00:00Z"], utc=True),
    )

    saved = write_asset_parquet(
        frame,
        root=tmp_path,
        provider="test",
        symbol="BTC",
        source_symbol="BTCUSDT",
        suffix="1s_202001",
    )

    loaded = read_asset_parquet(saved.path)
    assert saved.path.endswith(".parquet")
    assert saved.rows == 2
    assert loaded.index.is_monotonic_increasing
    assert loaded.index.name == "timestamp"
    assert list(loaded["close"]) == [2.0, 1.0]


def test_manifest_roundtrip_and_merge(tmp_path) -> None:
    old = DataFile("binance", "BTC", "BTCUSDT", "old.parquet", 1, "a", "b")
    updated = DataFile("binance", "BTC", "BTCUSDT", "old.parquet", 2, "a", "c")
    new = DataFile("binance", "ETH", "ETHUSDT", "new.parquet", 1, "a", "b")
    path = tmp_path / "manifest.json"

    write_manifest([old], path)
    merged = merge_manifest(read_manifest(path), [updated, new])

    assert len(merged) == 2
    assert {item.path: item.rows for item in merged} == {"old.parquet": 2, "new.parquet": 1}


def test_read_manifest_accepts_utf8_bom(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '[{"provider":"binance","symbol":"BTC","source_symbol":"BTCUSDT","path":"x.parquet","rows":1,"start":"a","end":"b"}]',
        encoding="utf-8-sig",
    )

    assert read_manifest(path)[0].symbol == "BTC"
