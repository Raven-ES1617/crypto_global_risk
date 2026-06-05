from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from parsers.histdata import (
    histdata_monthly_parquet_path,
    read_histdata_tick_last_zip,
    save_histdata_tick_last_to_parquet,
)
from parsers.storage import read_asset_parquet


def test_read_histdata_tick_last_zip_aggregates_ticks_to_seconds() -> None:
    frame = read_histdata_tick_last_zip(
        _zip_payload(
            "\n".join(
                [
                    "20200102 093000;100.0;1",
                    "20200102 093000;101.0;2",
                    "20200102 093001;102.0;3",
                ]
            )
        )
    )

    first_second = pd.Timestamp("2020-01-02T14:30:00Z")
    assert frame.index[0] == first_second
    assert frame.loc[first_second, "open"] == 100.0
    assert frame.loc[first_second, "high"] == 101.0
    assert frame.loc[first_second, "low"] == 100.0
    assert frame.loc[first_second, "close"] == 101.0
    assert frame.loc[first_second, "volume"] == 3.0
    assert frame.loc[first_second, "tick_count"] == 2


def test_save_histdata_tick_last_to_parquet_with_fake_downloader(tmp_path: Path) -> None:
    def downloader(pair: str, year: int, month: int, output_directory: Path) -> Path:
        path = output_directory / f"DAT_NT_{pair.upper()}_T_LAST_{year}{month:02d}.zip"
        path.write_bytes(_zip_payload("20200102 093000;100.0;1\n20200102 093001;101.0;1"))
        return path

    saved = save_histdata_tick_last_to_parquet(
        "spxusd",
        year=2020,
        month=1,
        root=tmp_path,
        asset_symbol="SPX",
        start="2020-01-02T14:30:00Z",
        end="2020-01-02T14:30:01Z",
        downloader=downloader,
    )
    loaded = read_asset_parquet(saved.path)

    assert saved.path == str(histdata_monthly_parquet_path(tmp_path, "SPX", 2020, 1))
    assert saved.rows == 2
    assert loaded["close"].tolist() == [100.0, 101.0]


def _zip_payload(text: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("DAT_NT_SPXUSD_T_LAST_202001.csv", text)
    return buffer.getvalue()
