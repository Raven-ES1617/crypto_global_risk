from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from calculations.run_window_gifs import WindowEstimate, _block_flow_long, select_even_windows, write_gif


def test_select_even_windows_spans_full_sample() -> None:
    prices = pd.DataFrame(
        {"BTC": range(20), "ETH": range(100, 120)},
        index=pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
    )

    windows = select_even_windows(
        prices,
        window=pd.Timedelta(days=4),
        step=pd.Timedelta(days=1),
        min_obs=4,
        max_windows=3,
    )

    assert len(windows) == 3
    assert windows[0][0] == prices.index.min()
    assert windows[-1][1] == prices.index[-1]


def test_write_gif_pads_different_frame_sizes(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (20, 10), "white").save(first)
    Image.new("RGB", (10, 20), "black").save(second)

    target = tmp_path / "out.gif"
    write_gif([first, second], path=target, duration_ms=100)

    assert target.exists()
    with Image.open(target) as gif:
        assert gif.n_frames == 2
        assert gif.size == (20, 20)


def test_block_flow_long_preserves_window_start() -> None:
    block_table = pd.DataFrame(
        [
            {
                "receiver_block": "crypto",
                "shock_block": "equity_index",
                "receiver_assets": 2,
                "shock_assets": 1,
                "pair_count": 2,
                "share_sum": 0.25,
                "average_receiver_share": 0.125,
                "average_pair_share": 0.125,
            }
        ]
    )
    window = WindowEstimate(
        frequency="1d",
        window_id=1,
        window_start="2018-12-17T00:00:00+00:00",
        window_end="2019-12-17T00:00:00+00:00",
        rows_available=260,
        status="ok",
        total_connectedness=0.5,
        lag_order_diff=1,
        coint_rank=0,
        error="",
        matrix=None,
    )

    rows = _block_flow_long(block_table, window)

    assert rows[0]["window_start"] == "2018-12-17T00:00:00+00:00"
    assert rows[0]["average_receiver_share"] == 0.125
    assert rows[0]["average_pair_share"] == 0.125
