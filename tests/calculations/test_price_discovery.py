from __future__ import annotations

import pandas as pd
import pytest

from calculations.price_discovery import aggregate_price_discovery_blocks


def test_aggregate_price_discovery_blocks_averages_window_pairs() -> None:
    windows = pd.DataFrame(
        [
            {
                "frequency": "1d",
                "period": "pre_covid",
                "window_id": 1,
                "window_start": "2019-01-01T00:00:00+00:00",
                "window_end": "2019-02-01T00:00:00+00:00",
                "metric": "gis",
                "left_asset": "BTC",
                "right_asset": "SPX",
                "pair": "BTC-SPX",
                "status": "ok",
                "left_share": 0.7,
                "left_lower": 0.6,
                "left_upper": 0.8,
                "left_bound_width": 0.2,
            },
            {
                "frequency": "1d",
                "period": "pre_covid",
                "window_id": 1,
                "window_start": "2019-01-01T00:00:00+00:00",
                "window_end": "2019-02-01T00:00:00+00:00",
                "metric": "gis",
                "left_asset": "ETH",
                "right_asset": "NASDAQ100",
                "pair": "ETH-NASDAQ100",
                "status": "ok",
                "left_share": 0.5,
                "left_lower": 0.4,
                "left_upper": 0.7,
                "left_bound_width": 0.3,
            },
            {
                "frequency": "1d",
                "period": "pre_covid",
                "window_id": 1,
                "window_start": "2019-01-01T00:00:00+00:00",
                "window_end": "2019-02-01T00:00:00+00:00",
                "metric": "gis",
                "left_asset": "BTC",
                "right_asset": "GOLD",
                "pair": "BTC-GOLD",
                "status": "error",
                "left_share": 1.0,
            },
        ]
    )

    result = aggregate_price_discovery_blocks(windows)

    row = result.iloc[0]
    assert row["left_block"] == "crypto"
    assert row["right_block"] == "equity_index"
    assert row["pair_count"] == 2
    assert row["left_share"] == pytest.approx(0.6)
    assert row["left_lower"] == pytest.approx(0.5)
    assert row["left_upper"] == pytest.approx(0.75)
    assert row["left_bound_width"] == pytest.approx(0.25)
