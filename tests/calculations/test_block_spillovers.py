from __future__ import annotations

import pandas as pd
import pytest

from calculations.gfevd_analysis import block_matrix, block_spillover_table
from calculations.artifacts import ArtifactRegistry
from calculations.run_pipeline import _register_gfevd_outputs


def test_block_spillovers_aggregate_gfevd_cells_by_receiver_block() -> None:
    gfevd = pd.DataFrame(
        {
            "BTC": [0.70, 0.20, 0.30],
            "ETH": [0.10, 0.60, 0.10],
            "SPX": [0.20, 0.20, 0.60],
        },
        index=["BTC", "ETH", "SPX"],
    )
    block_map = {"BTC": "crypto", "ETH": "crypto", "SPX": "equity"}

    table = block_spillover_table(
        gfevd,
        block_map=block_map,
        block_order=("crypto", "equity"),
    )

    crypto_from_crypto = _row(table, receiver_block="crypto", shock_block="crypto")
    crypto_from_equity = _row(table, receiver_block="crypto", shock_block="equity")
    equity_from_crypto = _row(table, receiver_block="equity", shock_block="crypto")
    equity_from_equity = _row(table, receiver_block="equity", shock_block="equity")

    assert crypto_from_crypto["share_sum"] == pytest.approx(0.10 + 0.20)
    assert crypto_from_crypto["average_receiver_share"] == pytest.approx((0.10 + 0.20) / 2)
    assert crypto_from_crypto["pair_count"] == 2
    assert crypto_from_crypto["average_pair_share"] == pytest.approx((0.10 + 0.20) / 2)

    assert crypto_from_equity["share_sum"] == pytest.approx(0.20 + 0.20)
    assert crypto_from_equity["average_receiver_share"] == pytest.approx((0.20 + 0.20) / 2)
    assert crypto_from_equity["pair_count"] == 2
    assert crypto_from_equity["average_pair_share"] == pytest.approx((0.20 + 0.20) / 2)

    assert equity_from_crypto["share_sum"] == pytest.approx(0.30 + 0.10)
    assert equity_from_crypto["average_receiver_share"] == pytest.approx(0.30 + 0.10)
    assert equity_from_crypto["pair_count"] == 2
    assert equity_from_crypto["average_pair_share"] == pytest.approx((0.30 + 0.10) / 2)

    assert equity_from_equity["share_sum"] == pytest.approx(0.0)
    assert equity_from_equity["average_receiver_share"] == pytest.approx(0.0)
    assert equity_from_equity["pair_count"] == 0
    assert equity_from_equity["average_pair_share"] == pytest.approx(0.0)


def test_block_matrix_uses_average_receiver_share_and_table_order() -> None:
    table = pd.DataFrame(
        [
            {"receiver_block": "crypto", "shock_block": "crypto", "average_receiver_share": 0.15},
            {"receiver_block": "crypto", "shock_block": "equity", "average_receiver_share": 0.20},
            {"receiver_block": "equity", "shock_block": "crypto", "average_receiver_share": 0.40},
            {"receiver_block": "equity", "shock_block": "equity", "average_receiver_share": 0.00},
        ]
    )

    matrix = block_matrix(table)

    assert list(matrix.index) == ["crypto", "equity"]
    assert list(matrix.columns) == ["crypto", "equity"]
    assert matrix.loc["crypto", "crypto"] == pytest.approx(0.15)
    assert matrix.loc["crypto", "equity"] == pytest.approx(0.20)
    assert matrix.loc["equity", "crypto"] == pytest.approx(0.40)
    assert matrix.loc["equity", "equity"] == pytest.approx(0.00)


def test_block_matrix_can_use_pair_adjusted_share() -> None:
    table = pd.DataFrame(
        [
            {
                "receiver_block": "crypto",
                "shock_block": "equity",
                "average_receiver_share": 0.20,
                "average_pair_share": 0.10,
            },
            {
                "receiver_block": "equity",
                "shock_block": "crypto",
                "average_receiver_share": 0.40,
                "average_pair_share": 0.20,
            },
        ]
    )

    matrix = block_matrix(table, value_col="average_pair_share")

    assert matrix.loc["crypto", "equity"] == pytest.approx(0.10)
    assert matrix.loc["equity", "crypto"] == pytest.approx(0.20)


def test_period_gfevd_outputs_register_block_files(tmp_path) -> None:
    registry = ArtifactRegistry(tmp_path)

    _register_gfevd_outputs(registry, "1d", "pre_covid", group="periods")

    paths = {entry.path: entry.kind for entry in registry.entries}
    assert paths["gfevd\\periods\\block_spillovers_pre_covid_1d.csv"] == "csv"
    assert paths["gfevd\\periods\\block_matrix_pre_covid_1d.csv"] == "csv"
    assert paths["gfevd\\periods\\block_matrix_total_pre_covid_1d.csv"] == "csv"
    assert paths["gfevd\\periods\\block_matrix_adjusted_pre_covid_1d.csv"] == "csv"
    assert paths["gfevd\\periods\\metadata_pre_covid_1d.json"] == "json"


def _row(table: pd.DataFrame, *, receiver_block: str, shock_block: str) -> pd.Series:
    rows = table[
        table["receiver_block"].eq(receiver_block)
        & table["shock_block"].eq(shock_block)
    ]
    assert len(rows) == 1
    return rows.iloc[0]
