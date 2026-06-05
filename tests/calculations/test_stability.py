from __future__ import annotations

import pandas as pd
import pytest

from calculations.stability import (
    add_diff_stability,
    add_price_discovery_dominance,
    block_net_flows,
    crypto_global_flows,
    frequency_consistency,
    metric_agreement,
    window_block_net_flows,
    window_crypto_global_flows,
)
from calculations.run_stability_artifacts import _block_ci_labels, _block_ci_mean_matrix


def test_block_net_flows_use_columns_as_sources_and_rows_as_receivers() -> None:
    matrix = pd.DataFrame(
        {
            "crypto": [0.0, 0.20, 0.10],
            "equity_index": [0.05, 0.0, 0.03],
            "fx_dollar": [0.07, 0.04, 0.0],
        },
        index=["crypto", "equity_index", "fx_dollar"],
    )

    result = block_net_flows(matrix).set_index("block")

    assert result.loc["crypto", "to_others"] == pytest.approx(0.30)
    assert result.loc["crypto", "from_others"] == pytest.approx(0.12)
    assert result.loc["crypto", "net_to_others"] == pytest.approx(0.18)


def test_crypto_global_flows_keep_direction_explicit() -> None:
    matrix = pd.DataFrame(
        {
            "crypto": [0.0, 0.20, 0.10],
            "equity_index": [0.05, 0.0, 0.03],
            "fx_dollar": [0.07, 0.04, 0.0],
        },
        index=["crypto", "equity_index", "fx_dollar"],
    )

    result = crypto_global_flows(matrix)

    assert result["global_to_crypto"] == pytest.approx(0.12)
    assert result["crypto_to_global"] == pytest.approx(0.30)
    assert result["net_crypto_to_global"] == pytest.approx(0.18)


def test_window_flow_helpers_build_rolling_net_tables() -> None:
    windows = pd.DataFrame(
        [
            {
                "frequency": "1d",
                "window_id": 1,
                "window_start": "2020-01-01T00:00:00+00:00",
                "window_end": "2020-02-01T00:00:00+00:00",
                "receiver_block": "crypto",
                "shock_block": "crypto",
                "average_receiver_share": 0.0,
            },
            {
                "frequency": "1d",
                "window_id": 1,
                "window_start": "2020-01-01T00:00:00+00:00",
                "window_end": "2020-02-01T00:00:00+00:00",
                "receiver_block": "crypto",
                "shock_block": "equity_index",
                "average_receiver_share": 0.1,
            },
            {
                "frequency": "1d",
                "window_id": 1,
                "window_start": "2020-01-01T00:00:00+00:00",
                "window_end": "2020-02-01T00:00:00+00:00",
                "receiver_block": "equity_index",
                "shock_block": "crypto",
                "average_receiver_share": 0.3,
            },
            {
                "frequency": "1d",
                "window_id": 1,
                "window_start": "2020-01-01T00:00:00+00:00",
                "window_end": "2020-02-01T00:00:00+00:00",
                "receiver_block": "equity_index",
                "shock_block": "equity_index",
                "average_receiver_share": 0.0,
            },
        ]
    )

    net = window_block_net_flows(windows, value_col="average_receiver_share", variant="total").set_index("block")
    crypto = window_crypto_global_flows(windows, value_col="average_receiver_share", variant="total").iloc[0]

    assert net.loc["crypto", "net_to_others"] == pytest.approx(0.2)
    assert crypto["global_to_crypto"] == pytest.approx(0.1)
    assert crypto["crypto_to_global"] == pytest.approx(0.3)


def test_price_discovery_dominance_classifies_ci_against_parity() -> None:
    frame = pd.DataFrame(
        {
            "mean": [0.7, 0.3, 0.51],
            "q025": [0.6, 0.1, 0.4],
            "q975": [0.9, 0.4, 0.6],
            "n_windows": [10, 10, 10],
        }
    )

    result = add_price_discovery_dominance(frame)

    assert result["dominance"].tolist() == ["left_dominates", "right_dominates", "parity_or_mixed"]
    assert result["stable_dominance"].tolist() == [True, True, False]


def test_diff_stability_requires_p_value_and_ci_excluding_zero() -> None:
    frame = pd.DataFrame(
        {
            "diff_mean": [0.2, 0.2],
            "diff_ci_low": [0.1, -0.1],
            "diff_ci_high": [0.3, 0.4],
            "p_value": [0.01, 0.01],
        }
    )

    result = add_diff_stability(frame)

    assert result["robust_change"].tolist() == [True, False]


def test_frequency_consistency_counts_signs_by_effect() -> None:
    frame = pd.DataFrame(
        {
            "frequency": ["1min", "1h", "1d", "1min", "1h", "1d"],
            "effect": ["a", "a", "a", "b", "b", "b"],
            "value": [0.1, 0.2, 0.3, 0.1, -0.2, 0.3],
        }
    )

    result = frequency_consistency(frame, group_cols=["effect"], value_col="value", label="test").set_index("effect")

    assert bool(result.loc["a", "consistent_nonzero_sign"]) is True
    assert bool(result.loc["b", "consistent_nonzero_sign"]) is False


def test_metric_agreement_compares_gis_and_hasbrouck_around_parity() -> None:
    frame = pd.DataFrame(
        {
            "frequency": ["1d", "1d", "1d", "1d"],
            "pair": ["BTC-SPX", "BTC-SPX", "BTC-GOLD", "BTC-GOLD"],
            "metric": ["gis", "hasbrouck_proxy", "gis", "hasbrouck_proxy"],
            "mean": [0.7, 0.6, 0.7, 0.4],
        }
    )

    result = metric_agreement(frame, group_cols=["frequency", "pair"]).set_index("pair")

    assert bool(result.loc["BTC-SPX", "same_side_of_parity"]) is True
    assert bool(result.loc["BTC-GOLD", "same_side_of_parity"]) is False


def test_frl_block_ci_labels_use_same_mean_as_heatmap() -> None:
    template = pd.DataFrame({"crypto": [0.50]}, index=["crypto"])
    ci = pd.DataFrame(
        {
            "receiver_block": ["crypto"],
            "shock_block": ["crypto"],
            "n_windows": [4],
            "mean": [0.10],
            "std": [0.02],
            "q025": [0.01],
            "q975": [0.20],
        }
    )

    matrix = _block_ci_mean_matrix(ci, template)
    labels = _block_ci_labels(matrix, ci)

    assert matrix.loc["crypto", "crypto"] == pytest.approx(0.10)
    assert labels is not None
    assert labels.loc["crypto", "crypto"] == "0.10\n[0.08,0.12]"
