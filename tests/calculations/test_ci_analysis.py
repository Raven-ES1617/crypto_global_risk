from __future__ import annotations

import pandas as pd
import pytest

from calculations.ci_analysis import compare_ci_summaries, summarize_ci


def test_summarize_ci_returns_quantiles_by_group() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0],
            "status": ["ok", "ok", "ok", "ok", "error"],
        }
    )

    result = summarize_ci(frame, ["group"], "value")

    a = result[result["group"].eq("a")].iloc[0]
    b = result[result["group"].eq("b")].iloc[0]
    assert a["n_windows"] == 3
    assert a["median"] == pytest.approx(2.0)
    assert b["n_windows"] == 1
    assert b["mean"] == pytest.approx(10.0)


def test_compare_ci_summaries_marks_clear_post_pre_difference() -> None:
    pre = pd.DataFrame(
        {
            "receiver_block": ["crypto"],
            "shock_block": ["equity"],
            "mean": [0.10],
            "std": [0.01],
            "n_windows": [16],
        }
    )
    post = pd.DataFrame(
        {
            "receiver_block": ["crypto"],
            "shock_block": ["equity"],
            "mean": [0.20],
            "std": [0.01],
            "n_windows": [16],
        }
    )

    result = compare_ci_summaries(pre, post, ["receiver_block", "shock_block"])

    row = result.iloc[0]
    assert row["diff_mean"] == pytest.approx(0.10)
    assert row["p_value"] < 0.001
    assert row["stars"] == "***"
