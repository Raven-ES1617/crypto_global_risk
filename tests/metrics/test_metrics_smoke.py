from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    calculate_gfevd,
    calculate_gis,
    calculate_hasbrouck_proxy,
    calculate_pairwise_hasbrouck_proxy,
)


def _prices(n_obs: int = 360, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(scale=0.01, size=n_obs))
    noise = rng.normal(scale=[0.005, 0.007, 0.009], size=(n_obs, 3))
    log_prices = trend[:, None] + noise
    return pd.DataFrame(np.exp(log_prices), columns=["BTC", "ETH", "SPX"])


def test_gis_sums_to_one() -> None:
    result = calculate_gis(_prices(), max_lags=4)
    assert result.shares["GIS"].between(0, 1).all()
    assert np.isclose(result.shares["GIS"].sum(), 1.0)


def test_gfevd_rows_sum_to_one() -> None:
    result = calculate_gfevd(_prices(), horizon=12, max_lags=4)
    assert result.table.shape == (3, 3)
    assert np.allclose(result.table.sum(axis=1).to_numpy(), 1.0)


def test_hasbrouck_proxy_bounds_are_valid() -> None:
    result = calculate_hasbrouck_proxy(_prices(), max_lags=4, max_orderings=12)
    assert result.summary["lower"].between(0, 1).all()
    assert result.summary["upper"].between(0, 1).all()
    assert (result.summary["lower"] <= result.summary["upper"]).all()
    assert np.isclose(result.summary["mean"].sum(), 1.0)


def test_pairwise_hasbrouck_proxy_shape() -> None:
    result = calculate_pairwise_hasbrouck_proxy(_prices(), max_lags=4)
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result), 0.5)
