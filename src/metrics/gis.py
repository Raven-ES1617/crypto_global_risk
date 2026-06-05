from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from metrics.common import (
    LagMethod,
    RankChoice,
    column_shares,
    fit_vecm,
    long_run_impact_matrix,
    modified_correlation_sqrt,
    prepare_log_prices,
    residual_covariance,
    resolve_rank,
    select_var_lag,
    symmetric_sqrt,
)


@dataclass(frozen=True)
class GISResult:
    shares: pd.DataFrame
    long_run_impact: np.ndarray
    factor_matrix: np.ndarray
    sigma: np.ndarray
    coint_rank: int
    lag_order_diff: int
    deterministic: str
    assets: list[str]
    factorization: str

    def as_dict(self) -> dict[str, object]:
        return {
            "gis": self.shares,
            "long_run_impact": self.long_run_impact,
            "factor_matrix": self.factor_matrix,
            "sigma": self.sigma,
            "coint_rank": self.coint_rank,
            "lag_order_diff": self.lag_order_diff,
            "deterministic": self.deterministic,
            "assets": self.assets,
            "factorization": self.factorization,
        }


def calculate_gis(
    price_data: pd.DataFrame,
    *,
    max_lags: int = 10,
    lag_method: LagMethod = "bic",
    coint_rank: RankChoice = "single_common_trend",
    det_order: int = 0,
    deterministic: str = "ci",
    factorization: Literal["modified", "symmetric"] = "modified",
    input_is_log: bool = False,
    max_obs: int | None = None,
) -> GISResult:
    prepared = prepare_log_prices(
        price_data,
        input_is_log=input_is_log,
        max_obs=max_obs,
    )
    log_prices = prepared.log_prices
    assets = prepared.assets

    lag_order = select_var_lag(log_prices, max_lags=max_lags, method=lag_method)
    rank = resolve_rank(
        coint_rank,
        log_prices,
        k_ar_diff=lag_order,
        det_order=det_order,
        fallback=len(assets) - 1,
        require_single_common_trend=True,
    )

    vecm_result, effective_deterministic = fit_vecm(
        log_prices,
        k_ar_diff=lag_order,
        coint_rank=rank,
        deterministic=deterministic,
    )
    sigma = residual_covariance(vecm_result)
    long_run = long_run_impact_matrix(vecm_result, coint_rank=rank)

    if factorization == "modified":
        factor = modified_correlation_sqrt(sigma)
    elif factorization == "symmetric":
        factor = symmetric_sqrt(sigma)
    else:
        raise ValueError("factorization must be 'modified' or 'symmetric'")

    effects = long_run @ factor
    shares = column_shares(effects)
    shares_df = (
        pd.DataFrame({"GIS": shares}, index=assets)
        .sort_values("GIS", ascending=False)
        .assign(Rank=lambda df: np.arange(1, len(df) + 1))
    )

    return GISResult(
        shares=shares_df,
        long_run_impact=long_run,
        factor_matrix=factor,
        sigma=sigma,
        coint_rank=rank,
        lag_order_diff=lag_order,
        deterministic=effective_deterministic,
        assets=assets,
        factorization=factorization,
    )
