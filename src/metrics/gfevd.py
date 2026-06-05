from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics.common import (
    LagMethod,
    RankChoice,
    fit_vecm,
    generalized_fevd,
    ma_representation,
    prepare_log_prices,
    residual_covariance,
    resolve_rank,
    row_normalize,
    select_var_lag,
)


@dataclass(frozen=True)
class GFEVDResult:
    table: pd.DataFrame
    raw: pd.DataFrame
    connectedness: pd.DataFrame
    total_connectedness: float
    sigma: np.ndarray
    ma_matrices: np.ndarray
    coint_rank: int
    lag_order_diff: int
    deterministic: str
    assets: list[str]
    horizon: int

    def as_dict(self) -> dict[str, object]:
        return {
            "gfevd": self.table,
            "raw_gfevd": self.raw,
            "connectedness": self.connectedness,
            "total_connectedness": self.total_connectedness,
            "sigma": self.sigma,
            "psi": self.ma_matrices,
            "coint_rank": self.coint_rank,
            "lag_order_diff": self.lag_order_diff,
            "deterministic": self.deterministic,
            "assets": self.assets,
            "horizon": self.horizon,
        }


def calculate_gfevd(
    price_data: pd.DataFrame,
    *,
    horizon: int = 20,
    max_lags: int = 10,
    lag_method: LagMethod = "aic",
    coint_rank: RankChoice = "auto",
    det_order: int = 0,
    deterministic: str = "ci",
    input_is_log: bool = False,
    max_obs: int | None = None,
) -> GFEVDResult:
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
        fallback=0,
        require_single_common_trend=False,
    )

    vecm_result, effective_deterministic = fit_vecm(
        log_prices,
        k_ar_diff=lag_order,
        coint_rank=rank,
        deterministic=deterministic,
    )
    sigma = residual_covariance(vecm_result)
    ma_mats = ma_representation(vecm_result.var_rep, horizon=horizon)

    raw = generalized_fevd(ma_mats, sigma)
    normalized = row_normalize(raw)

    raw_df = pd.DataFrame(raw, index=assets, columns=assets)
    table = pd.DataFrame(normalized, index=assets, columns=assets)
    connectedness, total_connectedness = _connectedness(table)

    return GFEVDResult(
        table=table,
        raw=raw_df,
        connectedness=connectedness,
        total_connectedness=total_connectedness,
        sigma=sigma,
        ma_matrices=ma_mats,
        coint_rank=rank,
        lag_order_diff=lag_order,
        deterministic=effective_deterministic,
        assets=assets,
        horizon=int(horizon),
    )


def _connectedness(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    off_diag = table.copy()
    values = off_diag.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    off_diag = pd.DataFrame(values, index=table.index, columns=table.columns)

    from_others = off_diag.sum(axis=1)
    to_others = off_diag.sum(axis=0)
    net = to_others - from_others
    result = pd.DataFrame(
        {
            "from_others": from_others,
            "to_others": to_others,
            "net": net,
        }
    )
    total = float(values.sum() / len(table.index))
    return result, total
