from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd

from metrics.common import (
    LagMethod,
    RankChoice,
    column_shares,
    fit_vecm,
    long_run_impact_matrix,
    prepare_log_prices,
    residual_covariance,
    resolve_rank,
    safe_cholesky,
    select_var_lag,
)


@dataclass(frozen=True)
class HasbrouckProxyResult:
    summary: pd.DataFrame
    order_shares: pd.DataFrame
    long_run_impact: np.ndarray
    sigma: np.ndarray
    coint_rank: int
    lag_order_diff: int
    deterministic: str
    assets: list[str]
    n_orderings: int

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "order_shares": self.order_shares,
            "long_run_impact": self.long_run_impact,
            "sigma": self.sigma,
            "coint_rank": self.coint_rank,
            "lag_order_diff": self.lag_order_diff,
            "deterministic": self.deterministic,
            "assets": self.assets,
            "n_orderings": self.n_orderings,
        }


def calculate_hasbrouck_proxy(
    price_data: pd.DataFrame,
    *,
    max_lags: int = 10,
    lag_method: LagMethod = "bic",
    coint_rank: RankChoice = "single_common_trend",
    det_order: int = 0,
    deterministic: str = "ci",
    max_orderings: int | None = 720,
    random_state: int | None = 42,
    input_is_log: bool = False,
    max_obs: int | None = None,
) -> HasbrouckProxyResult:
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

    orderings = _build_orderings(len(assets), max_orderings=max_orderings, random_state=random_state)
    share_rows = []
    order_labels = []
    for order in orderings:
        share_rows.append(_shares_for_order(long_run, sigma, order))
        order_labels.append(">".join(assets[idx] for idx in order))

    order_shares = pd.DataFrame(share_rows, index=order_labels, columns=assets)
    lower = order_shares.min(axis=0)
    upper = order_shares.max(axis=0)
    summary = pd.DataFrame(
        {
            "lower": lower,
            "upper": upper,
            "midpoint": (lower + upper) / 2.0,
            "mean": order_shares.mean(axis=0),
            "std": order_shares.std(axis=0, ddof=0),
        }
    ).sort_values("midpoint", ascending=False)
    summary["Rank"] = np.arange(1, len(summary) + 1)

    return HasbrouckProxyResult(
        summary=summary,
        order_shares=order_shares,
        long_run_impact=long_run,
        sigma=sigma,
        coint_rank=rank,
        lag_order_diff=lag_order,
        deterministic=effective_deterministic,
        assets=assets,
        n_orderings=len(orderings),
    )


def calculate_pairwise_hasbrouck_proxy(
    price_data: pd.DataFrame,
    *,
    max_lags: int = 10,
    lag_method: LagMethod = "bic",
    det_order: int = 0,
    deterministic: str = "ci",
    input_is_log: bool = False,
    max_obs: int | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(price_data)
    assets = [str(col) for col in df.columns]
    out = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    for asset in assets:
        out.loc[asset, asset] = 0.5

    for i, asset_i in enumerate(assets):
        for asset_j in assets[i + 1 :]:
            result = calculate_hasbrouck_proxy(
                df[[asset_i, asset_j]],
                max_lags=max_lags,
                lag_method=lag_method,
                coint_rank="single_common_trend",
                det_order=det_order,
                deterministic=deterministic,
                max_orderings=None,
                input_is_log=input_is_log,
                max_obs=max_obs,
            )
            share_i = float(result.summary.loc[asset_i, "midpoint"])
            out.loc[asset_i, asset_j] = share_i
            out.loc[asset_j, asset_i] = 1.0 - share_i
    return out


def _build_orderings(
    n_assets: int,
    *,
    max_orderings: int | None,
    random_state: int | None,
) -> list[tuple[int, ...]]:
    total = factorial(n_assets)
    if max_orderings is None or total <= max_orderings:
        return list(permutations(range(n_assets)))

    rng = np.random.default_rng(random_state)
    selected: set[tuple[int, ...]] = {tuple(range(n_assets)), tuple(reversed(range(n_assets)))}
    while len(selected) < int(max_orderings):
        selected.add(tuple(rng.permutation(n_assets).tolist()))
    return sorted(selected)


def _shares_for_order(long_run: np.ndarray, sigma: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    order_idx = np.asarray(order, dtype=int)
    sigma_ordered = sigma[np.ix_(order_idx, order_idx)]
    chol = safe_cholesky(sigma_ordered)
    effects_ordered = long_run[:, order_idx] @ chol
    shares_ordered = column_shares(effects_ordered)

    shares = np.empty_like(shares_ordered)
    shares[order_idx] = shares_ordered
    return shares
