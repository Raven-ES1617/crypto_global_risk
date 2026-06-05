from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

LagMethod = Literal["aic", "bic", "hqic", "fpe"]
RankChoice = int | Literal["auto", "single_common_trend"]


@dataclass(frozen=True)
class PreparedPrices:
    log_prices: pd.DataFrame
    assets: list[str]


def prepare_log_prices(
    price_data: pd.DataFrame,
    *,
    input_is_log: bool = False,
    max_obs: int | None = None,
    min_obs: int = 40,
) -> PreparedPrices:
    df = pd.DataFrame(price_data).copy()
    if df.empty:
        raise ValueError("price_data is empty")

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = _downsample_evenly(df, max_obs=max_obs)

    if not input_is_log:
        if (df <= 0).any().any():
            raise ValueError("price_data must be strictly positive unless input_is_log=True")
        log_prices = np.log(df)
    else:
        log_prices = df

    varying = [
        col
        for col in log_prices.columns
        if log_prices[col].nunique(dropna=True) > 1
        and float(log_prices[col].std(ddof=0)) > 0.0
    ]
    log_prices = log_prices[varying].dropna(how="any")

    if len(log_prices) < min_obs:
        raise ValueError(f"not enough aligned observations: got {len(log_prices)}, need {min_obs}")
    if log_prices.shape[1] < 2:
        raise ValueError("at least two non-constant series are required")

    log_prices.columns = [str(col) for col in log_prices.columns]
    return PreparedPrices(log_prices=log_prices, assets=list(log_prices.columns))


def _downsample_evenly(data: pd.DataFrame, *, max_obs: int | None) -> pd.DataFrame:
    if max_obs is None or len(data) <= max_obs:
        return data
    if max_obs < 2:
        raise ValueError("max_obs must be at least 2")
    positions = np.linspace(0, len(data) - 1, num=max_obs, dtype=int)
    positions = np.unique(positions)
    return data.iloc[positions].dropna(how="any")


def select_var_lag(
    log_prices: pd.DataFrame,
    *,
    max_lags: int = 10,
    method: LagMethod = "aic",
    fallback: int = 1,
) -> int:
    diff = log_prices.diff().dropna()
    if len(diff) < 10:
        return max(1, int(fallback))

    max_allowed = max(1, min(int(max_lags), len(diff) // 3))
    try:
        selected = VAR(diff).select_order(maxlags=max_allowed)
        lag = getattr(selected, method)
    except Exception:
        lag = fallback

    if lag is None or int(lag) < 1:
        return max(1, int(fallback))
    return int(lag)


def estimate_johansen_rank(
    log_prices: pd.DataFrame,
    *,
    k_ar_diff: int,
    det_order: int = 0,
    significance: Literal["90%", "95%", "99%"] = "95%",
    fallback: int | None = None,
) -> int:
    n = log_prices.shape[1]
    if n > 12:
        if fallback is None:
            raise ValueError("Johansen critical values are not available for more than 12 series")
        return _clip_rank(fallback, n)

    col = {"90%": 0, "95%": 1, "99%": 2}[significance]
    try:
        joh = coint_johansen(log_prices, det_order=det_order, k_ar_diff=int(k_ar_diff))
        critical = np.asarray(joh.cvt[:, col], dtype=float)
        if not np.isfinite(critical).all():
            raise ValueError("Johansen critical values contain non-finite entries")
        rank = int(np.sum(np.asarray(joh.lr1, dtype=float) > critical))
    except Exception:
        if fallback is None:
            raise
        rank = int(fallback)

    return _clip_rank(rank, n)


def resolve_rank(
    choice: RankChoice,
    log_prices: pd.DataFrame,
    *,
    k_ar_diff: int,
    det_order: int,
    fallback: int,
    require_single_common_trend: bool,
) -> int:
    n = log_prices.shape[1]

    if choice == "single_common_trend":
        rank = n - 1
    elif choice == "auto":
        rank = estimate_johansen_rank(
            log_prices,
            k_ar_diff=k_ar_diff,
            det_order=det_order,
            fallback=fallback,
        )
    else:
        rank = int(choice)

    rank = _clip_rank(rank, n)
    if require_single_common_trend and rank != n - 1:
        raise ValueError(
            "this metric requires one common stochastic trend, so coint_rank must be n_assets - 1; "
            f"got coint_rank={rank}, n_assets={n}"
        )
    return rank


def _clip_rank(rank: int, n_assets: int) -> int:
    return max(0, min(int(rank), n_assets - 1))


def fit_vecm(
    log_prices: pd.DataFrame,
    *,
    k_ar_diff: int,
    coint_rank: int,
    deterministic: str,
):
    effective = _deterministic_for_rank(deterministic, coint_rank)
    model = VECM(
        log_prices,
        k_ar_diff=int(k_ar_diff),
        coint_rank=int(coint_rank),
        deterministic=effective,
    )
    return model.fit(), effective


def _deterministic_for_rank(deterministic: str, coint_rank: int) -> str:
    if coint_rank > 0:
        return deterministic
    if deterministic == "ci":
        return "co"
    if deterministic == "li":
        return "lo"
    return deterministic


def residual_covariance(vecm_result) -> np.ndarray:
    sigma = getattr(vecm_result, "sigma_u", None)
    if sigma is None:
        sigma = np.cov(np.asarray(vecm_result.resid).T)
    return _symmetrize(np.atleast_2d(np.asarray(sigma, dtype=float)))


def ma_representation(var_mats: np.ndarray, *, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be positive")

    mats = np.asarray(var_mats, dtype=float)
    if mats.ndim != 3:
        raise ValueError("var_mats must have shape (p, n, n)")

    p, n, _ = mats.shape
    psi = np.zeros((int(horizon), n, n), dtype=float)
    psi[0] = np.eye(n)
    for h in range(1, int(horizon)):
        for lag in range(1, min(h, p) + 1):
            psi[h] += mats[lag - 1] @ psi[h - lag]
    return psi


def long_run_impact_matrix(vecm_result, *, coint_rank: int) -> np.ndarray:
    if coint_rank <= 0:
        raise ValueError("long-run price-discovery impact requires coint_rank > 0")

    alpha = np.asarray(vecm_result.alpha[:, :coint_rank], dtype=float)
    beta = np.asarray(vecm_result.beta[:, :coint_rank], dtype=float)
    n = alpha.shape[0]

    alpha_perp = null_space(alpha.T)
    beta_perp = null_space(beta.T)
    if alpha_perp.shape[1] == 0 or beta_perp.shape[1] == 0:
        raise ValueError("could not compute alpha_perp/beta_perp; cointegration rank is too high")

    gamma_bar = np.eye(n) - _sum_vecm_gamma(vecm_result, n)
    middle = alpha_perp.T @ gamma_bar @ beta_perp
    middle_inv = np.linalg.pinv(middle)
    return beta_perp @ middle_inv @ alpha_perp.T


def _sum_vecm_gamma(vecm_result, n_assets: int) -> np.ndarray:
    gamma = getattr(vecm_result, "gamma", None)
    if gamma is None:
        return np.zeros((n_assets, n_assets))

    gamma = np.asarray(gamma, dtype=float)
    if gamma.size == 0:
        return np.zeros((n_assets, n_assets))

    total = np.zeros((n_assets, n_assets))
    blocks = gamma.shape[1] // n_assets
    for idx in range(blocks):
        total += gamma[:, idx * n_assets : (idx + 1) * n_assets]
    return total


def null_space(matrix: np.ndarray, *, rtol: float = 1e-10) -> np.ndarray:
    arr = np.atleast_2d(np.asarray(matrix, dtype=float))
    _, singular_values, vh = np.linalg.svd(arr, full_matrices=True)
    if singular_values.size == 0:
        rank = 0
    else:
        tol = rtol * max(arr.shape) * singular_values[0]
        rank = int(np.sum(singular_values > tol))
    return vh[rank:].T.copy()


def generalized_fevd(psi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    sigma = _symmetrize(np.asarray(sigma, dtype=float))
    _, n, _ = psi.shape

    numerator = np.zeros((n, n), dtype=float)
    denominator = np.zeros(n, dtype=float)
    sigma_diag = np.diag(sigma)
    valid_diag = sigma_diag > 0

    for ph in psi:
        cross = ph @ sigma
        numerator[:, valid_diag] += (cross[:, valid_diag] ** 2) / sigma_diag[valid_diag]
        denominator += np.diag(ph @ sigma @ ph.T)

    theta = np.zeros((n, n), dtype=float)
    valid_den = denominator > 0
    theta[valid_den, :] = numerator[valid_den, :] / denominator[valid_den, None]
    return theta


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    sums = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr)
    valid = sums[:, 0] > 0
    out[valid] = arr[valid] / sums[valid]
    return out


def modified_correlation_sqrt(sigma: np.ndarray) -> np.ndarray:
    sigma = _symmetrize(np.asarray(sigma, dtype=float))
    variances = np.diag(sigma)
    if np.any(variances <= 0):
        raise ValueError("residual covariance must have positive diagonal entries")

    std = np.sqrt(variances)
    inv_std = np.diag(1.0 / std)
    corr = _symmetrize(inv_std @ sigma @ inv_std)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return np.diag(std) @ eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def symmetric_sqrt(sigma: np.ndarray) -> np.ndarray:
    sigma = _symmetrize(np.asarray(sigma, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def safe_cholesky(sigma: np.ndarray) -> np.ndarray:
    sigma = _symmetrize(np.asarray(sigma, dtype=float))
    jitter = 0.0
    eye = np.eye(sigma.shape[0])
    for _ in range(8):
        try:
            return np.linalg.cholesky(sigma + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
    return np.linalg.cholesky(sigma + jitter * eye)


def column_shares(effect_matrix: np.ndarray) -> np.ndarray:
    effects = np.asarray(effect_matrix, dtype=float)
    numerators = np.sum(effects**2, axis=0)
    total = float(np.sum(numerators))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("cannot normalize zero long-run shock contributions")
    return numerators / total


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0
