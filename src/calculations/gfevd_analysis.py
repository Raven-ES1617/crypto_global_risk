from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calculations.config import ASSET_BLOCKS, BLOCK_ORDER, FrequencyConfig
from metrics import calculate_gfevd


@dataclass(frozen=True)
class GFEVDRunMetadata:
    frequency: str
    period: str
    rows_available: int
    rows_used_limit: int | None
    horizon: int
    max_lags: int
    lag_order_diff: int
    coint_rank: int
    total_connectedness: float


def run_gfevd_for_panel(
    prices: pd.DataFrame,
    *,
    frequency: str,
    period: str,
    config: FrequencyConfig,
    output_dir: Path,
) -> GFEVDRunMetadata:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = calculate_gfevd(
        prices,
        horizon=config.horizon,
        max_lags=config.max_lags,
        lag_method="bic",
        coint_rank="auto",
        max_obs=config.max_obs,
    )

    stem = f"{period}_{frequency}"
    result.table.to_csv(output_dir / f"matrix_{stem}.csv")
    result.raw.to_csv(output_dir / f"raw_matrix_{stem}.csv")
    result.connectedness.to_csv(output_dir / f"asset_connectedness_{stem}.csv")
    block_table = block_spillover_table(result.table)
    block_table.to_csv(output_dir / f"block_spillovers_{stem}.csv", index=False)
    total_matrix = block_matrix(block_table, value_col="average_receiver_share")
    adjusted_matrix = block_matrix(block_table, value_col="average_pair_share")
    total_matrix.to_csv(output_dir / f"block_matrix_{stem}.csv")
    total_matrix.to_csv(output_dir / f"block_matrix_total_{stem}.csv")
    adjusted_matrix.to_csv(output_dir / f"block_matrix_adjusted_{stem}.csv")

    metadata = GFEVDRunMetadata(
        frequency=frequency,
        period=period,
        rows_available=int(len(prices)),
        rows_used_limit=config.max_obs,
        horizon=result.horizon,
        max_lags=config.max_lags,
        lag_order_diff=result.lag_order_diff,
        coint_rank=result.coint_rank,
        total_connectedness=result.total_connectedness,
    )
    pd.Series(asdict(metadata)).to_json(output_dir / f"metadata_{stem}.json", force_ascii=False, indent=2)
    return metadata


def block_spillover_table(
    gfevd: pd.DataFrame,
    *,
    block_map: dict[str, str] | None = None,
    block_order: tuple[str, ...] = BLOCK_ORDER,
) -> pd.DataFrame:
    block_map = block_map or ASSET_BLOCKS
    rows: list[dict[str, object]] = []
    values = gfevd.copy()
    values.index = values.index.astype(str)
    values.columns = values.columns.astype(str)

    for receiver_block in block_order:
        receivers = [asset for asset in values.index if block_map.get(asset) == receiver_block]
        if not receivers:
            continue
        for shock_block in block_order:
            shocks = [asset for asset in values.columns if block_map.get(asset) == shock_block]
            if not shocks:
                continue
            sub = values.loc[receivers, shocks].copy()
            if receiver_block == shock_block:
                for asset in set(receivers).intersection(shocks):
                    sub.loc[asset, asset] = 0.0
            share_sum = float(sub.to_numpy().sum())
            pair_count = _block_pair_count(receiver_block, shock_block, receivers, shocks)
            rows.append(
                {
                    "receiver_block": receiver_block,
                    "shock_block": shock_block,
                    "receiver_assets": len(receivers),
                    "shock_assets": len(shocks),
                    "pair_count": pair_count,
                    "share_sum": share_sum,
                    "average_receiver_share": share_sum / len(receivers),
                    "average_pair_share": 0.0 if pair_count == 0 else share_sum / pair_count,
                }
            )
    return pd.DataFrame(rows)


def block_matrix(
    block_table: pd.DataFrame,
    *,
    block_order: tuple[str, ...] | None = None,
    value_col: str = "average_receiver_share",
) -> pd.DataFrame:
    matrix = block_table.pivot(
        index="receiver_block",
        columns="shock_block",
        values=value_col,
    )
    if block_order is None:
        receiver_order = list(dict.fromkeys(block_table["receiver_block"].astype(str)))
        shock_order = list(dict.fromkeys(block_table["shock_block"].astype(str)))
    else:
        receiver_order = list(block_order)
        shock_order = list(block_order)
    return matrix.reindex(index=receiver_order, columns=shock_order)


def _block_pair_count(receiver_block: str, shock_block: str, receivers: list[str], shocks: list[str]) -> int:
    if receiver_block != shock_block:
        return len(receivers) * len(shocks)
    own_pairs = len(set(receivers).intersection(shocks))
    return len(receivers) * len(shocks) - own_pairs


def rolling_gfevd(
    prices: pd.DataFrame,
    *,
    frequency: str,
    config: FrequencyConfig,
    max_windows: int | None = None,
) -> pd.DataFrame:
    if config.rolling_window is None or config.rolling_step is None:
        return pd.DataFrame()

    window = pd.Timedelta(config.rolling_window)
    step = pd.Timedelta(config.rolling_step)
    if prices.empty:
        return pd.DataFrame()

    current_end = prices.index.min() + window
    final_end = prices.index.max()
    rows: list[dict[str, object]] = []
    attempts = 0
    while current_end <= final_end:
        if max_windows is not None and attempts >= max_windows:
            break
        attempts += 1
        current_start = current_end - window
        window_prices = prices.loc[(prices.index > current_start) & (prices.index <= current_end)]
        row: dict[str, object] = {
            "frequency": frequency,
            "window_start": current_start.isoformat(),
            "window_end": current_end.isoformat(),
            "rows_available": int(len(window_prices)),
            "status": "ok",
        }
        try:
            if len(window_prices) < 80:
                raise ValueError("not enough observations in rolling window")
            result = calculate_gfevd(
                window_prices,
                horizon=config.horizon,
                max_lags=max(1, min(config.max_lags, 2)),
                lag_method="bic",
                coint_rank="auto",
                max_obs=config.rolling_max_obs,
            )
            row.update(
                {
                    "total_connectedness": result.total_connectedness,
                    "lag_order_diff": result.lag_order_diff,
                    "coint_rank": result.coint_rank,
                    "error": "",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "total_connectedness": np.nan,
                    "lag_order_diff": np.nan,
                    "coint_rank": np.nan,
                    "error": str(exc),
                }
            )
        rows.append(row)
        current_end += step

    return pd.DataFrame(rows)
