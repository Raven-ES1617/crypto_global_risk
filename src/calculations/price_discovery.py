from __future__ import annotations

import pandas as pd

from calculations.config import ASSET_BLOCKS, PRICE_DISCOVERY_PAIRS, FrequencyConfig
from metrics import calculate_gis, calculate_hasbrouck_proxy


def run_pairwise_price_discovery(
    prices: pd.DataFrame,
    *,
    frequency: str,
    config: FrequencyConfig,
    pairs: tuple[tuple[str, str], ...] = PRICE_DISCOVERY_PAIRS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        if left not in prices.columns or right not in prices.columns:
            continue
        pair_prices = prices[[left, right]].dropna(how="any")
        common = {
            "frequency": frequency,
            "left_asset": left,
            "right_asset": right,
            "pair": f"{left}-{right}",
            "rows_available": int(len(pair_prices)),
        }
        if len(pair_prices) < 80:
            rows.append({**common, "metric": "hasbrouck_proxy", "status": "error", "error": "not enough rows"})
            rows.append({**common, "metric": "gis", "status": "error", "error": "not enough rows"})
            continue

        try:
            hb = calculate_hasbrouck_proxy(
                pair_prices,
                max_lags=config.max_lags,
                lag_method="bic",
                max_orderings=None,
                max_obs=config.max_obs,
            )
            rows.append(
                {
                    **common,
                    "metric": "hasbrouck_proxy",
                    "status": "ok",
                    "left_share": float(hb.summary.loc[left, "midpoint"]),
                    "right_share": float(hb.summary.loc[right, "midpoint"]),
                    "left_lower": float(hb.summary.loc[left, "lower"]),
                    "left_upper": float(hb.summary.loc[left, "upper"]),
                    "right_lower": float(hb.summary.loc[right, "lower"]),
                    "right_upper": float(hb.summary.loc[right, "upper"]),
                    "left_bound_width": float(hb.summary.loc[left, "upper"] - hb.summary.loc[left, "lower"]),
                    "right_bound_width": float(hb.summary.loc[right, "upper"] - hb.summary.loc[right, "lower"]),
                    "n_orderings": int(hb.n_orderings),
                    "lag_order_diff": hb.lag_order_diff,
                    "coint_rank": hb.coint_rank,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append({**common, "metric": "hasbrouck_proxy", "status": "error", "error": str(exc)})

        try:
            gis = calculate_gis(
                pair_prices,
                max_lags=config.max_lags,
                lag_method="bic",
                max_obs=config.max_obs,
            )
            rows.append(
                {
                    **common,
                    "metric": "gis",
                    "status": "ok",
                    "left_share": float(gis.shares.loc[left, "GIS"]),
                    "right_share": float(gis.shares.loc[right, "GIS"]),
                    "left_lower": pd.NA,
                    "left_upper": pd.NA,
                    "right_lower": pd.NA,
                    "right_upper": pd.NA,
                    "left_bound_width": pd.NA,
                    "right_bound_width": pd.NA,
                    "n_orderings": pd.NA,
                    "lag_order_diff": gis.lag_order_diff,
                    "coint_rank": gis.coint_rank,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append({**common, "metric": "gis", "status": "error", "error": str(exc)})

    return pd.DataFrame(rows)


def aggregate_price_discovery_blocks(pairwise_windows: pd.DataFrame) -> pd.DataFrame:
    """Average pairwise long-run shares by asset blocks inside each window."""
    required = {
        "frequency",
        "period",
        "window_id",
        "window_start",
        "window_end",
        "metric",
        "left_asset",
        "right_asset",
        "pair",
        "status",
        "left_share",
    }
    if pairwise_windows.empty or not required.issubset(pairwise_windows.columns):
        return pd.DataFrame()

    ok = pairwise_windows[pairwise_windows["status"].eq("ok")].copy()
    ok["left_share"] = pd.to_numeric(ok["left_share"], errors="coerce")
    ok = ok.dropna(subset=["left_share"])
    if ok.empty:
        return pd.DataFrame()

    ok["left_block"] = ok["left_asset"].map(lambda asset: ASSET_BLOCKS.get(str(asset), "unknown"))
    ok["right_block"] = ok["right_asset"].map(lambda asset: ASSET_BLOCKS.get(str(asset), "unknown"))
    ok = ok[ok["left_block"].ne(ok["right_block"])]
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "frequency",
        "period",
        "window_id",
        "window_start",
        "window_end",
        "metric",
        "left_block",
        "right_block",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "left_share": ("left_share", "mean"),
        "pair_count": ("pair", "nunique"),
    }
    optional_cols = (
        "left_lower",
        "left_upper",
        "left_bound_width",
        "right_lower",
        "right_upper",
        "right_bound_width",
        "n_orderings",
    )
    for col in optional_cols:
        if col in ok.columns:
            aggregations[col] = (col, "mean")

    return ok.groupby(group_cols, dropna=False).agg(**aggregations).reset_index()
