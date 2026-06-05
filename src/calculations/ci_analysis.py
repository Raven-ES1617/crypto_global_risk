from __future__ import annotations

from dataclasses import dataclass, replace
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from calculations.config import FREQUENCY_CONFIGS, FrequencyConfig
from calculations.gfevd_analysis import block_spillover_table
from calculations.price_discovery import run_pairwise_price_discovery
from metrics import calculate_gfevd


@dataclass(frozen=True)
class WindowCIConfig:
    window: str
    step: str
    min_obs: int
    max_obs: int | None
    max_lags: int


CI_WINDOW_CONFIGS: dict[str, WindowCIConfig] = {
    "1s": WindowCIConfig("5D", "5D", 10_000, 10_000, 2),
    "1min": WindowCIConfig("45D", "30D", 8_000, 8_000, 2),
    "1h": WindowCIConfig("60D", "30D", 500, 5_000, 2),
    "1d": WindowCIConfig("120D", "30D", 80, None, 2),
}


def run_gfevd_window_ci(
    prices: pd.DataFrame,
    *,
    frequency: str,
    period: str,
    max_windows: int | None = None,
) -> dict[str, pd.DataFrame]:
    config = FREQUENCY_CONFIGS[frequency]
    ci_config = CI_WINDOW_CONFIGS[frequency]

    tci_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []

    for window_id, (window_start, window_end, window_prices) in enumerate(
        iter_windows(
            prices,
            window=pd.Timedelta(ci_config.window),
            step=pd.Timedelta(ci_config.step),
            min_obs=ci_config.min_obs,
            max_windows=max_windows,
        ),
        start=1,
    ):
        base = {
            "frequency": frequency,
            "period": period,
            "window_id": window_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "rows_available": int(len(window_prices)),
        }
        try:
            result = calculate_gfevd(
                window_prices,
                horizon=config.horizon,
                max_lags=ci_config.max_lags,
                lag_method="bic",
                coint_rank="auto",
                max_obs=ci_config.max_obs,
            )
        except Exception as exc:
            tci_rows.append({**base, "status": "error", "total_connectedness": np.nan, "error": str(exc)})
            continue

        tci_rows.append(
            {
                **base,
                "status": "ok",
                "total_connectedness": result.total_connectedness,
                "coint_rank": result.coint_rank,
                "lag_order_diff": result.lag_order_diff,
                "error": "",
            }
        )
        matrix_rows.extend(_matrix_long(result.table, base))
        block_rows.extend(block_spillover_table(result.table).assign(**base).to_dict("records"))
        asset_rows.extend(_asset_connectedness_long(result.connectedness, base))

    windows = pd.DataFrame(tci_rows)
    matrix = pd.DataFrame(matrix_rows)
    blocks = pd.DataFrame(block_rows)
    assets = pd.DataFrame(asset_rows)
    return {
        "windows": windows,
        "tci_ci": summarize_ci(windows, [], "total_connectedness"),
        "matrix_ci": summarize_ci(matrix, ["receiver", "shock_source"], "value"),
        "block_ci": summarize_ci(blocks, ["receiver_block", "shock_block"], "average_receiver_share"),
        "block_adjusted_ci": summarize_ci(blocks, ["receiver_block", "shock_block"], "average_pair_share"),
        "asset_ci": summarize_ci(assets, ["asset", "measure"], "value"),
    }


def run_price_discovery_window_ci(
    prices: pd.DataFrame,
    *,
    frequency: str,
    period: str,
    max_windows: int | None = None,
) -> dict[str, pd.DataFrame]:
    ci_config = CI_WINDOW_CONFIGS[frequency]
    base_config = FREQUENCY_CONFIGS[frequency]
    metric_config = replace(
        base_config,
        max_obs=ci_config.max_obs,
        max_lags=ci_config.max_lags,
    )
    rows: list[pd.DataFrame] = []

    for window_id, (window_start, window_end, window_prices) in enumerate(
        iter_windows(
            prices,
            window=pd.Timedelta(ci_config.window),
            step=pd.Timedelta(ci_config.step),
            min_obs=ci_config.min_obs,
            max_windows=max_windows,
        ),
        start=1,
    ):
        estimates = run_pairwise_price_discovery(window_prices, frequency=frequency, config=metric_config)
        estimates.insert(0, "window_id", window_id)
        estimates.insert(1, "window_start", window_start.isoformat())
        estimates.insert(2, "window_end", window_end.isoformat())
        estimates.insert(3, "period", period)
        rows.append(estimates)

    windows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return {
        "pairwise_windows": windows,
        "left_share_ci": summarize_ci(
            windows,
            ["frequency", "period", "pair", "metric", "left_asset", "right_asset"],
            "left_share",
        ),
        "right_share_ci": summarize_ci(
            windows,
            ["frequency", "period", "pair", "metric", "left_asset", "right_asset"],
            "right_share",
        ),
        "left_bound_width_ci": summarize_ci(
            windows,
            ["frequency", "period", "pair", "metric", "left_asset", "right_asset"],
            "left_bound_width",
        ),
        "right_bound_width_ci": summarize_ci(
            windows,
            ["frequency", "period", "pair", "metric", "left_asset", "right_asset"],
            "right_bound_width",
        ),
    }


def iter_windows(
    prices: pd.DataFrame,
    *,
    window: pd.Timedelta,
    step: pd.Timedelta,
    min_obs: int,
    max_windows: int | None,
):
    if prices.empty:
        return
    current_start = prices.index.min()
    final_end = prices.index.max()
    yielded = 0
    while current_start + window <= final_end:
        current_end = current_start + window
        sample = prices.loc[(prices.index >= current_start) & (prices.index <= current_end)]
        if len(sample) >= min_obs:
            yield current_start, current_end, sample
            yielded += 1
            if max_windows is not None and yielded >= max_windows:
                break
        current_start += step


def summarize_ci(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    if frame.empty or value_col not in frame.columns:
        return pd.DataFrame()
    ok = frame.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"].eq("ok")]
    ok[value_col] = pd.to_numeric(ok[value_col], errors="coerce")
    ok = ok.dropna(subset=[value_col])
    if ok.empty:
        return pd.DataFrame()

    def summarize(group: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "n_windows": int(group.count()),
                "mean": float(group.mean()),
                "std": float(group.std(ddof=1)) if group.count() > 1 else 0.0,
                "q025": float(group.quantile(0.025)),
                "q05": float(group.quantile(0.05)),
                "median": float(group.quantile(0.5)),
                "q95": float(group.quantile(0.95)),
                "q975": float(group.quantile(0.975)),
                "min": float(group.min()),
                "max": float(group.max()),
            }
        )

    if group_cols:
        return ok.groupby(group_cols, dropna=False)[value_col].apply(summarize).unstack().reset_index()
    summary = summarize(ok[value_col]).to_frame().T
    return summary


def compare_ci_summaries(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Approximate post-minus-pre p-values from empirical window summaries."""
    required = set(group_cols) | {"mean", "std", "n_windows"}
    if pre.empty or post.empty or not required.issubset(pre.columns) or not required.issubset(post.columns):
        return pd.DataFrame()

    merged = pre.merge(post, on=group_cols, suffixes=("_pre", "_post"))
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        pre_mean = float(row["mean_pre"])
        post_mean = float(row["mean_post"])
        pre_std = float(row["std_pre"])
        post_std = float(row["std_post"])
        pre_n = max(1.0, float(row["n_windows_pre"]))
        post_n = max(1.0, float(row["n_windows_post"]))
        diff = post_mean - pre_mean
        se = sqrt((pre_std * pre_std) / pre_n + (post_std * post_std) / post_n)
        if se > 0:
            z_value = diff / se
            p_value = erfc(abs(z_value) / sqrt(2.0))
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
        else:
            z_value = 0.0
            p_value = 1.0 if diff == 0.0 else 0.0
            ci_low = diff
            ci_high = diff
        payload = {col: row[col] for col in group_cols}
        payload.update(
            {
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "diff_mean": diff,
                "std_error": se,
                "z_value": z_value,
                "p_value": p_value,
                "stars": p_value_stars(p_value),
                "diff_ci_low": ci_low,
                "diff_ci_high": ci_high,
                "pre_windows": int(pre_n),
                "post_windows": int(post_n),
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def p_value_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def save_ci_outputs(outputs: dict[str, pd.DataFrame], *, output_dir: Path, frequency: str, period: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / f"{name}_{period}_{frequency}.csv"
        if frame.empty:
            if path.exists():
                path.unlink()
            continue
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def _matrix_long(table: pd.DataFrame, base: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for receiver in table.index:
        for source in table.columns:
            rows.append(
                {
                    **base,
                    "receiver": str(receiver),
                    "shock_source": str(source),
                    "value": float(table.loc[receiver, source]),
                }
            )
    return rows


def _asset_connectedness_long(table: pd.DataFrame, base: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for asset in table.index:
        for measure in table.columns:
            rows.append(
                {
                    **base,
                    "asset": str(asset),
                    "measure": str(measure),
                    "value": float(table.loc[asset, measure]),
                }
            )
    return rows
