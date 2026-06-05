from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calculations.config import BLOCK_ORDER
from calculations.plots import (
    BLOCK_COLORS,
    DIVERGING_CMAP,
    FLOW_COLORS,
    NEUTRAL_COLOR,
    PERIOD_COLORS,
    SUPTITLE_LAYOUT_TOP,
    SUPTITLE_Y,
)


FULL_FREQUENCIES: tuple[str, ...] = ("1min", "1h", "1d")
PERIODS: tuple[str, ...] = ("pre_covid", "covid_and_after")
BLOCK_VARIANTS: dict[str, str] = {
    "total": "block_matrix",
    "adjusted": "block_matrix_adjusted",
}


def block_net_flows(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute TO, FROM, and NET spillovers by block from a block GFEVD matrix."""
    values = matrix.copy()
    values.index = values.index.astype(str)
    values.columns = values.columns.astype(str)
    rows: list[dict[str, object]] = []
    blocks = [block for block in BLOCK_ORDER if block in values.index and block in values.columns]
    for block in blocks:
        others = [other for other in blocks if other != block]
        to_others = float(values.loc[others, block].sum()) if others else 0.0
        from_others = float(values.loc[block, others].sum()) if others else 0.0
        rows.append(
            {
                "block": block,
                "to_others": to_others,
                "from_others": from_others,
                "net_to_others": to_others - from_others,
                "gross_external": to_others + from_others,
            }
        )
    return pd.DataFrame(rows)


def crypto_global_flows(matrix: pd.DataFrame) -> dict[str, float]:
    """Summarize cross-block flows between crypto and all non-crypto blocks."""
    values = matrix.copy()
    values.index = values.index.astype(str)
    values.columns = values.columns.astype(str)
    if "crypto" not in values.index or "crypto" not in values.columns:
        return {
            "global_to_crypto": np.nan,
            "crypto_to_global": np.nan,
            "net_crypto_to_global": np.nan,
            "gross_crypto_global": np.nan,
        }
    global_blocks = [block for block in BLOCK_ORDER if block != "crypto" and block in values.index and block in values.columns]
    global_to_crypto = float(values.loc["crypto", global_blocks].sum()) if global_blocks else 0.0
    crypto_to_global = float(values.loc[global_blocks, "crypto"].sum()) if global_blocks else 0.0
    return {
        "global_to_crypto": global_to_crypto,
        "crypto_to_global": crypto_to_global,
        "net_crypto_to_global": crypto_to_global - global_to_crypto,
        "gross_crypto_global": global_to_crypto + crypto_to_global,
    }


def window_block_net_flows(block_windows: pd.DataFrame, *, value_col: str, variant: str) -> pd.DataFrame:
    required = {"frequency", "window_id", "window_start", "window_end", "receiver_block", "shock_block", value_col}
    if block_windows.empty or not required.issubset(block_windows.columns):
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for keys, group in block_windows.groupby(["frequency", "window_id", "window_start", "window_end"], dropna=False):
        frequency, window_id, window_start, window_end = keys
        matrix = group.pivot(index="receiver_block", columns="shock_block", values=value_col)
        flows = block_net_flows(matrix)
        flows.insert(0, "variant", variant)
        flows.insert(0, "window_end", window_end)
        flows.insert(0, "window_start", window_start)
        flows.insert(0, "window_id", window_id)
        flows.insert(0, "frequency", frequency)
        rows.append(flows)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def window_crypto_global_flows(block_windows: pd.DataFrame, *, value_col: str, variant: str) -> pd.DataFrame:
    required = {"frequency", "window_id", "window_start", "window_end", "receiver_block", "shock_block", value_col}
    if block_windows.empty or not required.issubset(block_windows.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in block_windows.groupby(["frequency", "window_id", "window_start", "window_end"], dropna=False):
        frequency, window_id, window_start, window_end = keys
        matrix = group.pivot(index="receiver_block", columns="shock_block", values=value_col)
        rows.append(
            {
                "frequency": frequency,
                "window_id": window_id,
                "window_start": window_start,
                "window_end": window_end,
                "variant": variant,
                **crypto_global_flows(matrix),
            }
        )
    return pd.DataFrame(rows)


def add_interval_stability(
    frame: pd.DataFrame,
    *,
    mean_col: str = "mean",
    low_col: str = "q025",
    high_col: str = "q975",
    n_col: str = "n_windows",
    max_width: float = 0.15,
    max_relative_width: float = 1.0,
    min_windows: int = 8,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    for col in (mean_col, low_col, high_col, n_col):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    result["ci_width"] = result[high_col] - result[low_col]
    result["relative_ci_width"] = result["ci_width"] / result[mean_col].abs().clip(lower=1e-12)
    result["enough_windows"] = result[n_col] >= min_windows
    result["narrow_ci"] = (result["ci_width"] <= max_width) | (result["relative_ci_width"] <= max_relative_width)
    result["stability_flag"] = np.select(
        [~result["enough_windows"], result["narrow_ci"]],
        ["low_window_count", "stable"],
        default="volatile",
    )
    return result


def add_diff_stability(
    frame: pd.DataFrame,
    *,
    diff_col: str = "diff_mean",
    low_col: str = "diff_ci_low",
    high_col: str = "diff_ci_high",
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    for col in (diff_col, low_col, high_col, p_col):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    result["ci_excludes_zero"] = (result[low_col] > 0.0) | (result[high_col] < 0.0)
    result["significant"] = result[p_col] < alpha
    result["robust_change"] = result["ci_excludes_zero"] & result["significant"]
    result["direction"] = np.select(
        [result[diff_col] > 0.0, result[diff_col] < 0.0],
        ["post_up", "post_down"],
        default="flat",
    )
    return result


def add_price_discovery_dominance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = add_interval_stability(frame, max_width=0.35, max_relative_width=0.8)
    result["distance_from_parity"] = result["mean"] - 0.5
    result["dominance"] = np.select(
        [
            result["q025"] > 0.5,
            result["q975"] < 0.5,
            result["q025"].le(0.5) & result["q975"].ge(0.5),
        ],
        ["left_dominates", "right_dominates", "parity_or_mixed"],
        default="unknown",
    )
    result["stable_dominance"] = result["dominance"].isin(["left_dominates", "right_dominates"])
    return result


def frequency_consistency(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    label: str,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = set(group_cols) | {"frequency", value_col}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        signs: list[str] = []
        values: dict[str, float] = {}
        for _, row in group.iterrows():
            frequency = str(row["frequency"])
            value = float(row[value_col])
            values[frequency] = value
            if value > epsilon:
                signs.append("positive")
            elif value < -epsilon:
                signs.append("negative")
            else:
                signs.append("zero")
        nonzero = [sign for sign in signs if sign != "zero"]
        consistent = bool(nonzero) and len(set(nonzero)) == 1
        payload = {"check": label}
        payload.update({col: value for col, value in zip(group_cols, key_tuple, strict=True)})
        payload.update(
            {
                "frequencies": ",".join(sorted(values)),
                "positive_count": signs.count("positive"),
                "negative_count": signs.count("negative"),
                "zero_count": signs.count("zero"),
                "consistent_nonzero_sign": consistent,
                "direction": nonzero[0] if consistent else "mixed",
            }
        )
        for frequency in FULL_FREQUENCIES:
            payload[f"value_{frequency}"] = values.get(frequency, np.nan)
        rows.append(payload)
    return pd.DataFrame(rows)


def metric_agreement(frame: pd.DataFrame, *, group_cols: list[str], value_col: str = "mean") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = set(group_cols) | {"metric", value_col}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = {
            str(row["metric"]): float(row[value_col])
            for _, row in group.iterrows()
            if not pd.isna(row[value_col])
        }
        gis = values.get("gis")
        hasbrouck = values.get("hasbrouck_proxy")
        if gis is None or hasbrouck is None:
            continue
        gis_side = "left" if gis > 0.5 else "right" if gis < 0.5 else "parity"
        hasbrouck_side = "left" if hasbrouck > 0.5 else "right" if hasbrouck < 0.5 else "parity"
        payload = {col: value for col, value in zip(group_cols, key_tuple, strict=True)}
        payload.update(
            {
                "gis_share": gis,
                "hasbrouck_share": hasbrouck,
                "absolute_gap": abs(gis - hasbrouck),
                "same_side_of_parity": gis_side == hasbrouck_side,
                "gis_side": gis_side,
                "hasbrouck_side": hasbrouck_side,
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def plot_block_net_flows(frame: pd.DataFrame, *, variant: str, path: Path) -> None:
    data = frame[frame["variant"].eq(variant)].copy()
    if data.empty:
        return
    blocks = [block for block in BLOCK_ORDER if block in set(data["block"])]
    fig, axes = plt.subplots(len(FULL_FREQUENCIES), 1, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, frequency in zip(axes, FULL_FREQUENCIES, strict=True):
        subset = data[data["frequency"].eq(frequency)]
        x = np.arange(len(blocks), dtype=float)
        width = 0.36
        for offset, period, color in (
            (-width / 2, "pre_covid", PERIOD_COLORS["pre_covid"]),
            (width / 2, "covid_and_after", PERIOD_COLORS["covid_and_after"]),
        ):
            values = []
            indexed = subset[subset["period"].eq(period)].set_index("block")
            for block in blocks:
                values.append(float(indexed.loc[block, "net_to_others"]) if block in indexed.index else np.nan)
            ax.bar(x + offset, values, width=width, label=period, color=color)
        ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.9)
        ax.set_title(frequency)
        ax.set_ylabel("TO - FROM")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(np.arange(len(blocks)), blocks, rotation=25, ha="right")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Block net spillovers ({variant})", y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_crypto_global_gfevd(frame: pd.DataFrame, *, variant: str, path: Path) -> None:
    data = frame[frame["variant"].eq(variant)].copy()
    if data.empty:
        return
    measures = ["global_to_crypto", "crypto_to_global", "net_crypto_to_global"]
    labels = ["global -> crypto", "crypto -> global", "net crypto -> global"]
    fig, axes = plt.subplots(len(FULL_FREQUENCIES), 1, figsize=(11, 8.5), sharex=True)
    for ax, frequency in zip(axes, FULL_FREQUENCIES, strict=True):
        subset = data[data["frequency"].eq(frequency)]
        x = np.arange(len(measures), dtype=float)
        width = 0.36
        for offset, period, color in (
            (-width / 2, "pre_covid", PERIOD_COLORS["pre_covid"]),
            (width / 2, "covid_and_after", PERIOD_COLORS["covid_and_after"]),
        ):
            row = subset[subset["period"].eq(period)]
            values = [float(row.iloc[0][measure]) if not row.empty else np.nan for measure in measures]
            ax.bar(x + offset, values, width=width, label=period, color=color)
        ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.9)
        ax.set_title(frequency)
        ax.set_ylabel("share")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(np.arange(len(measures)), labels, rotation=20, ha="right")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Crypto vs global GFEVD flows ({variant})", y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_block_net_dynamics(frame: pd.DataFrame, *, variant: str, path: Path) -> None:
    data = frame[frame["variant"].eq(variant)].copy()
    if data.empty:
        return
    data["window_start"] = pd.to_datetime(data["window_start"], utc=True)
    blocks = [block for block in BLOCK_ORDER if block in set(data["block"])]
    fig, axes = plt.subplots(len(FULL_FREQUENCIES), 1, figsize=(12, 8.8), sharex=True, sharey=True)
    for ax, frequency in zip(axes, FULL_FREQUENCIES, strict=True):
        subset = data[data["frequency"].eq(frequency)]
        for block in blocks:
            line = subset[subset["block"].eq(block)].sort_values("window_start")
            if line.empty:
                continue
            ax.plot(
                line["window_start"],
                line["net_to_others"],
                linewidth=1.25,
                label=block,
                color=BLOCK_COLORS.get(block, BLOCK_COLORS["unknown"]),
            )
        ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.9)
        ax.set_title(frequency)
        ax.set_ylabel("TO - FROM")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Window start")
    fig.suptitle(f"Rolling block net spillovers ({variant})", y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_crypto_global_gfevd_dynamics(frame: pd.DataFrame, *, variant: str, path: Path) -> None:
    data = frame[frame["variant"].eq(variant)].copy()
    if data.empty:
        return
    data["window_start"] = pd.to_datetime(data["window_start"], utc=True)
    measures = [
        ("global_to_crypto", "global -> crypto", FLOW_COLORS["global_to_crypto"]),
        ("crypto_to_global", "crypto -> global", FLOW_COLORS["crypto_to_global"]),
        ("net_crypto_to_global", "net crypto -> global", FLOW_COLORS["net_crypto_to_global"]),
    ]
    fig, axes = plt.subplots(len(FULL_FREQUENCIES), 1, figsize=(12, 8.8), sharex=True, sharey=True)
    for ax, frequency in zip(axes, FULL_FREQUENCIES, strict=True):
        subset = data[data["frequency"].eq(frequency)].sort_values("window_start")
        if subset.empty:
            continue
        for measure, label, color in measures:
            ax.plot(subset["window_start"], subset[measure], linewidth=1.25, label=label, color=color)
        ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.9)
        ax.set_title(frequency)
        ax.set_ylabel("share")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Window start")
    fig.suptitle(f"Rolling crypto vs global GFEVD flows ({variant})", y=SUPTITLE_Y)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frequency_heatmap(frame: pd.DataFrame, *, title: str, path: Path, value_col: str = "value") -> None:
    if frame.empty or not {"effect", "frequency", value_col}.issubset(frame.columns):
        return
    pivot = frame.pivot_table(index="effect", columns="frequency", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(columns=[frequency for frequency in FULL_FREQUENCIES if frequency in pivot.columns])
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    bound = max(1e-6, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(pivot))))
    image = ax.imshow(values, cmap=DIVERGING_CMAP, vmin=-bound, vmax=bound)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{values[row, col]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hypothesis_dashboard(checks: pd.DataFrame, *, path: Path) -> None:
    if checks.empty:
        return
    frame = checks.copy()
    columns = ["hypothesis", "evidence", "status", "stability"]
    frame = frame[columns]
    fig_height = max(4.5, 0.65 * len(frame) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=frame.to_numpy(),
        colLabels=["Hypothesis", "Evidence", "Status", "Stability"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.24, 0.40, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2d3748")
            cell.set_text_props(color="white", weight="bold")
        elif col in (2, 3):
            text = str(cell.get_text().get_text()).lower()
            if "yes" in text or "stable" in text or "support" in text:
                cell.set_facecolor("#d9f2e3")
            elif "mixed" in text or "partial" in text:
                cell.set_facecolor("#fff4cc")
            elif "no" in text or "volatile" in text:
                cell.set_facecolor("#f8d7da")
    ax.set_title("Hypothesis stability dashboard", pad=18)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
