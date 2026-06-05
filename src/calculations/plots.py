from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from calculations.config import ASSET_BLOCKS, BLOCK_ORDER
from calculations.gfevd_analysis import block_matrix


FREQUENCY_ORDER_ARTICLE: tuple[str, ...] = ("1d", "1h", "1min")
FREQUENCY_LABELS: dict[str, str] = {
    "1d": "day",
    "1h": "hour",
    "1min": "minute",
    "1s": "second",
}
SUPTITLE_Y = 0.715
SUPTITLE_LAYOUT_TOP = 0.755
HEATMAP_SUPTITLE_Y = 0.94
HEATMAP_LAYOUT_TOP = 0.90
POSITIVE_CMAP = LinearSegmentedColormap.from_list(
    "risk_teal_plum",
    ["#F4F2ED", "#B9D8C7", "#66A99A", "#76547B", "#2B1E3A"],
)
DIVERGING_CMAP = "RdBu_r"
NEUTRAL_COLOR = "#2f3747"
BLOCK_LABELS: dict[str, str] = {
    "crypto": "Crypto",
    "equity_index": "Equity indexes",
    "fx_dollar": "Dollar/FX",
    "commodity": "Commodities",
    "unknown": "Other",
}
PERIOD_COLORS: dict[str, str] = {
    "pre_covid": "#6D597A",
    "covid_and_after": "#E76F51",
}
BLOCK_COLORS: dict[str, str] = {
    "crypto": "#6D597A",
    "equity_index": "#2A9D8F",
    "fx_dollar": "#E76F51",
    "commodity": "#B56576",
    "unknown": "#58606F",
}
BLOCK_LINE_COLORS: dict[str, str] = {
    "crypto": "#7B3294",
    "equity_index": "#00876C",
    "fx_dollar": "#D95F02",
    "commodity": "#B2182B",
    "unknown": "#4B5563",
}
BLOCK_LINE_STYLES: dict[str, str] = {
    "crypto": "-",
    "equity_index": "--",
    "fx_dollar": "-.",
    "commodity": ":",
    "unknown": "-",
}
METRIC_LINE_COLORS: dict[str, str] = {
    "gis": "#7B3294",
    "hasbrouck_proxy": "#00876C",
}
METRIC_LINE_STYLES: dict[str, str] = {
    "gis": "-",
    "hasbrouck_proxy": "--",
}
FLOW_COLORS: dict[str, str] = {
    "global_to_crypto": "#B56576",
    "crypto_to_global": "#6D597A",
    "net_crypto_to_global": "#2f3747",
}
ASSET_LINE_COLORS: tuple[str, ...] = (
    "#0077BB",
    "#009988",
    "#EE7733",
    "#CC3311",
    "#EE3377",
    "#AA4499",
    "#44AA99",
    "#882255",
    "#555555",
    "#332288",
)


def plot_normalized_prices(prices: pd.DataFrame, *, title: str, path: Path, max_points: int = 5_000) -> None:
    sampled = _downsample(prices, max_points=max_points)
    normalized = sampled.divide(sampled.iloc[0]).multiply(100.0)

    n_cols = 2
    n_rows = int(np.ceil(len(BLOCK_ORDER) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5.4), sharex=True)
    flat_axes = np.asarray(axes, dtype=object).ravel()
    for ax, block in zip(flat_axes, BLOCK_ORDER, strict=True):
        assets = [asset for asset in normalized.columns if ASSET_BLOCKS.get(asset) == block]
        if not assets:
            ax.set_visible(False)
            continue
        for idx, asset in enumerate(assets):
            ax.plot(
                normalized.index,
                normalized[asset],
                linewidth=0.9,
                label=asset,
                color=ASSET_LINE_COLORS[idx % len(ASSET_LINE_COLORS)],
            )
        ax.set_title(block)
        ax.set_ylabel("index=100")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=min(4, len(assets)), fontsize=8)
    for ax in flat_axes[len(BLOCK_ORDER) :]:
        ax.set_visible(False)

    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_gfevd_heatmap(table: pd.DataFrame, *, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(table.to_numpy(), cmap=POSITIVE_CMAP, vmin=0.0)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=90)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_title(title)
    ax.set_xlabel("Shock source")
    ax.set_ylabel("Receiver")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_gfevd_log_heatmap(
    table: pd.DataFrame,
    *,
    title: str,
    path: Path,
    epsilon: float = 1e-6,
) -> None:
    values = np.log10(np.clip(table.to_numpy(dtype=float), epsilon, None))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, cmap=POSITIVE_CMAP)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=90)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_title(title)
    ax.set_xlabel("Shock source")
    ax.set_ylabel("Receiver")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(f"log10(value + eps), eps={epsilon:g}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_pre_post_heatmaps(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    *,
    title: str,
    path: Path,
    log_scale: bool = False,
    epsilon: float = 1e-6,
    annotations: tuple[pd.DataFrame | None, pd.DataFrame | None] | None = None,
) -> None:
    pre_values = pre.to_numpy(dtype=float)
    post_values = post.to_numpy(dtype=float)
    colorbar_label = "GFEVD share"
    if log_scale:
        pre_values = np.log10(np.clip(pre_values, epsilon, None))
        post_values = np.log10(np.clip(post_values, epsilon, None))
        colorbar_label = f"log10(value + eps), eps={epsilon:g}"

    vmin = float(np.nanmin([np.nanmin(pre_values), np.nanmin(post_values)]))
    vmax = float(np.nanmax([np.nanmax(pre_values), np.nanmax(post_values)]))

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True, sharey=True)
    for ax, values, subtitle in zip(axes, (pre_values, post_values), ("pre_covid", "covid_and_after"), strict=True):
        image = ax.imshow(values, cmap=POSITIVE_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(subtitle)
        ax.set_xticks(range(pre.shape[1]), pre.columns, rotation=90)
        ax.set_yticks(range(pre.shape[0]), pre.index)
        ax.set_xlabel("Shock source")
        label_table = None
        if annotations is not None:
            label_table = annotations[0] if subtitle == "pre_covid" else annotations[1]
        _annotate_matrix(ax, label_table)
    axes[0].set_ylabel("Receiver")
    _set_suptitle(fig, title, heatmap=True)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    colorbar.set_label(colorbar_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_matrix_difference(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    *,
    title: str,
    path: Path,
    annotations: pd.DataFrame | None = None,
) -> None:
    diff = post.reindex_like(pre) - pre
    values = diff.to_numpy(dtype=float)
    bound = float(np.nanmax(np.abs(values)))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, cmap=DIVERGING_CMAP, vmin=-bound, vmax=bound)
    ax.set_xticks(range(diff.shape[1]), diff.columns, rotation=90)
    ax.set_yticks(range(diff.shape[0]), diff.index)
    ax.set_title(title)
    ax.set_xlabel("Shock source")
    ax.set_ylabel("Receiver")
    _annotate_matrix(ax, annotations)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("post - pre")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_block_heatmap(
    block_table: pd.DataFrame,
    *,
    title: str,
    path: Path,
    value_col: str = "average_receiver_share",
) -> None:
    matrix = block_matrix(block_table, value_col=value_col).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(matrix.to_numpy(), cmap=POSITIVE_CMAP, vmin=0.0)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Shock block")
    ax.set_ylabel("Receiver block")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix.iat[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_block_flow_dynamics(
    block_flows: pd.DataFrame,
    *,
    title: str,
    path: Path,
    value_col: str = "average_receiver_share",
) -> None:
    if block_flows.empty:
        return
    required = {"window_start", "receiver_block", "shock_block", value_col}
    if not required.issubset(block_flows.columns):
        return
    frame = block_flows.copy()
    frame = frame[frame["receiver_block"].ne(frame["shock_block"])]
    if frame.empty:
        return
    frame["window_start"] = pd.to_datetime(frame["window_start"], utc=True)

    fig, axes = plt.subplots(len(BLOCK_ORDER), 1, figsize=(12, 9), sharex=True, sharey=True)
    for ax, receiver_block in zip(axes, BLOCK_ORDER, strict=True):
        subset = frame[frame["receiver_block"].eq(receiver_block)]
        if subset.empty:
            ax.set_visible(False)
            continue
        for shock_block in BLOCK_ORDER:
            if shock_block == receiver_block:
                continue
            line = subset[subset["shock_block"].eq(shock_block)].sort_values("window_start")
            if line.empty:
                continue
            ax.plot(
                line["window_start"],
                line[value_col],
                linewidth=1.25,
                linestyle=BLOCK_LINE_STYLES.get(shock_block, BLOCK_LINE_STYLES["unknown"]),
                color=BLOCK_LINE_COLORS.get(shock_block, BLOCK_LINE_COLORS["unknown"]),
                label=f"from {_block_label(shock_block)}",
            )
        ax.set_title(f"receiver: {_block_label(receiver_block)}")
        ax.set_ylabel("share")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Window start")
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gfevd_network(
    table: pd.DataFrame,
    *,
    title: str,
    path: Path,
    top_n: int = 50,
    min_weight: float = 0.01,
) -> None:
    values = table.copy()
    values.index = values.index.astype(str)
    values.columns = values.columns.astype(str)
    edge_rows: list[tuple[str, str, float]] = []
    for receiver in values.index:
        for source in values.columns:
            if receiver == source:
                continue
            weight = float(values.loc[receiver, source])
            if weight >= min_weight:
                edge_rows.append((source, receiver, weight))
    edge_rows = sorted(edge_rows, key=lambda row: row[2], reverse=True)[:top_n]

    assets = list(values.index)
    positions = _circular_block_positions(assets)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.axis("off")

    max_weight = max((edge[2] for edge in edge_rows), default=1.0)
    for source, receiver, weight in edge_rows:
        start = positions[source]
        end = positions[receiver]
        edge = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8 + 14 * weight / max_weight,
            linewidth=0.4 + 4.0 * weight / max_weight,
            color="#2d3748",
            alpha=0.18 + 0.55 * weight / max_weight,
            connectionstyle="arc3,rad=0.08",
            shrinkA=15,
            shrinkB=15,
            zorder=1,
        )
        ax.add_patch(edge)

    for asset, (x_pos, y_pos) in positions.items():
        block = ASSET_BLOCKS.get(asset, "unknown")
        ax.scatter(
            [x_pos],
            [y_pos],
            s=520,
            color=BLOCK_COLORS[block],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(x_pos, y_pos, asset, ha="center", va="center", fontsize=8, color="white", zorder=4)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=block, markerfacecolor=BLOCK_COLORS[block], markersize=9)
        for block in BLOCK_ORDER
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_tci_confidence(summary: pd.DataFrame, *, title: str, path: Path) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values(["frequency", "period"]).copy()
    labels = ordered["frequency"] + " / " + ordered["period"]
    means = ordered["mean"].astype(float).to_numpy()
    lower = ordered["q025"].astype(float).to_numpy()
    upper = ordered["q975"].astype(float).to_numpy()
    yerr = np.vstack([means - lower, upper - means])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.errorbar(range(len(ordered)), means, yerr=yerr, fmt="o", capsize=4, color=BLOCK_COLORS["crypto"])
    ax.set_xticks(range(len(ordered)), labels, rotation=35, ha="right")
    ax.set_ylabel("Total connectedness")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rolling_tci(rolling: pd.DataFrame, *, title: str, path: Path) -> None:
    ok = rolling[rolling["status"].eq("ok")].copy()
    if ok.empty:
        return
    x_col = "window_start" if "window_start" in ok.columns else "window_end"
    ok[x_col] = pd.to_datetime(ok[x_col], utc=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ok[x_col], ok["total_connectedness"], linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Window start" if x_col == "window_start" else "Window end")
    ax.set_ylabel("Total connectedness")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _annotate_matrix(ax: plt.Axes, annotations: pd.DataFrame | None) -> None:
    if annotations is None or annotations.empty:
        return
    for row_idx, row_name in enumerate(annotations.index):
        for col_idx, col_name in enumerate(annotations.columns):
            label = str(annotations.loc[row_name, col_name])
            if not label or label == "nan":
                continue
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
            )


def plot_price_discovery(discovery: pd.DataFrame, *, frequency: str, path: Path) -> None:
    ok = discovery[
        discovery["status"].eq("ok")
        & discovery["metric"].eq("hasbrouck_proxy")
        & discovery["frequency"].eq(frequency)
    ].copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(ok["pair"], ok["left_share"], color=BLOCK_COLORS["crypto"])
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Left asset share")
    ax.set_title(f"Hasbrouck proxy, {frequency}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_price_discovery_share_pre_post(
    summary: pd.DataFrame,
    *,
    title: str,
    path: Path,
    label_col: str = "pair",
    x_label: str = "Left asset long-run share",
) -> None:
    required = {"period", "metric", label_col, "mean", "q025", "q975"}
    if summary.empty or not required.issubset(summary.columns):
        return
    frame = summary.copy()
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["q025"] = pd.to_numeric(frame["q025"], errors="coerce")
    frame["q975"] = pd.to_numeric(frame["q975"], errors="coerce")
    frame = frame.dropna(subset=["mean", "q025", "q975"])
    if frame.empty:
        return

    metrics = _ordered_metrics(frame["metric"].astype(str).unique())
    labels = sorted(frame[label_col].astype(str).unique())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(4.5, 0.5 * len(labels) * len(metrics))), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    period_styles = {
        "pre_covid": {"offset": -0.13, "color": PERIOD_COLORS["pre_covid"], "label": "pre_covid"},
        "covid_and_after": {"offset": 0.13, "color": PERIOD_COLORS["covid_and_after"], "label": "covid_and_after"},
    }
    y_positions = np.arange(len(labels), dtype=float)
    for ax, metric in zip(axes, metrics, strict=True):
        metric_frame = frame[frame["metric"].astype(str).eq(metric)]
        indexed = metric_frame.set_index(["period", label_col])
        for period, style in period_styles.items():
            means: list[float] = []
            left_err: list[float] = []
            right_err: list[float] = []
            for label in labels:
                key = (period, label)
                if key not in indexed.index:
                    means.append(np.nan)
                    left_err.append(0.0)
                    right_err.append(0.0)
                    continue
                row = indexed.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                mean = float(row["mean"])
                means.append(mean)
                left_err.append(max(0.0, mean - float(row["q025"])))
                right_err.append(max(0.0, float(row["q975"]) - mean))
            values = np.asarray(means, dtype=float)
            valid = ~np.isnan(values)
            if not valid.any():
                continue
            ax.errorbar(
                values[valid],
                y_positions[valid] + float(style["offset"]),
                xerr=np.vstack([np.asarray(left_err)[valid], np.asarray(right_err)[valid]]),
                fmt="o",
                capsize=3,
                color=str(style["color"]),
                label=str(style["label"]),
            )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlim(0.0, 1.0)
        ax.set_yticks(y_positions, labels)
        ax.set_title(_metric_label(metric))
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel(x_label)
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_price_discovery_share_diff(
    diff: pd.DataFrame,
    *,
    title: str,
    path: Path,
    label_col: str = "pair",
    x_label: str = "post - pre left asset long-run share",
) -> None:
    required = {"metric", label_col, "diff_mean", "diff_ci_low", "diff_ci_high", "stars"}
    if diff.empty or not required.issubset(diff.columns):
        return
    frame = diff.copy()
    for col in ("diff_mean", "diff_ci_low", "diff_ci_high"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["diff_mean", "diff_ci_low", "diff_ci_high"])
    if frame.empty:
        return

    metrics = _ordered_metrics(frame["metric"].astype(str).unique())
    labels = sorted(frame[label_col].astype(str).unique())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(4.5, 0.5 * len(labels) * len(metrics))), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    bound = max(0.05, float(np.nanmax(np.abs(frame[["diff_ci_low", "diff_ci_high"]].to_numpy(dtype=float)))))
    y_positions = np.arange(len(labels), dtype=float)
    for ax, metric in zip(axes, metrics, strict=True):
        metric_frame = frame[frame["metric"].astype(str).eq(metric)].set_index(label_col)
        means: list[float] = []
        left_err: list[float] = []
        right_err: list[float] = []
        stars: list[str] = []
        for label in labels:
            if label not in metric_frame.index:
                means.append(np.nan)
                left_err.append(0.0)
                right_err.append(0.0)
                stars.append("")
                continue
            row = metric_frame.loc[label]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            mean = float(row["diff_mean"])
            means.append(mean)
            left_err.append(max(0.0, mean - float(row["diff_ci_low"])))
            right_err.append(max(0.0, float(row["diff_ci_high"]) - mean))
            stars.append("" if pd.isna(row["stars"]) else str(row["stars"]))
        values = np.asarray(means, dtype=float)
        valid = ~np.isnan(values)
        if valid.any():
            ax.errorbar(
                values[valid],
                y_positions[valid],
                xerr=np.vstack([np.asarray(left_err)[valid], np.asarray(right_err)[valid]]),
                fmt="o",
                capsize=3,
                color=BLOCK_COLORS["crypto"],
            )
            for x_value, y_value, star in zip(values[valid], y_positions[valid], np.asarray(stars, dtype=object)[valid], strict=True):
                if star:
                    ax.text(x_value, y_value + 0.17, star, ha="center", va="bottom", fontsize=10)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlim(-bound * 1.1, bound * 1.1)
        ax.set_yticks(y_positions, labels)
        ax.set_title(_metric_label(metric))
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel(x_label)
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_price_discovery_block_dynamics(
    windows: pd.DataFrame,
    *,
    title: str,
    path: Path,
    y_label: str = "Left block long-run share",
) -> None:
    required = {"window_start", "metric", "right_block", "left_share"}
    if windows.empty or not required.issubset(windows.columns):
        return
    frame = windows.copy()
    frame["window_start"] = pd.to_datetime(frame["window_start"], utc=True)
    frame["left_share"] = pd.to_numeric(frame["left_share"], errors="coerce")
    frame = frame.dropna(subset=["left_share"])
    if frame.empty:
        return

    metrics = _ordered_metrics(frame["metric"].astype(str).unique())
    right_blocks = [block for block in BLOCK_ORDER if block in set(frame["right_block"].astype(str))]
    right_blocks.extend(sorted(set(frame["right_block"].astype(str)) - set(right_blocks)))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4.2 * len(metrics)), sharex=True, sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        metric_frame = frame[frame["metric"].astype(str).eq(metric)]
        for right_block in right_blocks:
            line = metric_frame[metric_frame["right_block"].astype(str).eq(right_block)].sort_values("window_start")
            if line.empty:
                continue
            _plot_long_run_share_line(
                ax,
                line,
                color=BLOCK_COLORS.get(right_block, BLOCK_COLORS["unknown"]),
                label=f"vs {right_block}",
                linewidth=1.35,
            )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(y_label)
        ax.set_title(_metric_label(metric))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Window start")
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pre_post_frequency_grid(
    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    title: str,
    path: Path,
    log_scale: bool = False,
    epsilon: float = 1e-6,
    annotate_values: bool = False,
    annotations_by_frequency: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] | None = None,
    tick_fontsize: int = 7,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
) -> None:
    frequencies = [frequency for frequency in FREQUENCY_ORDER_ARTICLE if frequency in matrices]
    if not frequencies:
        return

    transformed: dict[tuple[str, str], np.ndarray] = {}
    all_values: list[np.ndarray] = []
    for frequency in frequencies:
        pre, post = matrices[frequency]
        for period, matrix in (("pre_covid", pre), ("covid_and_after", post)):
            values = matrix.to_numpy(dtype=float)
            if log_scale:
                values = np.log10(np.clip(values, epsilon, None))
            transformed[(frequency, period)] = values
            all_values.append(values)

    resolved_vmin = float(np.nanmin([np.nanmin(values) for values in all_values])) if vmin is None else float(vmin)
    resolved_vmax = float(np.nanmax([np.nanmax(values) for values in all_values])) if vmax is None else float(vmax)
    fig, axes = plt.subplots(2, len(frequencies), figsize=(5.1 * len(frequencies), 9.0), squeeze=False)
    last_image = None
    for col, frequency in enumerate(frequencies):
        pre, post = matrices[frequency]
        for row, (period, matrix) in enumerate((("pre_covid", pre), ("covid_and_after", post))):
            ax = axes[row, col]
            values = transformed[(frequency, period)]
            last_image = ax.imshow(values, cmap=POSITIVE_CMAP, vmin=resolved_vmin, vmax=resolved_vmax)
            ax.set_title(f"{FREQUENCY_LABELS.get(frequency, frequency)} / {period}")
            ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90, fontsize=tick_fontsize)
            ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=tick_fontsize)
            if row == 1:
                ax.set_xlabel("Shock source")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel("Receiver")
            annotation_table = None
            if annotations_by_frequency is not None and frequency in annotations_by_frequency:
                annotation_table = annotations_by_frequency[frequency][row]
            if annotation_table is not None:
                _annotate_matrix(ax, annotation_table)
            elif annotate_values and matrix.shape[0] <= 6 and matrix.shape[1] <= 6:
                _annotate_numeric_matrix(ax, matrix)
    _set_suptitle(fig, title, heatmap=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=HEATMAP_LAYOUT_TOP, bottom=0.14, hspace=0.34, wspace=0.28)
    colorbar_ax = fig.add_axes([0.905, 0.21, 0.016, 0.58])
    colorbar = fig.colorbar(last_image, cax=colorbar_ax)
    if colorbar_label is None:
        colorbar_label = f"log10(value + eps), eps={epsilon:g}" if log_scale else "GFEVD share"
    colorbar.set_label(colorbar_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_block_flow_dynamics_frequency_grid(
    block_flows_by_frequency: dict[str, pd.DataFrame],
    *,
    title: str,
    path: Path,
    value_col: str = "average_pair_share",
) -> None:
    frequencies = [frequency for frequency in FREQUENCY_ORDER_ARTICLE if frequency in block_flows_by_frequency]
    if not frequencies:
        return
    blocks = list(BLOCK_ORDER)
    source_blocks: list[str] = []
    for frequency in frequencies:
        frame = block_flows_by_frequency[frequency]
        if "shock_block" not in frame.columns:
            continue
        source_blocks.extend(str(block) for block in frame["shock_block"].dropna().unique())
    source_blocks = [block for block in blocks if block in set(source_blocks)]
    fig, axes = plt.subplots(
        len(blocks),
        len(frequencies),
        figsize=(5.0 * len(frequencies), 2.05 * len(blocks)),
        squeeze=False,
    )
    for col, frequency in enumerate(frequencies):
        frame = block_flows_by_frequency[frequency].copy()
        required = {"window_start", "receiver_block", "shock_block", value_col}
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame[frame["receiver_block"].ne(frame["shock_block"])]
        frame["window_start"] = pd.to_datetime(frame["window_start"], utc=True)
        for row, receiver_block in enumerate(blocks):
            ax = axes[row, col]
            subset = frame[frame["receiver_block"].eq(receiver_block)]
            for shock_block in blocks:
                if shock_block == receiver_block:
                    continue
                line = subset[subset["shock_block"].eq(shock_block)].sort_values("window_start")
                if line.empty:
                    continue
                ax.plot(
                    line["window_start"],
                    line[value_col],
                    linewidth=1.15,
                    linestyle=BLOCK_LINE_STYLES.get(shock_block, BLOCK_LINE_STYLES["unknown"]),
                    label=_block_label(shock_block),
                    color=BLOCK_LINE_COLORS.get(shock_block, BLOCK_LINE_COLORS["unknown"]),
                )
            if row == 0:
                ax.set_title(FREQUENCY_LABELS.get(frequency, frequency))
            if col == 0:
                ax.set_ylabel(f"{_block_label(receiver_block)}\nshare")
            ax.grid(alpha=0.18)
            if row == len(blocks) - 1:
                ax.set_xlabel("Window start")
            else:
                ax.tick_params(labelbottom=False)
            _format_date_axis(ax)
    handles = _block_line_legend_handles(source_blocks, prefix="Shock source: ")
    labels = [handle.get_label() for handle in handles]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0.065, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_block_net_frequency_grid(
    frame: pd.DataFrame,
    *,
    variant: str,
    title: str,
    path: Path,
) -> None:
    required = {"frequency", "variant", "window_start", "block", "net_to_others"}
    if frame.empty or not required.issubset(frame.columns):
        return
    data = frame[frame["variant"].eq(variant)].copy()
    if data.empty:
        return
    data["window_start"] = pd.to_datetime(data["window_start"], utc=True)
    frequencies = [frequency for frequency in FREQUENCY_ORDER_ARTICLE if frequency in set(data["frequency"])]
    blocks = [block for block in BLOCK_ORDER if block in set(data["block"])]
    if not frequencies or not blocks:
        return
    fig, axes = plt.subplots(
        len(blocks),
        len(frequencies),
        figsize=(5.0 * len(frequencies), 1.8 * len(blocks)),
        squeeze=False,
    )
    for col, frequency in enumerate(frequencies):
        for row, block in enumerate(blocks):
            ax = axes[row, col]
            line = data[data["frequency"].eq(frequency) & data["block"].eq(block)].sort_values("window_start")
            if not line.empty:
                ax.plot(
                    line["window_start"],
                    line["net_to_others"],
                    linewidth=1.25,
                    linestyle=BLOCK_LINE_STYLES.get(block, BLOCK_LINE_STYLES["unknown"]),
                    color=BLOCK_LINE_COLORS.get(block, BLOCK_LINE_COLORS["unknown"]),
                )
            ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.8)
            if row == 0:
                ax.set_title(FREQUENCY_LABELS.get(frequency, frequency))
            if col == 0:
                ax.set_ylabel(f"{_block_label(block)}\nTO - FROM")
            ax.grid(alpha=0.18)
            if row == len(blocks) - 1:
                ax.set_xlabel("Window start")
            else:
                ax.tick_params(labelbottom=False)
            _format_date_axis(ax)
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_price_discovery_frequency_grid(
    windows_by_frequency: dict[str, pd.DataFrame],
    *,
    title: str,
    path: Path,
) -> None:
    frequencies = [frequency for frequency in FREQUENCY_ORDER_ARTICLE if frequency in windows_by_frequency]
    if not frequencies:
        return
    frames: list[pd.DataFrame] = []
    for frequency in frequencies:
        frame = windows_by_frequency[frequency].copy()
        if not frame.empty:
            frame["frequency"] = frequency
            frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    required = {"frequency", "window_start", "metric", "right_block", "left_share"}
    if not required.issubset(data.columns):
        return
    data["window_start"] = pd.to_datetime(data["window_start"], utc=True)
    data["left_share"] = pd.to_numeric(data["left_share"], errors="coerce")
    data = data.dropna(subset=["left_share"])
    metrics = _ordered_metrics(data["metric"].astype(str).unique())
    right_blocks = [block for block in BLOCK_ORDER if block in set(data["right_block"].astype(str))]
    right_blocks.extend(sorted(set(data["right_block"].astype(str)) - set(right_blocks)))
    if not metrics or not right_blocks:
        return

    fig, axes = plt.subplots(
        len(right_blocks),
        len(frequencies),
        figsize=(5.2 * len(frequencies), 2.35 * len(right_blocks)),
        squeeze=False,
    )
    for col, frequency in enumerate(frequencies):
        for row, right_block in enumerate(right_blocks):
            ax = axes[row, col]
            subset = data[data["frequency"].eq(frequency) & data["right_block"].astype(str).eq(right_block)]
            for metric in metrics:
                line = subset[subset["metric"].astype(str).eq(metric)].sort_values("window_start")
                if line.empty:
                    continue
                _plot_long_run_share_line(
                    ax,
                    line,
                    color=METRIC_LINE_COLORS.get(metric, BLOCK_LINE_COLORS["unknown"]),
                    label=_metric_label(metric),
                    linewidth=1.35,
                    linestyle=METRIC_LINE_STYLES.get(metric, "-"),
                    band_alpha=0.08,
                )
            ax.axhline(0.5, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.8)
            ax.set_ylim(0.0, 1.0)
            if row == 0:
                ax.set_title(FREQUENCY_LABELS.get(frequency, frequency))
            if col == 0:
                ax.set_ylabel(
                    f"vs {_block_label(right_block)}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=28,
                )
            ax.grid(alpha=0.16)
            if row == len(right_blocks) - 1:
                ax.set_xlabel("Window start")
            else:
                ax.tick_params(labelbottom=False)
            _format_date_axis(ax)
    handles = _metric_line_legend_handles(metrics)
    labels = [handle.get_label() for handle in handles]
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(2, len(handles)), frameon=False)
    _set_suptitle(fig, title)
    fig.tight_layout(rect=(0.025, 0.07, 1, SUPTITLE_LAYOUT_TOP))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _annotate_numeric_matrix(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    threshold = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = values[row, col]
            color = "white" if value >= threshold else "#1f2933"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)


def _plot_long_run_share_line(
    ax: plt.Axes,
    line: pd.DataFrame,
    *,
    color: str,
    label: str,
    linewidth: float,
    linestyle: str = "-",
    band_alpha: float = 0.12,
) -> None:
    x_values = mdates.date2num(pd.to_datetime(line["window_start"], utc=True).dt.tz_convert(None))
    shares = pd.to_numeric(line["left_share"], errors="coerce")
    if {"left_lower", "left_upper"}.issubset(line.columns):
        lower = pd.to_numeric(line["left_lower"], errors="coerce")
        upper = pd.to_numeric(line["left_upper"], errors="coerce")
        valid_band = lower.notna() & upper.notna()
        if valid_band.any():
            ax.fill_between(
                x_values[valid_band.to_numpy()],
                lower[valid_band].to_numpy(dtype=float),
                upper[valid_band].to_numpy(dtype=float),
                color=color,
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x_values[valid_band.to_numpy()],
                lower[valid_band].to_numpy(dtype=float),
                color=color,
                alpha=0.35,
                linewidth=0.55,
                linestyle=linestyle,
                zorder=2,
            )
            ax.plot(
                x_values[valid_band.to_numpy()],
                upper[valid_band].to_numpy(dtype=float),
                color=color,
                alpha=0.35,
                linewidth=0.55,
                linestyle=linestyle,
                zorder=2,
            )
    valid_line = shares.notna()
    if valid_line.any():
        ax.plot(
            x_values[valid_line.to_numpy()],
            shares[valid_line].to_numpy(dtype=float),
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            color=color,
            zorder=3,
        )


def _block_label(block: str) -> str:
    return BLOCK_LABELS.get(str(block), str(block))


def _block_line_legend_handles(blocks: list[str], *, prefix: str = "") -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=BLOCK_LINE_COLORS.get(block, BLOCK_LINE_COLORS["unknown"]),
            linestyle=BLOCK_LINE_STYLES.get(block, BLOCK_LINE_STYLES["unknown"]),
            linewidth=1.7,
            label=f"{prefix}{_block_label(block)}",
        )
        for block in blocks
    ]


def _metric_line_legend_handles(metrics: list[str]) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=METRIC_LINE_COLORS.get(metric, BLOCK_LINE_COLORS["unknown"]),
            linestyle=METRIC_LINE_STYLES.get(metric, "-"),
            linewidth=1.7,
            label=_metric_label(metric),
        )
        for metric in metrics
    ]


def _set_suptitle(fig: plt.Figure, title: str, *, heatmap: bool = False) -> None:
    fig.suptitle(title, y=HEATMAP_SUPTITLE_Y if heatmap else SUPTITLE_Y)


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _downsample(frame: pd.DataFrame, *, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=max_points, dtype=int)
    positions = np.unique(positions)
    return frame.iloc[positions]


def _circular_block_positions(assets: list[str]) -> dict[str, tuple[float, float]]:
    grouped: list[str] = []
    for block in BLOCK_ORDER:
        grouped.extend([asset for asset in assets if ASSET_BLOCKS.get(asset) == block])
    grouped.extend([asset for asset in assets if asset not in grouped])

    n_assets = len(grouped)
    if n_assets == 0:
        return {}
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_assets, endpoint=False)
    return {
        asset: (float(np.cos(angle)), float(np.sin(angle)))
        for asset, angle in zip(grouped, angles, strict=True)
    }


def _ordered_metrics(metrics: list[str] | np.ndarray) -> list[str]:
    preferred = ["gis", "hasbrouck_proxy"]
    available = [str(metric) for metric in metrics]
    ordered = [metric for metric in preferred if metric in available]
    ordered.extend(sorted(metric for metric in available if metric not in ordered))
    return ordered


def _metric_label(metric: str) -> str:
    labels = {
        "gis": "GIS",
        "hasbrouck_proxy": "Hasbrouck midpoint/bounds",
    }
    return labels.get(metric, metric)
