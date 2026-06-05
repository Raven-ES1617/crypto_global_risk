from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from calculations.ci_analysis import CI_WINDOW_CONFIGS
from calculations.config import DEFAULT_ARTIFACT_ROOT, FREQUENCY_CONFIGS
from calculations.gfevd_analysis import block_spillover_table
from calculations.plots import POSITIVE_CMAP, plot_block_flow_dynamics, plot_gfevd_network
from metrics import calculate_gfevd


DEFAULT_FREQUENCIES = ("1min", "1h", "1d")


@dataclass(frozen=True)
class WindowEstimate:
    frequency: str
    window_id: int
    window_start: str
    window_end: str
    rows_available: int
    status: str
    total_connectedness: float | None
    lag_order_diff: int | None
    coint_rank: int | None
    error: str
    matrix: pd.DataFrame | None

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("matrix")
        return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.artifacts_root)
    written: list[Path] = []

    for frequency in args.frequencies:
        print(f"[window-gifs] {frequency}", flush=True)
        panel = pd.read_parquet(_panel_path(root, frequency))
        estimates = estimate_window_matrices(panel, frequency=frequency, max_frames=args.max_frames)

        data_dir = root / "gfevd" / "window_gifs"
        matrix_figure_dir = root / "figures" / "gfevd" / "window_gifs"
        network_figure_dir = root / "figures" / "gfevd" / "networks" / "window_gifs"
        dynamics_figure_dir = root / "figures" / "gfevd" / "dynamics"
        data_dir.mkdir(parents=True, exist_ok=True)
        matrix_figure_dir.mkdir(parents=True, exist_ok=True)
        network_figure_dir.mkdir(parents=True, exist_ok=True)
        dynamics_figure_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = data_dir / f"window_metadata_{frequency}.csv"
        pd.DataFrame([item.metadata() for item in estimates]).to_csv(metadata_path, index=False)
        written.append(metadata_path)

        ok_estimates = [item for item in estimates if item.status == "ok" and item.matrix is not None]
        if not ok_estimates:
            print(f"  no successful windows for {frequency}", flush=True)
            continue

        matrix_path = data_dir / f"matrix_windows_{frequency}.csv"
        matrix_rows = []
        block_rows = []
        for item in ok_estimates:
            assert item.matrix is not None
            matrix_rows.extend(_matrix_long(item.matrix, item))
            block_rows.extend(_block_flow_long(block_spillover_table(item.matrix), item))
        pd.DataFrame(matrix_rows).to_csv(matrix_path, index=False)
        written.append(matrix_path)

        block_path = data_dir / f"block_flow_windows_{frequency}.csv"
        block_frame = pd.DataFrame(block_rows)
        block_frame.to_csv(block_path, index=False)
        written.append(block_path)

        matrix_gif = matrix_figure_dir / f"matrix_windows_{frequency}.gif"
        log_matrix_gif = matrix_figure_dir / f"matrix_windows_log_{frequency}.gif"
        network_gif = network_figure_dir / f"network_windows_{frequency}.gif"
        block_flow_fig = dynamics_figure_dir / f"block_flows_{frequency}.png"
        block_flow_adjusted_fig = dynamics_figure_dir / f"block_flows_adjusted_{frequency}.png"
        _remove_old_network_artifacts(matrix_figure_dir / f"network_windows_{frequency}.gif")

        render_matrix_gif(ok_estimates, path=matrix_gif, duration_ms=args.duration_ms, log_scale=False)
        render_matrix_gif(ok_estimates, path=log_matrix_gif, duration_ms=args.duration_ms, log_scale=True)
        render_network_gif(
            ok_estimates,
            path=network_gif,
            duration_ms=args.duration_ms,
            top_n=args.network_top_n,
            min_weight=args.network_min_weight,
        )
        plot_block_flow_dynamics(block_frame, title=f"Block total information flows, {frequency}", path=block_flow_fig)
        plot_block_flow_dynamics(
            block_frame,
            title=f"Block adjusted information flows, {frequency}",
            path=block_flow_adjusted_fig,
            value_col="average_pair_share",
        )
        written.extend([matrix_gif, log_matrix_gif, network_gif, block_flow_fig, block_flow_adjusted_fig])
        print(f"  frames={len(ok_estimates)} -> {matrix_gif}, {network_gif}", flush=True)

    manifest_path = root / "reports" / "window_gif_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": [str(path.relative_to(root)) for path in written]}).to_csv(manifest_path, index=False)
    print(f"[done] window GIF artifacts written to {root}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling-window GFEVD matrix and network GIFs.")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--frequencies", nargs="+", choices=sorted(FREQUENCY_CONFIGS), default=list(DEFAULT_FREQUENCIES))
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--duration-ms", type=int, default=700)
    parser.add_argument("--network-top-n", type=int, default=50)
    parser.add_argument("--network-min-weight", type=float, default=0.01)
    return parser.parse_args(argv)


def estimate_window_matrices(prices: pd.DataFrame, *, frequency: str, max_frames: int | None) -> list[WindowEstimate]:
    frequency_config = FREQUENCY_CONFIGS[frequency]
    window_config = CI_WINDOW_CONFIGS[frequency]
    windows = select_even_windows(
        prices,
        window=pd.Timedelta(window_config.window),
        step=pd.Timedelta(window_config.step),
        min_obs=window_config.min_obs,
        max_windows=max_frames,
    )

    estimates: list[WindowEstimate] = []
    for window_id, (window_start, window_end, window_prices) in enumerate(windows, start=1):
        base = {
            "frequency": frequency,
            "window_id": window_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "rows_available": int(len(window_prices)),
        }
        try:
            result = calculate_gfevd(
                window_prices,
                horizon=frequency_config.horizon,
                max_lags=window_config.max_lags,
                lag_method="bic",
                coint_rank="auto",
                max_obs=window_config.max_obs,
            )
            estimates.append(
                WindowEstimate(
                    **base,
                    status="ok",
                    total_connectedness=float(result.total_connectedness),
                    lag_order_diff=int(result.lag_order_diff),
                    coint_rank=int(result.coint_rank),
                    error="",
                    matrix=result.table,
                )
            )
        except Exception as exc:
            estimates.append(
                WindowEstimate(
                    **base,
                    status="error",
                    total_connectedness=None,
                    lag_order_diff=None,
                    coint_rank=None,
                    error=str(exc),
                    matrix=None,
                )
            )
    return estimates


def select_even_windows(
    prices: pd.DataFrame,
    *,
    window: pd.Timedelta,
    step: pd.Timedelta,
    min_obs: int,
    max_windows: int | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    if prices.empty:
        return []

    all_windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    current_start = prices.index.min()
    final_end = prices.index.max()
    while current_start + window <= final_end:
        current_end = current_start + window
        sample = prices.loc[(prices.index >= current_start) & (prices.index <= current_end)]
        if len(sample) >= min_obs:
            all_windows.append((current_start, current_end, sample))
        current_start += step

    if max_windows is None or len(all_windows) <= max_windows:
        return all_windows

    positions = np.linspace(0, len(all_windows) - 1, num=max_windows)
    indexes = sorted({int(round(position)) for position in positions})
    return [all_windows[index] for index in indexes]


def render_matrix_gif(
    estimates: list[WindowEstimate],
    *,
    path: Path,
    duration_ms: int,
    log_scale: bool,
) -> None:
    frame_dir = path.with_suffix("") / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    matrices = [item.matrix for item in estimates if item.matrix is not None]
    assert matrices

    if log_scale:
        values = [np.log10(np.clip(matrix.to_numpy(dtype=float), 1e-6, None)) for matrix in matrices]
    else:
        values = [matrix.to_numpy(dtype=float) for matrix in matrices]
    vmin = float(np.nanmin([np.nanmin(item) for item in values]))
    vmax = float(np.nanmax([np.nanmax(item) for item in values]))

    frame_paths: list[Path] = []
    for item, matrix, matrix_values in zip(estimates, matrices, values, strict=True):
        frame_path = frame_dir / f"frame_{item.window_id:03d}.png"
        _plot_matrix_frame(
            matrix,
            matrix_values,
            title=_window_title("GFEVD matrix", item, log_scale=log_scale),
            path=frame_path,
            vmin=vmin,
            vmax=vmax,
            colorbar_label="log10(GFEVD share)" if log_scale else "GFEVD share",
        )
        frame_paths.append(frame_path)
    write_gif(frame_paths, path=path, duration_ms=duration_ms)
    shutil.rmtree(frame_dir, ignore_errors=True)


def render_network_gif(
    estimates: list[WindowEstimate],
    *,
    path: Path,
    duration_ms: int,
    top_n: int,
    min_weight: float,
) -> None:
    frame_dir = path.with_suffix("") / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for item in estimates:
        if item.matrix is None:
            continue
        frame_path = frame_dir / f"frame_{item.window_id:03d}.png"
        plot_gfevd_network(
            item.matrix,
            title=_window_title("GFEVD network", item),
            path=frame_path,
            top_n=top_n,
            min_weight=min_weight,
        )
        frame_paths.append(frame_path)
    write_gif(frame_paths, path=path, duration_ms=duration_ms)
    shutil.rmtree(frame_dir, ignore_errors=True)


def write_gif(frame_paths: list[Path], *, path: Path, duration_ms: int) -> None:
    if not frame_paths:
        return
    images = [Image.open(frame).convert("RGB") for frame in frame_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    padded = []
    for image in images:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        padded.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    padded[0].save(path, save_all=True, append_images=padded[1:], duration=duration_ms, loop=0, optimize=True)
    for image in images:
        image.close()


def _plot_matrix_frame(
    matrix: pd.DataFrame,
    values: np.ndarray,
    *,
    title: str,
    path: Path,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, cmap=POSITIVE_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Shock source")
    ax.set_ylabel("Receiver")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _matrix_long(matrix: pd.DataFrame, window: WindowEstimate) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for receiver in matrix.index:
        for source in matrix.columns:
            rows.append(
                {
                    "frequency": window.frequency,
                    "window_id": window.window_id,
                    "window_start": window.window_start,
                    "window_end": window.window_end,
                    "receiver": str(receiver),
                    "shock_source": str(source),
                    "value": float(matrix.loc[receiver, source]),
                }
            )
    return rows


def _block_flow_long(block_table: pd.DataFrame, window: WindowEstimate) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in block_table.iterrows():
        rows.append(
            {
                "frequency": window.frequency,
                "window_id": window.window_id,
                "window_start": window.window_start,
                "window_end": window.window_end,
                "receiver_block": str(row["receiver_block"]),
                "shock_block": str(row["shock_block"]),
                "receiver_assets": int(row["receiver_assets"]),
                "shock_assets": int(row["shock_assets"]),
                "pair_count": int(row["pair_count"]),
                "share_sum": float(row["share_sum"]),
                "average_receiver_share": float(row["average_receiver_share"]),
                "average_pair_share": float(row["average_pair_share"]),
            }
        )
    return rows


def _window_title(prefix: str, window: WindowEstimate, *, log_scale: bool = False) -> str:
    start = pd.Timestamp(window.window_start).strftime("%Y-%m-%d")
    end = pd.Timestamp(window.window_end).strftime("%Y-%m-%d")
    suffix = " log scale" if log_scale else ""
    return (
        f"{prefix}{suffix}, {window.frequency}, window {window.window_id}: {start} -> {end}\n"
        f"TCI={window.total_connectedness:.4f}, rows={window.rows_available}"
    )


def _panel_path(root: Path, frequency: str) -> Path:
    if frequency == "1s":
        return root / "panels" / "panel_1s_sample.parquet"
    return root / "panels" / f"panel_{frequency}.parquet"


def _remove_old_network_artifacts(old_gif: Path) -> None:
    if old_gif.exists():
        old_gif.unlink()
    old_frame_dir = old_gif.with_suffix("")
    if old_frame_dir.exists():
        shutil.rmtree(old_frame_dir)


if __name__ == "__main__":
    raise SystemExit(main())
