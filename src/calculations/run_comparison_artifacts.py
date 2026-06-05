from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from calculations.artifacts import ArtifactEntry
from calculations.ci_analysis import (
    compare_ci_summaries,
    run_gfevd_window_ci,
    run_price_discovery_window_ci,
    save_ci_outputs,
    summarize_ci,
)
from calculations.config import DEFAULT_ARTIFACT_ROOT, PERIOD_SPLITS
from calculations.plots import (
    plot_gfevd_log_heatmap,
    plot_gfevd_network,
    plot_matrix_difference,
    plot_pre_post_heatmaps,
    plot_price_discovery_block_dynamics,
    plot_price_discovery_share_diff,
    plot_price_discovery_share_pre_post,
    plot_tci_confidence,
)
from calculations.price_discovery import aggregate_price_discovery_blocks


FULL_FREQUENCIES = ("1min", "1h", "1d")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.artifacts_root)
    entries: list[ArtifactEntry] = []

    all_tci_ci: list[pd.DataFrame] = []
    if not args.skip_gfevd_ci:
        print("[ci] empirical GFEVD window intervals", flush=True)
    for frequency in ([] if args.skip_gfevd_ci else args.ci_frequencies):
        if frequency == "1s":
            period_ranges = {"sample": (None, None)}
            panel_path = root / "panels" / "panel_1s_sample.parquet"
        else:
            period_ranges = {
                "pre_covid": PERIOD_SPLITS["pre_covid"],
                "covid_and_after": PERIOD_SPLITS["covid_and_after"],
            }
            panel_path = root / "panels" / f"panel_{frequency}.parquet"
        prices = pd.read_parquet(panel_path)
        for period, bounds in period_ranges.items():
            period_prices = prices if bounds == (None, None) else _slice_utc(prices, bounds[0], bounds[1])
            if len(period_prices) < 80:
                continue
            print(f"[ci] {frequency} {period}: rows={len(period_prices)}", flush=True)
            outputs = run_gfevd_window_ci(
                period_prices,
                frequency=frequency,
                period=period,
                max_windows=args.max_windows_per_period,
            )
            written = save_ci_outputs(
                outputs,
                output_dir=root / "gfevd" / "confidence",
                frequency=frequency,
                period=period,
            )
            entries.extend(_entry(root, path, "csv", f"Window CI output for {period}, {frequency}.") for path in written)
            tci_ci = outputs["tci_ci"].copy()
            if not tci_ci.empty:
                tci_ci.insert(0, "period", period)
                tci_ci.insert(0, "frequency", frequency)
                all_tci_ci.append(tci_ci)

    if all_tci_ci:
        all_tci = pd.concat(all_tci_ci, ignore_index=True)
        all_tci_path = root / "gfevd" / "confidence" / "tci_ci_all.csv"
        all_tci_path.parent.mkdir(parents=True, exist_ok=True)
        all_tci.to_csv(all_tci_path, index=False)
        entries.append(_entry(root, all_tci_path, "csv", "Total connectedness empirical CI across frequencies."))
        tci_fig = root / "figures" / "gfevd" / "confidence" / "tci_ci_all.png"
        plot_tci_confidence(all_tci, title="Empirical window CI for total connectedness", path=tci_fig)
        entries.append(_entry(root, tci_fig, "png", "Total connectedness empirical CI chart."))

    if not args.skip_price_discovery_ci:
        print("[ci] empirical price discovery window intervals", flush=True)
        for frequency in args.ci_frequencies:
            if frequency == "1s":
                period_ranges = {"sample": (None, None)}
                panel_path = root / "panels" / "panel_1s_sample.parquet"
            else:
                period_ranges = {
                    "pre_covid": PERIOD_SPLITS["pre_covid"],
                    "covid_and_after": PERIOD_SPLITS["covid_and_after"],
                }
                panel_path = root / "panels" / f"panel_{frequency}.parquet"
            prices = pd.read_parquet(panel_path)
            for period, bounds in period_ranges.items():
                period_prices = prices if bounds == (None, None) else _slice_utc(prices, bounds[0], bounds[1])
                if len(period_prices) < 80:
                    continue
                print(f"[ci-price] {frequency} {period}: rows={len(period_prices)}", flush=True)
                outputs = run_price_discovery_window_ci(
                    period_prices,
                    frequency=frequency,
                    period=period,
                    max_windows=args.max_windows_per_period,
                )
                written = save_ci_outputs(
                    outputs,
                    output_dir=root / "price_discovery" / "confidence",
                    frequency=frequency,
                    period=period,
                )
                entries.extend(
                    _entry(root, path, "csv", f"Price discovery window CI output for {period}, {frequency}.")
                    for path in written
                )

    if not all_tci_ci:
        existing_tci = root / "gfevd" / "confidence" / "tci_ci_all.csv"
        if existing_tci.exists():
            all_tci_ci.append(pd.read_csv(existing_tci))

    if args.skip_gfevd_ci:
        entries.extend(
            _entry(root, path, "csv", "Existing GFEVD confidence artifact.")
            for path in sorted((root / "gfevd" / "confidence").glob("*.csv"))
        )
    if args.skip_price_discovery_ci:
        entries.extend(
            _entry(root, path, "csv", "Existing price-discovery confidence artifact.")
            for path in sorted((root / "price_discovery" / "confidence").glob("*.csv"))
        )

    print("[plots] pre/post matrices and networks", flush=True)
    entries.extend(_write_prepost_and_network_artifacts(root))
    entries.extend(_write_main_matrix_artifacts(root))
    entries.extend(_write_price_discovery_long_run_artifacts(root))

    report_path = root / "reports" / "pre_post_ci_network_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_build_report(all_tci_ci), encoding="utf-8")
    entries.append(_entry(root, report_path, "markdown", "Summary of pre/post, CI, and network artifacts."))

    entries = _dedupe_entries(entries)
    manifest_path = root / "reports" / "comparison_ci_network_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": [entry.__dict__ for entry in entries]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] comparison artifacts written to {root}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre/post, CI, and network GFEVD artifacts.")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--max-windows-per-period", type=int, default=16)
    parser.add_argument("--skip-gfevd-ci", action="store_true")
    parser.add_argument("--skip-price-discovery-ci", action="store_true")
    parser.add_argument(
        "--ci-frequencies",
        nargs="+",
        choices=("1s", "1min", "1h", "1d"),
        default=["1min", "1h", "1d"],
    )
    return parser.parse_args(argv)


def _read_main_matrix(root: Path, period: str, frequency: str) -> pd.DataFrame:
    return pd.read_csv(root / "gfevd" / "matrices" / f"matrix_{period}_{frequency}.csv", index_col=0)


def _read_period_matrix(root: Path, period: str, frequency: str) -> pd.DataFrame:
    return pd.read_csv(root / "gfevd" / "periods" / f"matrix_{period}_{frequency}.csv", index_col=0)


def _read_period_block_matrix(root: Path, period: str, frequency: str, *, adjusted: bool = False) -> pd.DataFrame:
    prefix = "block_matrix_adjusted" if adjusted else "block_matrix"
    return pd.read_csv(root / "gfevd" / "periods" / f"{prefix}_{period}_{frequency}.csv", index_col=0)


def _read_block_ci(root: Path, period: str, frequency: str, *, adjusted: bool = False) -> pd.DataFrame:
    prefix = "block_adjusted_ci" if adjusted else "block_ci"
    path = root / "gfevd" / "confidence" / f"{prefix}_{period}_{frequency}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_price_discovery_ci(root: Path, period: str, frequency: str) -> pd.DataFrame:
    path = root / "price_discovery" / "confidence" / f"left_share_ci_{period}_{frequency}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_price_discovery_windows(root: Path, period: str, frequency: str) -> pd.DataFrame:
    path = root / "price_discovery" / "confidence" / f"pairwise_windows_{period}_{frequency}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_prepost_and_network_artifacts(root: Path) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for frequency in FULL_FREQUENCIES:
        pre = _read_period_matrix(root, "pre_covid", frequency)
        post = _read_period_matrix(root, "covid_and_after", frequency)
        pre_blocks = _read_period_block_matrix(root, "pre_covid", frequency)
        post_blocks = _read_period_block_matrix(root, "covid_and_after", frequency)
        pre_blocks_adjusted = _read_period_block_matrix(root, "pre_covid", frequency, adjusted=True)
        post_blocks_adjusted = _read_period_block_matrix(root, "covid_and_after", frequency, adjusted=True)
        pre_block_ci = _read_block_ci(root, "pre_covid", frequency)
        post_block_ci = _read_block_ci(root, "covid_and_after", frequency)
        pre_block_adjusted_ci = _read_block_ci(root, "pre_covid", frequency, adjusted=True)
        post_block_adjusted_ci = _read_block_ci(root, "covid_and_after", frequency, adjusted=True)
        block_diff_ci = compare_ci_summaries(pre_block_ci, post_block_ci, ["receiver_block", "shock_block"])
        block_adjusted_diff_ci = compare_ci_summaries(
            pre_block_adjusted_ci,
            post_block_adjusted_ci,
            ["receiver_block", "shock_block"],
        )
        block_diff_ci_path = root / "gfevd" / "confidence" / f"block_diff_ci_covid_minus_pre_{frequency}.csv"
        block_adjusted_diff_ci_path = (
            root / "gfevd" / "confidence" / f"block_adjusted_diff_ci_covid_minus_pre_{frequency}.csv"
        )
        if not block_diff_ci.empty:
            block_diff_ci_path.parent.mkdir(parents=True, exist_ok=True)
            block_diff_ci.to_csv(block_diff_ci_path, index=False)
        if not block_adjusted_diff_ci.empty:
            block_adjusted_diff_ci_path.parent.mkdir(parents=True, exist_ok=True)
            block_adjusted_diff_ci.to_csv(block_adjusted_diff_ci_path, index=False)

        prepost_path = root / "figures" / "gfevd" / "pre_post" / f"matrix_pre_post_{frequency}.png"
        prepost_log_path = root / "figures" / "gfevd" / "pre_post" / f"matrix_pre_post_log_{frequency}.png"
        diff_path = root / "figures" / "gfevd" / "pre_post" / f"matrix_diff_covid_minus_pre_{frequency}.png"
        block_prepost_path = root / "figures" / "gfevd" / "pre_post" / f"block_matrix_pre_post_{frequency}.png"
        block_diff_path = root / "figures" / "gfevd" / "pre_post" / f"block_matrix_diff_covid_minus_pre_{frequency}.png"
        block_adjusted_prepost_path = (
            root / "figures" / "gfevd" / "pre_post" / f"block_matrix_adjusted_pre_post_{frequency}.png"
        )
        block_adjusted_diff_path = (
            root / "figures" / "gfevd" / "pre_post" / f"block_matrix_adjusted_diff_covid_minus_pre_{frequency}.png"
        )
        pre_block_labels = _block_ci_labels(pre_blocks, pre_block_ci)
        post_block_labels = _block_ci_labels(post_blocks, post_block_ci)
        block_diff_labels = _block_diff_labels(post_blocks.reindex_like(pre_blocks) - pre_blocks, block_diff_ci)
        pre_block_adjusted_labels = _block_ci_labels(pre_blocks_adjusted, pre_block_adjusted_ci)
        post_block_adjusted_labels = _block_ci_labels(post_blocks_adjusted, post_block_adjusted_ci)
        block_adjusted_diff_labels = _block_diff_labels(
            post_blocks_adjusted.reindex_like(pre_blocks_adjusted) - pre_blocks_adjusted,
            block_adjusted_diff_ci,
        )

        plot_pre_post_heatmaps(pre, post, title=f"GFEVD pre/post, {frequency}", path=prepost_path)
        plot_pre_post_heatmaps(pre, post, title=f"GFEVD pre/post log scale, {frequency}", path=prepost_log_path, log_scale=True)
        plot_matrix_difference(pre, post, title=f"GFEVD difference, covid_and_after - pre_covid, {frequency}", path=diff_path)
        plot_pre_post_heatmaps(
            pre_blocks,
            post_blocks,
            title=f"Block total GFEVD pre/post with 95% window CI, {frequency}",
            path=block_prepost_path,
            annotations=(pre_block_labels, post_block_labels),
        )
        plot_matrix_difference(
            pre_blocks,
            post_blocks,
            title=f"Block total GFEVD difference with p-value stars, covid_and_after - pre_covid, {frequency}",
            path=block_diff_path,
            annotations=block_diff_labels,
        )
        plot_pre_post_heatmaps(
            pre_blocks_adjusted,
            post_blocks_adjusted,
            title=f"Block adjusted GFEVD pre/post with 95% window CI, {frequency}",
            path=block_adjusted_prepost_path,
            annotations=(pre_block_adjusted_labels, post_block_adjusted_labels),
        )
        plot_matrix_difference(
            pre_blocks_adjusted,
            post_blocks_adjusted,
            title=f"Block adjusted GFEVD difference with p-value stars, covid_and_after - pre_covid, {frequency}",
            path=block_adjusted_diff_path,
            annotations=block_adjusted_diff_labels,
        )
        entries.extend(
            [
                _entry(root, prepost_path, "png", f"Pre/post GFEVD matrices for {frequency}."),
                _entry(root, prepost_log_path, "png", f"Pre/post log-scale GFEVD matrices for {frequency}."),
                _entry(root, diff_path, "png", f"Post-minus-pre GFEVD matrix difference for {frequency}."),
                _entry(root, block_prepost_path, "png", f"Pre/post block total GFEVD matrices for {frequency}."),
                _entry(root, block_diff_path, "png", f"Post-minus-pre block total GFEVD matrix difference for {frequency}."),
                _entry(root, block_adjusted_prepost_path, "png", f"Pre/post block adjusted GFEVD matrices for {frequency}."),
                _entry(
                    root,
                    block_adjusted_diff_path,
                    "png",
                    f"Post-minus-pre block adjusted GFEVD matrix difference for {frequency}.",
                ),
            ]
        )
        if block_diff_ci_path.exists():
            entries.append(_entry(root, block_diff_ci_path, "csv", f"Approximate post-minus-pre block p-values for {frequency}."))
        if block_adjusted_diff_ci_path.exists():
            entries.append(
                _entry(
                    root,
                    block_adjusted_diff_ci_path,
                    "csv",
                    f"Approximate post-minus-pre block adjusted p-values for {frequency}.",
                )
            )

        for period, matrix in (("pre_covid", pre), ("covid_and_after", post)):
            network_path = root / "figures" / "gfevd" / "networks" / f"network_{period}_{frequency}.png"
            plot_gfevd_network(matrix, title=f"GFEVD network, {period}, {frequency}", path=network_path)
            entries.append(_entry(root, network_path, "png", f"GFEVD network for {period}, {frequency}."))
    return entries


def _write_main_matrix_artifacts(root: Path) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    for frequency, period in (("1s", "sample"), ("1min", "full"), ("1h", "full"), ("1d", "full")):
        matrix_file = root / "gfevd" / "matrices" / f"matrix_{period}_{frequency}.csv"
        if not matrix_file.exists():
            continue
        matrix = pd.read_csv(matrix_file, index_col=0)
        log_path = root / "figures" / "gfevd" / "log_matrices" / f"matrix_{period}_{frequency}_log.png"
        network_path = root / "figures" / "gfevd" / "networks" / f"network_{period}_{frequency}.png"
        plot_gfevd_log_heatmap(matrix, title=f"GFEVD log scale, {period}, {frequency}", path=log_path)
        plot_gfevd_network(matrix, title=f"GFEVD network, {period}, {frequency}", path=network_path)
        entries.append(_entry(root, log_path, "png", f"Log-scale GFEVD matrix for {period}, {frequency}."))
        entries.append(_entry(root, network_path, "png", f"GFEVD network for {period}, {frequency}."))
    return entries


def _write_price_discovery_long_run_artifacts(root: Path) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    summary_dir = root / "price_discovery" / "summary"
    figure_dir = root / "figures" / "price_discovery" / "long_run"

    for frequency in FULL_FREQUENCIES:
        pre_ci = _read_price_discovery_ci(root, "pre_covid", frequency)
        post_ci = _read_price_discovery_ci(root, "covid_and_after", frequency)
        if not pre_ci.empty and not post_ci.empty:
            pair_ci = pd.concat([pre_ci, post_ci], ignore_index=True)
            pair_ci_path = summary_dir / f"long_run_pair_ci_{frequency}.csv"
            pair_ci_path.parent.mkdir(parents=True, exist_ok=True)
            pair_ci.to_csv(pair_ci_path, index=False)
            entries.append(_entry(root, pair_ci_path, "csv", f"Pairwise GIS/Hasbrouck empirical CI for {frequency}."))

            pair_diff = compare_ci_summaries(
                pre_ci,
                post_ci,
                ["frequency", "pair", "metric", "left_asset", "right_asset"],
            )
            pair_diff_path = summary_dir / f"long_run_pair_diff_covid_minus_pre_{frequency}.csv"
            if not pair_diff.empty:
                pair_diff.to_csv(pair_diff_path, index=False)
                entries.append(
                    _entry(root, pair_diff_path, "csv", f"Pairwise GIS/Hasbrouck post-minus-pre tests for {frequency}.")
                )

            pair_prepost_fig = figure_dir / f"pair_pre_post_{frequency}.png"
            pair_diff_fig = figure_dir / f"pair_diff_covid_minus_pre_{frequency}.png"
            plot_price_discovery_share_pre_post(
                pair_ci,
                title=f"Long-run price discovery shares, pre/post, {frequency}",
                path=pair_prepost_fig,
                label_col="pair",
                x_label="Crypto-side long-run information share",
            )
            plot_price_discovery_share_diff(
                pair_diff,
                title=f"Long-run price discovery share difference, covid_and_after - pre_covid, {frequency}",
                path=pair_diff_fig,
                label_col="pair",
                x_label="post - pre crypto-side share",
            )
            if pair_prepost_fig.exists():
                entries.append(_entry(root, pair_prepost_fig, "png", f"Pairwise GIS/Hasbrouck pre/post chart for {frequency}."))
            if pair_diff_fig.exists():
                entries.append(_entry(root, pair_diff_fig, "png", f"Pairwise GIS/Hasbrouck difference chart for {frequency}."))

        pre_windows = _read_price_discovery_windows(root, "pre_covid", frequency)
        post_windows = _read_price_discovery_windows(root, "covid_and_after", frequency)
        if pre_windows.empty or post_windows.empty:
            continue

        block_windows = aggregate_price_discovery_blocks(pd.concat([pre_windows, post_windows], ignore_index=True))
        if block_windows.empty:
            continue
        block_windows_path = summary_dir / f"crypto_global_long_run_windows_{frequency}.csv"
        block_windows_path.parent.mkdir(parents=True, exist_ok=True)
        block_windows.to_csv(block_windows_path, index=False)
        entries.append(
            _entry(root, block_windows_path, "csv", f"Window crypto-vs-global GIS/Hasbrouck block averages for {frequency}.")
        )

        block_ci = summarize_ci(
            block_windows,
            ["frequency", "period", "metric", "left_block", "right_block"],
            "left_share",
        )
        if block_ci.empty:
            continue
        block_ci["block_pair"] = block_ci["left_block"].astype(str) + " -> " + block_ci["right_block"].astype(str)
        block_ci_path = summary_dir / f"crypto_global_long_run_ci_{frequency}.csv"
        block_ci.to_csv(block_ci_path, index=False)
        entries.append(_entry(root, block_ci_path, "csv", f"Crypto-vs-global GIS/Hasbrouck block CI for {frequency}."))

        pre_block_ci = block_ci[block_ci["period"].eq("pre_covid")]
        post_block_ci = block_ci[block_ci["period"].eq("covid_and_after")]
        block_diff = compare_ci_summaries(
            pre_block_ci,
            post_block_ci,
            ["frequency", "metric", "left_block", "right_block", "block_pair"],
        )
        block_diff_path = summary_dir / f"crypto_global_long_run_diff_covid_minus_pre_{frequency}.csv"
        if not block_diff.empty:
            block_diff.to_csv(block_diff_path, index=False)
            entries.append(
                _entry(root, block_diff_path, "csv", f"Crypto-vs-global GIS/Hasbrouck block post-minus-pre tests for {frequency}.")
            )

        block_prepost_fig = figure_dir / f"crypto_global_pre_post_{frequency}.png"
        block_diff_fig = figure_dir / f"crypto_global_diff_covid_minus_pre_{frequency}.png"
        block_dynamics_fig = figure_dir / f"crypto_global_dynamics_{frequency}.png"
        plot_price_discovery_share_pre_post(
            block_ci,
            title=f"Crypto long-run information share by global block, pre/post, {frequency}",
            path=block_prepost_fig,
            label_col="block_pair",
            x_label="Crypto block long-run information share",
        )
        plot_price_discovery_share_diff(
            block_diff,
            title=f"Crypto long-run information share difference by global block, {frequency}",
            path=block_diff_fig,
            label_col="block_pair",
            x_label="post - pre crypto block share",
        )
        plot_price_discovery_block_dynamics(
            block_windows,
            title=f"Crypto long-run information share dynamics by global block, {frequency}",
            path=block_dynamics_fig,
            y_label="Crypto block long-run share",
        )
        for path, description in (
            (block_prepost_fig, f"Crypto-vs-global GIS/Hasbrouck pre/post block chart for {frequency}."),
            (block_diff_fig, f"Crypto-vs-global GIS/Hasbrouck block difference chart for {frequency}."),
            (block_dynamics_fig, f"Crypto-vs-global GIS/Hasbrouck dynamics chart for {frequency}."),
        ):
            if path.exists():
                entries.append(_entry(root, path, "png", description))

    return entries


def _block_ci_labels(matrix: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame | None:
    if ci.empty:
        return None
    labels = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    indexed = ci.set_index(["receiver_block", "shock_block"])
    for receiver in labels.index:
        for shock in labels.columns:
            key = (receiver, shock)
            value = float(matrix.loc[receiver, shock])
            if key in indexed.index:
                row = indexed.loc[key]
                labels.loc[receiver, shock] = f"{value:.2f}\n[{float(row['q025']):.2f},{float(row['q975']):.2f}]"
            else:
                labels.loc[receiver, shock] = f"{value:.2f}"
    return labels


def _block_diff_labels(diff: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame | None:
    if stats.empty:
        return None
    labels = pd.DataFrame("", index=diff.index, columns=diff.columns)
    indexed = stats.set_index(["receiver_block", "shock_block"])
    for receiver in labels.index:
        for shock in labels.columns:
            key = (receiver, shock)
            value = float(diff.loc[receiver, shock])
            if key in indexed.index:
                row = indexed.loc[key]
                stars = str(row["stars"]) if not pd.isna(row["stars"]) else ""
                p_value = float(row["p_value"])
                p_label = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
                labels.loc[receiver, shock] = f"{value:+.2f}{stars}\n{p_label}"
            else:
                labels.loc[receiver, shock] = f"{value:+.2f}"
    return labels


def _slice_utc(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]


def _entry(root: Path, path: Path, kind: str, description: str) -> ArtifactEntry:
    try:
        recorded = str(path.relative_to(root))
    except ValueError:
        recorded = str(path)
    return ArtifactEntry(path=recorded, kind=kind, description=description)


def _dedupe_entries(entries: list[ArtifactEntry]) -> list[ArtifactEntry]:
    deduped: dict[str, ArtifactEntry] = {}
    for entry in entries:
        deduped[entry.path] = entry
    return list(deduped.values())


def _build_report(tci_tables: list[pd.DataFrame]) -> str:
    lines = [
        "# Pre/Post, CI, Network Artifacts",
        "",
        "Confidence intervals are empirical time-window intervals, not analytic standard errors.",
        "For each frequency and period the code estimates GFEVD repeatedly on rolling/subsample windows and reports 2.5%, 50%, and 97.5% quantiles.",
        "This is useful for the regime hypothesis because it shows whether connectedness is stable inside a period or moves by windows.",
        "",
        "## Key Files",
        "",
        "- `figures/gfevd/pre_post`: pre/post matrices, log matrices, and post-minus-pre differences.",
        "- `figures/gfevd/pre_post/block_matrix_*`: block total and block adjusted pre/post matrices with 95% window CI and post-minus-pre differences with p-value stars.",
        "- `figures/gfevd/periods`: standalone pre_covid and covid_and_after asset/block heatmaps.",
        "- `figures/gfevd/networks`: directed GFEVD network graphs; edge direction is shock source -> receiver.",
        "- `figures/price_discovery/long_run`: GIS and Hasbrouck proxy pre/post, differences, and crypto-vs-global dynamics.",
        "- `gfevd/periods`: static pre_covid and covid_and_after GFEVD matrices, block tables, and metadata.",
        "- `gfevd/confidence`: window estimates and empirical CI tables for TCI, matrix cells, blocks, and asset connectedness.",
        "- `price_discovery/confidence`: window estimates and empirical CI tables for pairwise GIS and Hasbrouck proxy shares.",
        "- `price_discovery/summary`: long-run price discovery summaries used in the GIS/Hasbrouck figures.",
        "",
        "## TCI Intervals",
        "",
    ]
    if not tci_tables:
        lines.append("No CI tables were produced.")
    else:
        all_tci = pd.concat(tci_tables, ignore_index=True)
        for _, row in all_tci.sort_values(["frequency", "period"]).iterrows():
            lines.append(
                f"- `{row['frequency']}` / `{row['period']}`: "
                f"mean={float(row['mean']):.4f}, "
                f"95% CI=[{float(row['q025']):.4f}, {float(row['q975']):.4f}], "
                f"windows={int(row['n_windows'])}"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
