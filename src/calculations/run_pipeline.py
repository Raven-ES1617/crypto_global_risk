from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from calculations.artifacts import ArtifactRegistry
from calculations.config import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_ROOT,
    FREQUENCY_CONFIGS,
    PERIOD_SPLITS,
    PRICE_SANITY_RANGES,
    RESEARCH_ASSETS,
    RESEARCH_END,
    RESEARCH_START,
    SECOND_SAMPLE_END,
    SECOND_SAMPLE_START,
    SOURCE_FREQUENCY_BY_SYMBOL,
)
from calculations.gfevd_analysis import block_spillover_table, rolling_gfevd, run_gfevd_for_panel
from calculations.panels import PanelBuildResult, build_price_panel, save_panel_artifacts
from calculations.plots import (
    plot_block_heatmap,
    plot_gfevd_heatmap,
    plot_normalized_prices,
    plot_price_discovery,
    plot_rolling_tci,
)
from calculations.price_discovery import run_pairwise_price_discovery


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    artifact_root = Path(args.artifacts_root)
    registry = ArtifactRegistry(artifact_root)

    frequencies = tuple(args.frequencies)
    panel_results: dict[str, pd.DataFrame] = {}
    gfevd_metadata: list[dict[str, object]] = []
    discovery_tables: list[pd.DataFrame] = []

    for frequency in frequencies:
        config = FREQUENCY_CONFIGS[frequency]
        if frequency == "1s" and not args.full_seconds:
            start, end = args.second_sample_start, args.second_sample_end
            label = "sample"
        else:
            start, end = args.start, args.end
            label = "full"

        panel_name = f"panel_{frequency}" if label == "full" else f"panel_{frequency}_{label}"
        prices_path = registry.path("panels", f"{panel_name}.parquet")
        returns_path = registry.path("returns", f"returns_{frequency}_{label}.parquet")
        stats_path = registry.path("reports", "data_quality", f"panel_stats_{frequency}_{label}.csv")

        print(f"[panel] {frequency}: {start} -> {end}", flush=True)
        if prices_path.exists() and returns_path.exists() and not args.rebuild_panels:
            print(f"[panel] {frequency}: using existing {prices_path}", flush=True)
            panel = PanelBuildResult(
                prices=pd.read_parquet(prices_path),
                returns=pd.read_parquet(returns_path),
                raw_prices=pd.DataFrame(),
                stats=[],
            )
        else:
            panel = build_price_panel(
                RESEARCH_ASSETS,
                data_root=data_root,
                frequency=frequency,
                pandas_rule=config.pandas_rule,
                start=start,
                end=end,
                stale_limit_periods=config.stale_limit_periods,
                price_ranges=PRICE_SANITY_RANGES,
                source_frequency_by_symbol=SOURCE_FREQUENCY_BY_SYMBOL,
            )
            save_panel_artifacts(
                panel,
                frequency=frequency,
                panel_path=prices_path,
                returns_path=returns_path,
                stats_path=stats_path,
            )
        registry.add(prices_path, kind="parquet", description=f"Aligned {frequency} price panel ({label}).")
        registry.add(returns_path, kind="parquet", description=f"Aligned {frequency} log-return panel ({label}).")
        if stats_path.exists():
            registry.add(stats_path, kind="csv", description=f"{frequency} input coverage and resampling stats.")
        panel_results[frequency] = panel.prices

        plot_path = registry.path("figures", "panels", f"normalized_prices_{frequency}_{label}.png")
        plot_normalized_prices(panel.prices, title=f"Normalized prices, {frequency} ({label})", path=plot_path)
        registry.add(plot_path, kind="png", description=f"Normalized price paths by asset block, {frequency}.")

        print(f"[gfevd] {frequency}: static full/sample", flush=True)
        gfevd_dir = registry.path("gfevd", "matrices")
        metadata = run_gfevd_for_panel(
            panel.prices,
            frequency=frequency,
            period=label,
            config=config,
            output_dir=gfevd_dir,
        )
        gfevd_metadata.append(asdict(metadata))
        _register_gfevd_outputs(registry, frequency, label)

        matrix_path = gfevd_dir / f"matrix_{label}_{frequency}.csv"
        block_path = gfevd_dir / f"block_spillovers_{label}_{frequency}.csv"
        matrix = pd.read_csv(matrix_path, index_col=0)
        blocks = pd.read_csv(block_path)
        heatmap_path = registry.path("figures", "gfevd", f"matrix_{label}_{frequency}.png")
        block_fig_path = registry.path("figures", "gfevd", f"blocks_{label}_{frequency}.png")
        block_adjusted_fig_path = registry.path("figures", "gfevd", f"blocks_adjusted_{label}_{frequency}.png")
        plot_gfevd_heatmap(matrix, title=f"GFEVD matrix, {frequency} ({label})", path=heatmap_path)
        plot_block_heatmap(blocks, title=f"Block total spillovers, {frequency} ({label})", path=block_fig_path)
        plot_block_heatmap(
            blocks,
            title=f"Block adjusted spillovers, {frequency} ({label})",
            path=block_adjusted_fig_path,
            value_col="average_pair_share",
        )
        registry.add(heatmap_path, kind="png", description=f"GFEVD heatmap, {frequency}.")
        registry.add(block_fig_path, kind="png", description=f"Block total GFEVD heatmap, {frequency}.")
        registry.add(block_adjusted_fig_path, kind="png", description=f"Block adjusted GFEVD heatmap, {frequency}.")

        if label == "full":
            for period, (period_start, period_end) in PERIOD_SPLITS.items():
                if period == "full":
                    continue
                period_prices = _slice_utc(panel.prices, period_start, period_end)
                if len(period_prices) < 80:
                    continue
                print(f"[gfevd] {frequency}: {period}", flush=True)
                period_metadata = run_gfevd_for_panel(
                    period_prices,
                    frequency=frequency,
                    period=period,
                    config=config,
                    output_dir=registry.path("gfevd", "periods"),
                )
                gfevd_metadata.append(asdict(period_metadata))
                _register_gfevd_outputs(registry, frequency, period, group="periods")

                period_matrix_path = registry.path("gfevd", "periods", f"matrix_{period}_{frequency}.csv")
                period_block_path = registry.path("gfevd", "periods", f"block_spillovers_{period}_{frequency}.csv")
                period_matrix = pd.read_csv(period_matrix_path, index_col=0)
                period_blocks = pd.read_csv(period_block_path)
                period_heatmap_path = registry.path("figures", "gfevd", "periods", f"matrix_{period}_{frequency}.png")
                period_block_fig_path = registry.path("figures", "gfevd", "periods", f"blocks_{period}_{frequency}.png")
                period_block_adjusted_fig_path = registry.path(
                    "figures",
                    "gfevd",
                    "periods",
                    f"blocks_adjusted_{period}_{frequency}.png",
                )
                plot_gfevd_heatmap(
                    period_matrix,
                    title=f"GFEVD matrix, {frequency} ({period})",
                    path=period_heatmap_path,
                )
                plot_block_heatmap(
                    period_blocks,
                    title=f"Block total spillovers, {frequency} ({period})",
                    path=period_block_fig_path,
                )
                plot_block_heatmap(
                    period_blocks,
                    title=f"Block adjusted spillovers, {frequency} ({period})",
                    path=period_block_adjusted_fig_path,
                    value_col="average_pair_share",
                )
                registry.add(period_heatmap_path, kind="png", description=f"GFEVD heatmap for {period}, {frequency}.")
                registry.add(
                    period_block_fig_path,
                    kind="png",
                    description=f"Block total GFEVD heatmap for {period}, {frequency}.",
                )
                registry.add(
                    period_block_adjusted_fig_path,
                    kind="png",
                    description=f"Block adjusted GFEVD heatmap for {period}, {frequency}.",
                )

            rolling = rolling_gfevd(panel.prices, frequency=frequency, config=config, max_windows=args.max_rolling_windows)
            if not rolling.empty:
                rolling_path = registry.path("gfevd", "rolling", f"rolling_tci_{frequency}.csv")
                rolling.to_csv(rolling_path, index=False)
                registry.add(rolling_path, kind="csv", description=f"Rolling total connectedness, {frequency}.")
                rolling_fig = registry.path("figures", "gfevd", f"rolling_tci_{frequency}.png")
                plot_rolling_tci(rolling, title=f"Rolling total connectedness, {frequency}", path=rolling_fig)
                if rolling_fig.exists():
                    registry.add(rolling_fig, kind="png", description=f"Rolling total connectedness plot, {frequency}.")

        print(f"[price-discovery] {frequency}", flush=True)
        discovery = run_pairwise_price_discovery(panel.prices, frequency=frequency, config=config)
        discovery_path = registry.path("price_discovery", f"pairwise_{frequency}_{label}.csv")
        discovery.to_csv(discovery_path, index=False)
        registry.add(discovery_path, kind="csv", description=f"Pairwise GIS and Hasbrouck proxy, {frequency}.")
        discovery_tables.append(discovery)
        discovery_fig = registry.path("figures", "price_discovery", f"hasbrouck_{frequency}_{label}.png")
        plot_price_discovery(discovery, frequency=frequency, path=discovery_fig)
        if discovery_fig.exists():
            registry.add(discovery_fig, kind="png", description=f"Hasbrouck proxy pair chart, {frequency}.")

    if discovery_tables:
        all_discovery = pd.concat(discovery_tables, ignore_index=True)
        all_discovery_path = registry.path("price_discovery", "pairwise_all_frequencies.csv")
        all_discovery.to_csv(all_discovery_path, index=False)
        registry.add(all_discovery_path, kind="csv", description="Pairwise price-discovery metrics across frequencies.")

    metadata_path = registry.path("gfevd", "gfevd_run_metadata.csv")
    pd.DataFrame(gfevd_metadata).to_csv(metadata_path, index=False)
    registry.add(metadata_path, kind="csv", description="GFEVD run settings and connectedness totals.")

    report = build_summary_report(
        frequencies=frequencies,
        gfevd_metadata=gfevd_metadata,
        full_seconds=args.full_seconds,
    )
    registry.write_text("reports", "calculation_summary.md", text=report, description="Human-readable calculation summary.")
    registry.write_manifest(
        metadata={
            "assets": list(RESEARCH_ASSETS),
            "frequencies": list(frequencies),
            "start": args.start,
            "end": args.end,
            "second_sample_start": args.second_sample_start,
            "second_sample_end": args.second_sample_end,
        }
    )
    print(f"[done] artifacts written to {artifact_root}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research panels, risk metrics, and figures.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--start", default=RESEARCH_START)
    parser.add_argument("--end", default=RESEARCH_END)
    parser.add_argument("--second-sample-start", default=SECOND_SAMPLE_START)
    parser.add_argument("--second-sample-end", default=SECOND_SAMPLE_END)
    parser.add_argument("--full-seconds", action="store_true", help="Build full-period 1s panel. This is heavy.")
    parser.add_argument("--rebuild-panels", action="store_true", help="Ignore existing panel parquet files.")
    parser.add_argument("--max-rolling-windows", type=int, default=None)
    parser.add_argument(
        "--frequencies",
        nargs="+",
        choices=sorted(FREQUENCY_CONFIGS),
        default=["1min", "1h", "1d"],
    )
    return parser.parse_args(argv)


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


def _register_gfevd_outputs(
    registry: ArtifactRegistry,
    frequency: str,
    period: str,
    *,
    group: str = "matrices",
) -> None:
    for name, description in (
        (f"matrix_{period}_{frequency}.csv", "Row-normalized GFEVD matrix."),
        (f"raw_matrix_{period}_{frequency}.csv", "Raw GFEVD matrix before row normalization."),
        (f"asset_connectedness_{period}_{frequency}.csv", "Asset-level from/to/net connectedness."),
        (f"block_spillovers_{period}_{frequency}.csv", "Block-level spillover table."),
        (f"block_matrix_{period}_{frequency}.csv", "Block-level spillover matrix."),
        (f"block_matrix_total_{period}_{frequency}.csv", "Block total spillover matrix."),
        (f"block_matrix_adjusted_{period}_{frequency}.csv", "Block size-adjusted spillover matrix."),
        (f"metadata_{period}_{frequency}.json", "GFEVD estimation metadata."),
    ):
        kind = "json" if name.endswith(".json") else "csv"
        registry.add(registry.path("gfevd", group, name), kind=kind, description=f"{description} {period}, {frequency}.")


def build_summary_report(
    *,
    frequencies: tuple[str, ...],
    gfevd_metadata: list[dict[str, object]],
    full_seconds: bool,
) -> str:
    metadata = pd.DataFrame(gfevd_metadata)
    lines = [
        "# Calculation Artifacts",
        "",
        "Panels are built on one asset set: crypto, SPX/NASDAQ100, dollar FX block, gold, and Brent.",
        "",
        "Second-level data is calculated on a sample window by default because the full 1s panel is large. Use `--full-seconds` for the full 1s mode.",
        "",
        "UDXUSD uses sanity filter `50..150`: early HistData quotes before 2018-12-17 are around 24000-27000 and do not look like dollar-index quotes.",
        "",
        f"Frequencies in this run: {', '.join(frequencies)}.",
        f"Full 1s period: {'yes' if full_seconds else 'no'}.",
        "",
        "## GFEVD total connectedness",
        "",
    ]
    if metadata.empty:
        lines.append("GFEVD was not calculated.")
    else:
        for _, row in metadata.sort_values(["frequency", "period"]).iterrows():
            lines.append(
                f"- `{row['frequency']}` / `{row['period']}`: "
                f"TCI={float(row['total_connectedness']):.4f}, "
                f"rows={int(row['rows_available'])}, "
                f"rank={int(row['coint_rank'])}, lag={int(row['lag_order_diff'])}"
            )
    lines.extend(
        [
            "",
            "## How to Read",
            "",
            "- `gfevd/matrices` shows shock transmission: rows receive shocks, columns send shocks.",
            "- `gfevd/rolling` is for regime checks: wave-like TCI means market integration is not constant.",
            "- `price_discovery` shows parity or dominance in pairs: values near 0.5 are closer to parity; strong imbalance points to the price leader.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
