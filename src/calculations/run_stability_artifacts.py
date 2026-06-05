from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from calculations.artifacts import ArtifactEntry
from calculations.config import DEFAULT_ARTIFACT_ROOT
from calculations.plots import (
    plot_block_flow_dynamics_frequency_grid,
    plot_block_net_frequency_grid,
    plot_normalized_prices,
    plot_pre_post_frequency_grid,
    plot_price_discovery_frequency_grid,
)
from calculations.stability import (
    BLOCK_VARIANTS,
    FULL_FREQUENCIES,
    PERIODS,
    add_diff_stability,
    add_interval_stability,
    add_price_discovery_dominance,
    block_net_flows,
    crypto_global_flows,
    frequency_consistency,
    metric_agreement,
    plot_block_net_flows,
    plot_block_net_dynamics,
    plot_crypto_global_gfevd,
    plot_crypto_global_gfevd_dynamics,
    plot_frequency_heatmap,
    plot_hypothesis_dashboard,
    window_block_net_flows,
    window_crypto_global_flows,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.artifacts_root)
    entries: list[ArtifactEntry] = []
    stability_dir = root / "stability"
    figure_dir = root / "figures" / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    block_net = build_block_net_flows(root)
    entries.append(_write_csv(root, stability_dir / "block_net_flows.csv", block_net, "Block TO/FROM/NET spillovers."))
    for variant in BLOCK_VARIANTS:
        path = figure_dir / f"block_net_flows_{variant}.png"
        plot_block_net_flows(block_net, variant=variant, path=path)
        if path.exists():
            entries.append(_entry(root, path, "png", f"Block net spillover pre/post chart, {variant}."))

    crypto_global = build_crypto_global_flows(root)
    entries.append(
        _write_csv(root, stability_dir / "crypto_global_gfevd_flows.csv", crypto_global, "Crypto-vs-global GFEVD flows.")
    )
    for variant in BLOCK_VARIANTS:
        path = figure_dir / f"crypto_global_gfevd_{variant}.png"
        plot_crypto_global_gfevd(crypto_global, variant=variant, path=path)
        if path.exists():
            entries.append(_entry(root, path, "png", f"Crypto-vs-global GFEVD pre/post chart, {variant}."))

    window_block_net, window_crypto_global = build_window_flow_stability(root)
    entries.append(
        _write_csv(root, stability_dir / "window_block_net_flows.csv", window_block_net, "Rolling block TO/FROM/NET spillovers.")
    )
    entries.append(
        _write_csv(
            root,
            stability_dir / "window_crypto_global_gfevd_flows.csv",
            window_crypto_global,
            "Rolling crypto-vs-global GFEVD flows.",
        )
    )
    for variant in BLOCK_VARIANTS:
        path = figure_dir / f"window_block_net_flows_{variant}.png"
        plot_block_net_dynamics(window_block_net, variant=variant, path=path)
        if path.exists():
            entries.append(_entry(root, path, "png", f"Rolling block net spillover chart, {variant}."))
        path = figure_dir / f"window_crypto_global_gfevd_{variant}.png"
        plot_crypto_global_gfevd_dynamics(window_crypto_global, variant=variant, path=path)
        if path.exists():
            entries.append(_entry(root, path, "png", f"Rolling crypto-vs-global GFEVD chart, {variant}."))

    block_ci, block_diff = build_block_stability(root)
    entries.append(_write_csv(root, stability_dir / "block_ci_stability.csv", block_ci, "Block GFEVD interval stability checks."))
    entries.append(
        _write_csv(root, stability_dir / "block_diff_stability.csv", block_diff, "Block GFEVD post-minus-pre stability checks.")
    )

    tci_stability, tci_diff = build_tci_stability(root)
    entries.append(_write_csv(root, stability_dir / "tci_stability.csv", tci_stability, "TCI interval stability checks."))
    entries.append(_write_csv(root, stability_dir / "tci_diff_stability.csv", tci_diff, "TCI post-minus-pre stability checks."))

    pair_ci, pair_diff, block_pd_ci, block_pd_diff, agreements = build_price_discovery_stability(root)
    entries.append(
        _write_csv(root, stability_dir / "price_discovery_pair_stability.csv", pair_ci, "Pairwise GIS/Hasbrouck dominance checks.")
    )
    entries.append(
        _write_csv(
            root,
            stability_dir / "price_discovery_pair_diff_stability.csv",
            pair_diff,
            "Pairwise GIS/Hasbrouck post-minus-pre stability checks.",
        )
    )
    entries.append(
        _write_csv(
            root,
            stability_dir / "price_discovery_crypto_global_stability.csv",
            block_pd_ci,
            "Crypto-vs-global GIS/Hasbrouck dominance checks.",
        )
    )
    entries.append(
        _write_csv(
            root,
            stability_dir / "price_discovery_crypto_global_diff_stability.csv",
            block_pd_diff,
            "Crypto-vs-global GIS/Hasbrouck post-minus-pre stability checks.",
        )
    )
    entries.append(
        _write_csv(
            root,
            stability_dir / "price_discovery_metric_agreement.csv",
            agreements,
            "GIS vs Hasbrouck agreement checks.",
        )
    )

    frequency_effects = build_frequency_effects(tci_diff, crypto_global, block_pd_diff)
    entries.append(_write_csv(root, stability_dir / "frequency_effects.csv", frequency_effects, "Cross-frequency effect table."))
    frequency_consistency_checks = build_frequency_consistency(block_diff, pair_diff, block_pd_diff, frequency_effects)
    entries.append(
        _write_csv(
            root,
            stability_dir / "frequency_consistency.csv",
            frequency_consistency_checks,
            "Cross-frequency sign consistency checks.",
        )
    )
    frequency_fig = figure_dir / "frequency_effects_heatmap.png"
    plot_frequency_heatmap(frequency_effects, title="Post-minus-pre effects by frequency", path=frequency_fig)
    if frequency_fig.exists():
        entries.append(_entry(root, frequency_fig, "png", "Frequency comparison heatmap for key effects."))

    hypothesis_checks = build_hypothesis_checks(
        tci_diff=tci_diff,
        block_ci=block_ci,
        block_diff=block_diff,
        crypto_global=crypto_global,
        pair_ci=pair_ci,
        pair_diff=pair_diff,
        block_pd_ci=block_pd_ci,
        agreements=agreements,
        frequency_checks=frequency_consistency_checks,
    )
    entries.append(
        _write_csv(root, stability_dir / "hypothesis_stability_checks.csv", hypothesis_checks, "Hypothesis-level stability checks.")
    )
    dashboard = figure_dir / "hypothesis_stability_dashboard.png"
    plot_hypothesis_dashboard(hypothesis_checks, path=dashboard)
    if dashboard.exists():
        entries.append(_entry(root, dashboard, "png", "Hypothesis stability dashboard."))

    entries.extend(_write_frl_article_figures(root, window_block_net))

    report_path = root / "reports" / "stability_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_stability_report(hypothesis_checks, tci_diff, frequency_consistency_checks), encoding="utf-8")
    entries.append(_entry(root, report_path, "markdown", "Stability checks summary."))

    manifest_path = root / "reports" / "stability_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": [asdict(entry) for entry in _dedupe_entries(entries)]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] stability artifacts written to {root}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stability checks and robustness artifacts.")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser.parse_args(argv)


def build_block_net_flows(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frequency in FULL_FREQUENCIES:
        for period in PERIODS:
            for variant, prefix in BLOCK_VARIANTS.items():
                path = root / "gfevd" / "periods" / f"{prefix}_{period}_{frequency}.csv"
                if not path.exists():
                    continue
                matrix = pd.read_csv(path, index_col=0)
                table = block_net_flows(matrix)
                table.insert(0, "variant", variant)
                table.insert(0, "period", period)
                table.insert(0, "frequency", frequency)
                rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_crypto_global_flows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frequency in FULL_FREQUENCIES:
        for period in PERIODS:
            for variant, prefix in BLOCK_VARIANTS.items():
                path = root / "gfevd" / "periods" / f"{prefix}_{period}_{frequency}.csv"
                if not path.exists():
                    continue
                matrix = pd.read_csv(path, index_col=0)
                rows.append(
                    {
                        "frequency": frequency,
                        "period": period,
                        "variant": variant,
                        **crypto_global_flows(matrix),
                    }
                )
    return pd.DataFrame(rows)


def build_window_flow_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows: list[pd.DataFrame] = []
    crypto_rows: list[pd.DataFrame] = []
    for frequency in FULL_FREQUENCIES:
        path = root / "gfevd" / "window_gifs" / f"block_flow_windows_{frequency}.csv"
        if not path.exists():
            continue
        windows = pd.read_csv(path)
        for variant, value_col in (("total", "average_receiver_share"), ("adjusted", "average_pair_share")):
            block_rows.append(window_block_net_flows(windows, value_col=value_col, variant=variant))
            crypto_rows.append(window_crypto_global_flows(windows, value_col=value_col, variant=variant))
    block = pd.concat([frame for frame in block_rows if not frame.empty], ignore_index=True) if block_rows else pd.DataFrame()
    crypto = pd.concat([frame for frame in crypto_rows if not frame.empty], ignore_index=True) if crypto_rows else pd.DataFrame()
    return block, crypto


def build_block_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_rows: list[pd.DataFrame] = []
    diff_rows: list[pd.DataFrame] = []
    for frequency in FULL_FREQUENCIES:
        for period in PERIODS:
            for variant, prefix in (("total", "block_ci"), ("adjusted", "block_adjusted_ci")):
                path = root / "gfevd" / "confidence" / f"{prefix}_{period}_{frequency}.csv"
                if not path.exists():
                    continue
                frame = pd.read_csv(path)
                frame.insert(0, "variant", variant)
                frame.insert(0, "period", period)
                frame.insert(0, "frequency", frequency)
                max_width = 0.15 if variant == "total" else 0.04
                ci_rows.append(add_interval_stability(frame, max_width=max_width, max_relative_width=1.0))
        for variant, prefix in (("total", "block_diff_ci"), ("adjusted", "block_adjusted_diff_ci")):
            path = root / "gfevd" / "confidence" / f"{prefix}_covid_minus_pre_{frequency}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "variant", variant)
            frame.insert(0, "frequency", frequency)
            diff_rows.append(add_diff_stability(frame))
    ci = pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame()
    diff = pd.concat(diff_rows, ignore_index=True) if diff_rows else pd.DataFrame()
    return ci, diff


def build_tci_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "gfevd" / "confidence" / "tci_ci_all.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    tci = add_interval_stability(pd.read_csv(path), max_width=0.12, max_relative_width=0.35)
    rows: list[dict[str, object]] = []
    for frequency, group in tci.groupby("frequency", dropna=False):
        pre = group[group["period"].eq("pre_covid")]
        post = group[group["period"].eq("covid_and_after")]
        if pre.empty or post.empty:
            continue
        rows.append(_summary_diff_payload(str(frequency), pre.iloc[0], post.iloc[0]))
    return tci, add_diff_stability(pd.DataFrame(rows)) if rows else pd.DataFrame()


def build_price_discovery_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_ci_rows: list[pd.DataFrame] = []
    pair_diff_rows: list[pd.DataFrame] = []
    block_ci_rows: list[pd.DataFrame] = []
    block_diff_rows: list[pd.DataFrame] = []
    for frequency in FULL_FREQUENCIES:
        pair_ci_path = root / "price_discovery" / "summary" / f"long_run_pair_ci_{frequency}.csv"
        pair_diff_path = root / "price_discovery" / "summary" / f"long_run_pair_diff_covid_minus_pre_{frequency}.csv"
        block_ci_path = root / "price_discovery" / "summary" / f"crypto_global_long_run_ci_{frequency}.csv"
        block_diff_path = root / "price_discovery" / "summary" / f"crypto_global_long_run_diff_covid_minus_pre_{frequency}.csv"
        if pair_ci_path.exists():
            pair_ci_rows.append(add_price_discovery_dominance(pd.read_csv(pair_ci_path)))
        if pair_diff_path.exists():
            pair_diff_rows.append(add_diff_stability(pd.read_csv(pair_diff_path)))
        if block_ci_path.exists():
            block_ci_rows.append(add_price_discovery_dominance(pd.read_csv(block_ci_path)))
        if block_diff_path.exists():
            block_diff_rows.append(add_diff_stability(pd.read_csv(block_diff_path)))
    pair_ci = pd.concat(pair_ci_rows, ignore_index=True) if pair_ci_rows else pd.DataFrame()
    pair_diff = pd.concat(pair_diff_rows, ignore_index=True) if pair_diff_rows else pd.DataFrame()
    block_ci = pd.concat(block_ci_rows, ignore_index=True) if block_ci_rows else pd.DataFrame()
    block_diff = pd.concat(block_diff_rows, ignore_index=True) if block_diff_rows else pd.DataFrame()
    pair_agreement = metric_agreement(pair_ci, group_cols=["frequency", "period", "pair"], value_col="mean")
    block_agreement = metric_agreement(block_ci, group_cols=["frequency", "period", "block_pair"], value_col="mean")
    if not pair_agreement.empty:
        pair_agreement.insert(0, "level", "pair")
    if not block_agreement.empty:
        block_agreement.insert(0, "level", "block")
    agreements = pd.concat([pair_agreement, block_agreement], ignore_index=True) if not pair_agreement.empty or not block_agreement.empty else pd.DataFrame()
    return pair_ci, pair_diff, block_ci, block_diff, agreements


def build_frequency_effects(tci_diff: pd.DataFrame, crypto_global: pd.DataFrame, block_pd_diff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not tci_diff.empty:
        for _, row in tci_diff.iterrows():
            rows.append({"effect": "TCI post-pre", "frequency": row["frequency"], "value": float(row["diff_mean"])})

    if not crypto_global.empty:
        for variant, group in crypto_global.groupby("variant", dropna=False):
            for frequency, freq_group in group.groupby("frequency", dropna=False):
                pre = freq_group[freq_group["period"].eq("pre_covid")]
                post = freq_group[freq_group["period"].eq("covid_and_after")]
                if pre.empty or post.empty:
                    continue
                for measure, label in (
                    ("gross_crypto_global", "GFEVD crypto-global gross"),
                    ("net_crypto_to_global", "GFEVD net crypto->global"),
                    ("global_to_crypto", "GFEVD global->crypto"),
                    ("crypto_to_global", "GFEVD crypto->global"),
                ):
                    rows.append(
                        {
                            "effect": f"{label} ({variant})",
                            "frequency": frequency,
                            "value": float(post.iloc[0][measure]) - float(pre.iloc[0][measure]),
                        }
                    )

    if not block_pd_diff.empty:
        for (frequency, metric), group in block_pd_diff.groupby(["frequency", "metric"], dropna=False):
            rows.append(
                {
                    "effect": f"Long-run crypto share ({metric})",
                    "frequency": frequency,
                    "value": float(group["diff_mean"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_frequency_consistency(
    block_diff: pd.DataFrame,
    pair_diff: pd.DataFrame,
    block_pd_diff: pd.DataFrame,
    frequency_effects: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        frequency_consistency(
            block_diff,
            group_cols=["variant", "receiver_block", "shock_block"],
            value_col="diff_mean",
            label="block_gfevd_diff",
        ),
        frequency_consistency(
            pair_diff,
            group_cols=["pair", "metric"],
            value_col="diff_mean",
            label="pair_price_discovery_diff",
        ),
        frequency_consistency(
            block_pd_diff,
            group_cols=["metric", "block_pair"],
            value_col="diff_mean",
            label="block_price_discovery_diff",
        ),
        frequency_consistency(
            frequency_effects,
            group_cols=["effect"],
            value_col="value",
            label="key_effect_diff",
        ),
    ]
    checks = [frame for frame in checks if not frame.empty]
    return pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()


def build_hypothesis_checks(
    *,
    tci_diff: pd.DataFrame,
    block_ci: pd.DataFrame,
    block_diff: pd.DataFrame,
    crypto_global: pd.DataFrame,
    pair_ci: pd.DataFrame,
    pair_diff: pd.DataFrame,
    block_pd_ci: pd.DataFrame,
    agreements: pd.DataFrame,
    frequency_checks: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    tci_up = int((tci_diff.get("diff_mean", pd.Series(dtype=float)) > 0).sum()) if not tci_diff.empty else 0
    tci_robust = int(tci_diff.get("robust_change", pd.Series(dtype=bool)).sum()) if not tci_diff.empty else 0
    rows.append(
        {
            "hypothesis": "После COVID интеграция глубже",
            "evidence": f"TCI вырос на {tci_up}/{len(tci_diff)} частотах; robust TCI diff: {tci_robust}/{len(tci_diff)}.",
            "status": "partial_support" if tci_up else "no_support",
            "stability": "stable" if tci_robust else "mixed",
        }
    )

    crypto_global_diff = _crypto_global_gross_diff(crypto_global)
    gross_up = int((crypto_global_diff["diff_gross"] > 0).sum()) if not crypto_global_diff.empty else 0
    rows.append(
        {
            "hypothesis": "Crypto-global spillovers выросли",
            "evidence": f"Gross crypto-global GFEVD flow вырос в {gross_up}/{len(crypto_global_diff)} variant-frequency ячейках.",
            "status": "support" if gross_up >= max(1, len(crypto_global_diff) // 2) else "mixed",
            "stability": "checked_by_total_and_adjusted",
        }
    )

    volatile_blocks = int(block_ci["stability_flag"].eq("volatile").sum()) if not block_ci.empty else 0
    stable_blocks = int(block_ci["stability_flag"].eq("stable").sum()) if not block_ci.empty else 0
    rows.append(
        {
            "hypothesis": "Интеграция режимная",
            "evidence": f"Block CI cells: stable={stable_blocks}, volatile={volatile_blocks}; window GIFs/dynamics доступны.",
            "status": "support" if volatile_blocks else "mixed",
            "stability": "window_checked",
        }
    )

    robust_block_changes = int(block_diff["robust_change"].sum()) if not block_diff.empty else 0
    rows.append(
        {
            "hypothesis": "Направление шоков важно",
            "evidence": f"Robust post/pre block-direction changes: {robust_block_changes}/{len(block_diff)}.",
            "status": "support" if robust_block_changes else "mixed",
            "stability": "pvalue_and_ci_checked",
        }
    )

    stable_pd = int(pair_ci["stable_dominance"].sum()) if not pair_ci.empty else 0
    robust_pd_diff = int(pair_diff["robust_change"].sum()) if not pair_diff.empty else 0
    rows.append(
        {
            "hypothesis": "Long-run price discovery дополняет GFEVD",
            "evidence": f"Stable pair dominance cells={stable_pd}/{len(pair_ci)}; robust post/pre pair shifts={robust_pd_diff}/{len(pair_diff)}.",
            "status": "support" if stable_pd else "mixed",
            "stability": "gis_hasbrouck_ci_checked",
        }
    )

    same_metric = int(agreements["same_side_of_parity"].sum()) if not agreements.empty else 0
    rows.append(
        {
            "hypothesis": "GIS и Hasbrouck согласованы",
            "evidence": f"Same side of parity in {same_metric}/{len(agreements)} pair/block cells.",
            "status": "support" if same_metric >= max(1, int(0.75 * len(agreements))) else "mixed",
            "stability": "metric_agreement_checked",
        }
    )

    consistent = int(frequency_checks["consistent_nonzero_sign"].sum()) if not frequency_checks.empty else 0
    rows.append(
        {
            "hypothesis": "Частотная картина согласована",
            "evidence": f"Consistent nonzero sign across frequencies in {consistent}/{len(frequency_checks)} checks.",
            "status": "partial_support" if consistent else "mixed",
            "stability": "frequency_sign_checked",
        }
    )

    block_pd_stable = int(block_pd_ci["stable_dominance"].sum()) if not block_pd_ci.empty else 0
    rows.append(
        {
            "hypothesis": "Crypto-vs-global long-run leadership виден",
            "evidence": f"Stable crypto/global long-run dominance cells={block_pd_stable}/{len(block_pd_ci)}.",
            "status": "support" if block_pd_stable else "mixed",
            "stability": "block_price_discovery_checked",
        }
    )

    return pd.DataFrame(rows)


def _write_frl_article_figures(root: Path, window_block_net: pd.DataFrame) -> list[ArtifactEntry]:
    figure_dir = root / "figures" / "frl"
    figure_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ArtifactEntry] = []

    panel_path = root / "panels" / "panel_1d.parquet"
    if panel_path.exists():
        fig_path = figure_dir / "fig1_market_panel.png"
        plot_normalized_prices(
            pd.read_parquet(panel_path),
            title="Normalized prices by asset block, daily panel",
            path=fig_path,
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 1: 2x2 normalized price panel."))

    block_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    block_annotations: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] = {}
    asset_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for frequency in ("1d", "1h", "1min"):
        pre_block = root / "gfevd" / "periods" / f"block_matrix_adjusted_pre_covid_{frequency}.csv"
        post_block = root / "gfevd" / "periods" / f"block_matrix_adjusted_covid_and_after_{frequency}.csv"
        pre_block_ci = root / "gfevd" / "confidence" / f"block_adjusted_ci_pre_covid_{frequency}.csv"
        post_block_ci = root / "gfevd" / "confidence" / f"block_adjusted_ci_covid_and_after_{frequency}.csv"
        pre_asset = root / "gfevd" / "periods" / f"matrix_pre_covid_{frequency}.csv"
        post_asset = root / "gfevd" / "periods" / f"matrix_covid_and_after_{frequency}.csv"
        if pre_block.exists() and post_block.exists():
            pre_template = pd.read_csv(pre_block, index_col=0)
            post_template = pd.read_csv(post_block, index_col=0)
            pre_ci = pd.read_csv(pre_block_ci) if pre_block_ci.exists() else pd.DataFrame()
            post_ci = pd.read_csv(post_block_ci) if post_block_ci.exists() else pd.DataFrame()
            pre_matrix = _block_ci_mean_matrix(pre_ci, pre_template)
            post_matrix = _block_ci_mean_matrix(post_ci, post_template)
            block_matrices[frequency] = (pre_matrix, post_matrix)
            pre_labels = _block_ci_labels(pre_matrix, pre_ci)
            post_labels = _block_ci_labels(post_matrix, post_ci)
            block_annotations[frequency] = (pre_labels, post_labels)
        if pre_asset.exists() and post_asset.exists():
            asset_matrices[frequency] = (pd.read_csv(pre_asset, index_col=0), pd.read_csv(post_asset, index_col=0))

    if block_matrices:
        fig_path = figure_dir / "fig2_block_matrix_adjusted_pre_post.png"
        plot_pre_post_frequency_grid(
            block_matrices,
            title="Block-size-adjusted GFEVD matrices by frequency",
            path=fig_path,
            annotations_by_frequency=block_annotations,
            tick_fontsize=8,
            vmin=0.0,
            vmax=0.4,
            colorbar_label="GFEVD share (0 to 0.4)",
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 2: block pre/post GFEVD matrices by frequency."))

    if asset_matrices:
        fig_path = figure_dir / "fig3_asset_matrix_pre_post.png"
        plot_pre_post_frequency_grid(
            asset_matrices,
            title="Asset-level GFEVD matrices by frequency, log scale",
            path=fig_path,
            log_scale=True,
            tick_fontsize=5,
            vmin=-6.0,
            vmax=0.0,
            colorbar_label="log10 share (0=1; -2=0.01; -6=1e-6)",
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 3: asset-level pre/post GFEVD matrices by frequency."))

    block_windows: dict[str, pd.DataFrame] = {}
    for frequency in ("1d", "1h", "1min"):
        path = root / "gfevd" / "window_gifs" / f"block_flow_windows_{frequency}.csv"
        if path.exists():
            block_windows[frequency] = pd.read_csv(path)
    if block_windows:
        fig_path = figure_dir / "fig4_block_flow_dynamics.png"
        plot_block_flow_dynamics_frequency_grid(
            block_windows,
            title="Rolling adjusted block GFEVD flows by frequency",
            path=fig_path,
            value_col="average_pair_share",
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 4: rolling adjusted block flows by frequency."))

    if not window_block_net.empty:
        fig_path = figure_dir / "fig5_frequency_net_flows.png"
        plot_block_net_frequency_grid(
            window_block_net,
            variant="adjusted",
            title="Rolling adjusted block net spillovers by frequency",
            path=fig_path,
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 5: rolling adjusted net spillovers by frequency."))

    long_run_windows: dict[str, pd.DataFrame] = {}
    for frequency in ("1d", "1h", "1min"):
        path = root / "price_discovery" / "summary" / f"crypto_global_long_run_windows_{frequency}.csv"
        if path.exists():
            long_run_windows[frequency] = pd.read_csv(path)
    if long_run_windows:
        fig_path = figure_dir / "fig6_long_run_common_factor.png"
        plot_price_discovery_frequency_grid(
            long_run_windows,
            title="Crypto-side long-run common-factor shares by frequency",
            path=fig_path,
        )
        entries.append(_entry(root, fig_path, "png", "FRL Figure 6: long-run common-factor shares by frequency."))

    readme_path = figure_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# FRL Figures",
                "",
                "This folder is the canonical figure set used by the article template.",
                "The files are generated by `src/calculations/run_stability_artifacts.py` from the main project artifacts.",
                "",
                "- `fig1_market_panel.png`: daily normalized prices in a compact 2x2 block panel.",
                "- `fig2_block_matrix_adjusted_pre_post.png`: rolling-mean adjusted block GFEVD pre/post matrices for day, hour, and minute; color scale is fixed at 0..0.4 and labels include approximate 95% mean intervals.",
                "- `fig3_asset_matrix_pre_post.png`: non-aggregated asset-level GFEVD pre/post matrices for day, hour, and minute; log10 color scale is fixed at -6..0.",
                "- `fig4_block_flow_dynamics.png`: rolling adjusted block GFEVD flows by receiver block and frequency.",
                "- `fig5_frequency_net_flows.png`: rolling adjusted block net spillovers by block and frequency.",
                "- `fig6_long_run_common_factor.png`: GIS and Hasbrouck-style long-run shares by counterpart block and frequency; Hasbrouck lines include lower/upper ordering bands when available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    entries.append(_entry(root, readme_path, "markdown", "FRL figure set README."))
    return entries


def _block_ci_mean_matrix(ci: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    if ci.empty or not {"receiver_block", "shock_block", "mean"}.issubset(ci.columns):
        return template
    matrix = pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)
    indexed = ci.set_index(["receiver_block", "shock_block"])
    for receiver in matrix.index:
        for shock in matrix.columns:
            key = (receiver, shock)
            if key not in indexed.index:
                matrix.loc[receiver, shock] = float(template.loc[receiver, shock])
                continue
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            matrix.loc[receiver, shock] = float(row["mean"])
    return matrix


def _block_ci_labels(matrix: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame | None:
    if ci.empty:
        return None
    labels = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    indexed = ci.set_index(["receiver_block", "shock_block"])
    for receiver in labels.index:
        for shock in labels.columns:
            value = float(matrix.loc[receiver, shock])
            key = (receiver, shock)
            if key not in indexed.index:
                labels.loc[receiver, shock] = f"{value:.2f}"
                continue
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            low, high = _mean_interval(row)
            labels.loc[receiver, shock] = f"{value:.2f}\n[{low:.2f},{high:.2f}]"
    return labels


def _mean_interval(row: pd.Series) -> tuple[float, float]:
    mean = float(row["mean"])
    n_windows = max(1.0, float(row.get("n_windows", 1.0)))
    std = float(row.get("std", 0.0))
    half_width = 1.96 * std / sqrt(n_windows)
    return mean - half_width, mean + half_width


def build_stability_report(hypothesis_checks: pd.DataFrame, tci_diff: pd.DataFrame, frequency_checks: pd.DataFrame) -> str:
    lines = [
        "# Проверки устойчивости",
        "",
        "Этот отчет собирается из уже готовых GFEVD, GIS и Hasbrouck артефактов.",
        "",
        "## Какие проверки используются",
        "",
        "- ширина CI и относительная ширина CI по оконным оценкам;",
        "- минимальное число rolling-окон;",
        "- p-value для post-minus-pre и проверка, исключает ли 95% diff-интервал ноль;",
        "- совпадает ли знак эффекта на `1min`, `1h`, `1d`;",
        "- согласны ли GIS и Hasbrouck относительно линии паритета `0.5`.",
        "",
        "## Сводка по гипотезам",
        "",
    ]
    for _, row in hypothesis_checks.iterrows():
        lines.append(f"- **{row['hypothesis']}**: `{row['status']}`, `{row['stability']}`. {row['evidence']}")
    lines.extend(["", "## TCI post-minus-pre", ""])
    if tci_diff.empty:
        lines.append("Нет таблицы TCI diff.")
    else:
        for _, row in tci_diff.sort_values("frequency").iterrows():
            lines.append(
                f"- `{row['frequency']}`: diff={float(row['diff_mean']):+.4f}, "
                f"p={float(row['p_value']):.4f}, robust={bool(row['robust_change'])}"
            )
    lines.extend(["", "## Согласованность между частотами", ""])
    if frequency_checks.empty:
        lines.append("Нет таблицы frequency consistency.")
    else:
        total = len(frequency_checks)
        consistent = int(frequency_checks["consistent_nonzero_sign"].sum())
        lines.append(f"Одинаковый ненулевой знак на частотах найден в {consistent}/{total} проверках.")
    return "\n".join(lines) + "\n"


def _summary_diff_payload(frequency: str, pre: pd.Series, post: pd.Series) -> dict[str, object]:
    pre_mean = float(pre["mean"])
    post_mean = float(post["mean"])
    pre_std = float(pre["std"])
    post_std = float(post["std"])
    pre_n = max(1.0, float(pre["n_windows"]))
    post_n = max(1.0, float(post["n_windows"]))
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
    return {
        "frequency": frequency,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "diff_mean": diff,
        "std_error": se,
        "z_value": z_value,
        "p_value": p_value,
        "stars": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "",
        "diff_ci_low": ci_low,
        "diff_ci_high": ci_high,
        "pre_windows": int(pre_n),
        "post_windows": int(post_n),
    }


def _crypto_global_gross_diff(crypto_global: pd.DataFrame) -> pd.DataFrame:
    if crypto_global.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (frequency, variant), group in crypto_global.groupby(["frequency", "variant"], dropna=False):
        pre = group[group["period"].eq("pre_covid")]
        post = group[group["period"].eq("covid_and_after")]
        if pre.empty or post.empty:
            continue
        rows.append(
            {
                "frequency": frequency,
                "variant": variant,
                "pre_gross": float(pre.iloc[0]["gross_crypto_global"]),
                "post_gross": float(post.iloc[0]["gross_crypto_global"]),
                "diff_gross": float(post.iloc[0]["gross_crypto_global"]) - float(pre.iloc[0]["gross_crypto_global"]),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(root: Path, path: Path, frame: pd.DataFrame, description: str) -> ArtifactEntry:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return _entry(root, path, "csv", description)


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


if __name__ == "__main__":
    raise SystemExit(main())
