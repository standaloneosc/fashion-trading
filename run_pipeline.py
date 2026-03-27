from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest.metrics import aggregate_rollout_summaries
from src.backtest.simulator import run_strategy_backtest
from src.backtest.stress import run_stress_tests
from src.collect.io import load_market_data
from src.config import PLOTS_DIR, REPORTS_DIR, SETTINGS, ensure_directories
from src.features.build_features import build_feature_store, build_training_frame
from src.models.sale_prob import fit_sale_probability_model
from src.models.survival import empirical_survival_curve, fit_survival_model
from src.plots.make_plots import make_plots
from src.strategies.strategies import STRATEGIES


def write_readme_summary(
    strategy_comparison: pd.DataFrame,
    stress_tests: pd.DataFrame,
    listings: pd.DataFrame,
    sold: pd.DataFrame,
    output_path: Path,
) -> None:
    best = strategy_comparison.sort_values("total_pnl", ascending=False).iloc[0]
    baseline = strategy_comparison[strategy_comparison["strategy"] == "Baseline"].iloc[0]
    profit_uplift = 0.0
    sharpe_multiple = 0.0
    if baseline["avg_profit_trade"] != 0:
        profit_uplift = 100 * (best["avg_profit_trade"] - baseline["avg_profit_trade"]) / abs(baseline["avg_profit_trade"])
    if baseline["sharpe"] != 0:
        sharpe_multiple = best["sharpe"] / baseline["sharpe"]

    demand_shock = stress_tests[stress_tests["scenario"] == "demand_shock"]
    resilient = demand_shock.sort_values("sharpe", ascending=False).iloc[0]

    summary = f"""# README Summary

## Problem Statement

Resale engine: pricing and acquisition using mispricing, liquidity, momentum, and inventory-risk signals.

## Dataset

- Active listings rows: {len(listings)}
- Sold history rows: {len(sold)}
- Multi-venue CSV panels under `data/`.

## Best Strategy

- Best total PnL: {best['strategy']}
- Simulated trades: {int(best['sim_trades'])}
- Avg profit per trade: {best['avg_profit_trade']:.2f}
- Total PnL: {best['total_pnl']:.2f}
- Sharpe: {best['sharpe']:.2f}
- Max drawdown: {best['max_drawdown']:.2f}

## Robustness Highlights

- Most resilient under demand shock: {resilient['strategy']} with Sharpe {resilient['sharpe']:.2f}
- Stress scenarios: demand shock, regime shift, adversarial noise.

## Notes

Vs baseline: ~{profit_uplift:.1f}% higher average profit per trade; Sharpe ratio ~{sharpe_multiple:.2f}x baseline in this run.
"""
    output_path.write_text(summary)


def _run_strategy_rollouts(strategy, featured_listings, sale_model, survival_model):
    trades_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for path in range(SETTINGS.backtest_rollouts):
        trades, curve, _ = run_strategy_backtest(
            strategy,
            featured_listings,
            sale_model,
            survival_model,
            rng_seed=SETTINGS.simulator_seed_base + path * 7919,
        )
        trades_frames.append(trades)
        curve_frames.append(curve)
    merged_summary = aggregate_rollout_summaries(trades_frames, curve_frames)
    merged_summary["strategy"] = strategy.name
    merged_trades = pd.concat(trades_frames, ignore_index=True)
    representative_curve = curve_frames[0]
    return merged_trades, representative_curve, merged_summary


def main() -> None:
    ensure_directories()
    listings, sold = load_market_data()
    featured_listings, sold_features = build_feature_store(listings, sold)
    training = build_training_frame(featured_listings, sold_features)

    sale_artifacts = fit_sale_probability_model(training, REPORTS_DIR / "model_metrics.json")
    survival_artifacts = fit_survival_model(training, REPORTS_DIR / "survival_metrics.json")
    survival_curve = empirical_survival_curve(training)

    strategy_summaries: list[dict] = []
    all_equity_curves: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        trades, curve, summary = _run_strategy_rollouts(
            strategy,
            featured_listings,
            sale_artifacts.model,
            survival_artifacts.model,
        )
        strategy_summaries.append(summary)
        all_equity_curves.append(curve)
        all_trades.append(trades)

    strategy_comparison = pd.DataFrame(strategy_summaries).sort_values("total_pnl", ascending=False)
    equity_curves = pd.concat(all_equity_curves, ignore_index=True) if all_equity_curves else pd.DataFrame()
