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

