from __future__ import annotations

import pandas as pd

from src.backtest.simulator import run_strategy_backtest
from src.config import SETTINGS


def run_stress_tests(strategies, listings: pd.DataFrame, sale_model, survival_model) -> pd.DataFrame:
    rows: list[dict] = []
    scenarios = [
        ("base", 1.0, 1.0, 1.0),
        ("demand_shock", SETTINGS.demand_shock_scale, 1.0, 1.0),
        ("regime_shift", 1.0, SETTINGS.regime_shift_dispersion_scale, 1.0),
        ("adversarial_noise", 1.0, 1.0, SETTINGS.adversarial_price_scale),
    ]
    for strategy in strategies:
        for scenario_name, hazard_scale, dispersion_scale, price_noise_scale in scenarios:
            _, _, summary = run_strategy_backtest(
                strategy,
                listings,
