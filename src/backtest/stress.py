from __future__ import annotations

import pandas as pd

from src.backtest.simulator import run_strategy_backtest
from src.config import SETTINGS


def run_stress_tests(strategies, listings: pd.DataFrame, sale_model, survival_model) -> pd.DataFrame:
