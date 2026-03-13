from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.backtest.metrics import summarize_trades
from src.config import SETTINGS
from src.models.sale_prob import predict_sale_probability
from src.models.survival import predict_daily_hazard
from src.strategies.base import InventoryItem


def run_strategy_backtest(
    strategy,
    listings: pd.DataFrame,
    sale_model,
    survival_model,
    hazard_scale: float = 1.0,
    dispersion_scale: float = 1.0,
    price_noise_scale: float = 1.0,
