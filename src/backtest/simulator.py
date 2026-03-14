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
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if listings.empty:
        empty_trades = pd.DataFrame(columns=["listing_id", "brand", "profit", "holding_days", "day"])
        empty_curve = pd.DataFrame(columns=["day", "daily_pnl", "equity", "cash", "inventory_count"])
        return empty_trades, empty_curve, summarize_trades(empty_trades, empty_curve)

    listings = listings.sort_values("timestamp_observed").copy()
    listings["day"] = listings["timestamp_observed"].dt.floor("D")
    inventory: list[InventoryItem] = []
    cash = SETTINGS.starting_cash
    trades: list[dict] = []
    equity_points: list[dict] = []
    seed = SETTINGS.simulator_seed_base if rng_seed is None else rng_seed
    rng = np.random.default_rng(seed)

    for day, day_candidates in listings.groupby("day", sort=True):
