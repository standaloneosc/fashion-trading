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
        day_candidates = day_candidates.copy()
        day_candidates["dispersion_iqr_30d"] = day_candidates["dispersion_iqr_30d"] * dispersion_scale

        state = {
            "cash": cash,
            "inventory": inventory,
            "sale_model": sale_model,
            "survival_model": survival_model,
        }
        proposed = strategy.decide_buys(day, day_candidates, state)
        per_brand_counts = pd.Series([item.brand for item in inventory]).value_counts().to_dict()
        purchases: list[InventoryItem] = []
        for item in proposed:
            if len(inventory) + len(purchases) >= SETTINGS.max_inventory:
                continue
            if per_brand_counts.get(item.brand, 0) >= SETTINGS.max_brand_inventory:
                continue
            total_cost = item.cost + item.features.get("shipping_price", SETTINGS.default_shipping_cost)
            if total_cost <= cash:
                cash -= total_cost
                purchases.append(item)
                per_brand_counts[item.brand] = per_brand_counts.get(item.brand, 0) + 1
        inventory.extend(purchases)

        strategy.set_prices(day, inventory, state)
        daily_pnl = 0.0
        if inventory:
            inventory_frame = pd.DataFrame([item.features for item in inventory])
            sale_probs = predict_sale_probability(sale_model, inventory_frame)
            hazards = predict_daily_hazard(survival_model, inventory_frame)
            sell_probs = np.clip(
                (0.5 * sale_probs + 0.5 * hazards) * hazard_scale * SETTINGS.simulator_demand_multiplier,
                0.02,
                0.92,
            )

            remaining: list[InventoryItem] = []
            for item, sell_prob in zip(inventory, sell_probs):
