from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.models.sale_prob import predict_sale_probability
from src.models.survival import predict_daily_hazard
from src.strategies.base import InventoryItem


@dataclass(slots=True)
class StrategyContext:
    sale_model: object
    survival_model: object


class BaseStrategy:
    name = "Baseline"

    def decide_buys(self, day: pd.Timestamp, candidates: pd.DataFrame, state: dict) -> list[InventoryItem]:
        threshold = candidates["rolling_median_sold"] * 0.82
        eligible = candidates[candidates["listed_price"] <= threshold].head(24)
        return [_make_inventory_item(day, row, row["rolling_median_sold"]) for _, row in eligible.iterrows()]

    def set_prices(self, day: pd.Timestamp, inventory: list[InventoryItem], state: dict) -> None:
        for item in inventory:
            holding_days = max((day - item.acquired_day).days, 0)
            item.ask_price = max(
                item.cost * 1.08,
                item.features["rolling_median_sold"] * (1.055 * (0.997 ** (holding_days // 7))),
            )


class MispricingLiquidityStrategy(BaseStrategy):
    name = "Mispricing + Liquidity"

    def decide_buys(self, day: pd.Timestamp, candidates: pd.DataFrame, state: dict) -> list[InventoryItem]:
        eligible = candidates[
            (candidates["z_score"] <= SETTINGS.z_buy)
            & (candidates["liquidity_7d"] >= SETTINGS.liquidity_min)
        ].copy()
        if eligible.empty:
            return []
        sale_probs = predict_sale_probability(state["sale_model"], eligible)
        eligible["sale_prob"] = sale_probs
        eligible = eligible.sort_values(["sale_prob", "z_score"], ascending=[False, True]).head(24)
        items: list[InventoryItem] = []
        for _, row in eligible.iterrows():
            ask = max(row["listed_price"] * 1.1, row["rolling_median_sold"] * (1 + 0.1 * row["sale_prob"]))
            items.append(_make_inventory_item(day, row, ask))
        return items


class MomentumRarityStrategy(BaseStrategy):
    name = "Momentum + Rarity"

    def decide_buys(self, day: pd.Timestamp, candidates: pd.DataFrame, state: dict) -> list[InventoryItem]:
        adjusted_fv = candidates["rolling_median_sold"] * np.exp(0.15 * candidates["momentum_30d"].clip(-1, 1))
        entry_edge = (adjusted_fv - candidates["listed_price"]) / adjusted_fv.replace(0, np.nan)
        tilt = entry_edge + 0.1 * candidates["rarity"] + 0.05 * candidates["momentum_30d"]
        eligible = candidates[tilt.fillna(-999) > 0.04].copy().head(24)
