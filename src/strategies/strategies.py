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
