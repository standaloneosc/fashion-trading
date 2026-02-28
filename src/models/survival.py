from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

SURVIVAL_FEATURES = [
    "price_ratio",
    "z_score",
    "active_depth",
    "momentum_30d",
    "dispersion_iqr_30d",
    "dispersion_std_log_30d",
    "rarity",
    "seller_score",
    "shipping_price",
    "new_listings_7d",
    "month",
    "day_of_week",
]


@dataclass(slots=True)
class SurvivalArtifacts:
    model: CoxPHFitter | None
    metrics: dict[str, float | dict[str, float]]

