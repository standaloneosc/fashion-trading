from __future__ import annotations

"""
Signal mapping (resume language):
- Listing count / depth: active_depth, liquidity_7d / liquidity_30d, new_listings_24h
- Price spread: dispersion_iqr_30d, dispersion_std_log_30d, z_score
- Recent sales frequency: sell_through, momentum_30d, returns_7d
"""

import math

import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.features.normalize import normalize_market_frame


def _rolling_bucket_features(sold: pd.DataFrame) -> pd.DataFrame:
    if sold.empty:
        return sold

    sold = sold.sort_values("sold_at").copy()
    sold["bucket"] = sold["brand"] + "|" + sold["category"] + "|" + sold["size"]
    sold["sold_day"] = sold["sold_at"].dt.floor("D")
    sold["time_to_sale_days"] = (sold["sold_at"] - sold["created_at"]).dt.total_seconds().div(86400).clip(lower=1)

    feature_frames: list[pd.DataFrame] = []
