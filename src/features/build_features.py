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
    for bucket, group in sold.groupby("bucket", sort=False):
        group = group.sort_values("sold_day").copy()
        rolling_prices = group["sold_price"].rolling(SETTINGS.dispersion_window_days, min_periods=1)
        group["rolling_median_sold"] = group["sold_price"].rolling(SETTINGS.momentum_window_days, min_periods=1).median()
        group["dispersion_iqr_30d"] = rolling_prices.quantile(0.75) - rolling_prices.quantile(0.25)
        group["dispersion_std_log_30d"] = np.log(group["sold_price"].clip(lower=1)).rolling(
            SETTINGS.dispersion_window_days, min_periods=2
        ).std()
        group["returns_7d"] = group["rolling_median_sold"].pct_change(7).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        group["liquidity_7d"] = group["sold_day"].rolling("7D", on=group["sold_day"]).count() if False else np.nan
        if len(group) >= 2:
            day_ordinal = (group["sold_day"] - group["sold_day"].min()).dt.days
            group["momentum_30d"] = (
                group["rolling_median_sold"]
                .rolling(SETTINGS.momentum_window_days, min_periods=2)
                .apply(lambda values: _safe_slope(values), raw=False)
                .fillna(0.0)
            )
