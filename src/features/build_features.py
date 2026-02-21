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
        else:
            group["momentum_30d"] = 0.0
        feature_frames.append(group)

    featured = pd.concat(feature_frames, ignore_index=True)
    return featured


def _safe_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = values.to_numpy(dtype=float)
    if np.isnan(y).all():
        return 0.0
    slope = np.polyfit(x, np.nan_to_num(y, nan=np.nanmedian(y)), 1)[0]
    return float(slope)


def _active_snapshot_features(listings: pd.DataFrame) -> pd.DataFrame:
    if listings.empty:
        return listings

    listings = listings.sort_values("timestamp_observed").copy()
    listings["bucket"] = listings["brand"] + "|" + listings["category"] + "|" + listings["size"]
    listings["obs_day"] = listings["timestamp_observed"].dt.floor("D")
    listings["listing_age_days"] = (
        (listings["timestamp_observed"] - listings["created_at"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
    )

    frames: list[pd.DataFrame] = []
    for bucket, group in listings.groupby("bucket", sort=False):
        group = group.sort_values("obs_day").copy()
        daily_counts = group.groupby("obs_day")["listing_id"].transform("count")
        group["active_depth"] = daily_counts
        group["new_listings_24h"] = group.groupby("obs_day")["listing_id"].transform("count")
        counts_by_day = group.groupby("obs_day")["listing_id"].transform("count")
        group["new_listings_7d"] = counts_by_day.rolling(7, min_periods=1).sum().to_numpy()
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def build_feature_store(listings: pd.DataFrame, sold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    listings = normalize_market_frame(listings)
    sold = normalize_market_frame(sold)
