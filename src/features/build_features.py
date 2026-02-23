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
    if not sold.empty:
        sold = sold.dropna(subset=["sold_at", "sold_price", "created_at"])
    if not listings.empty:
        listings = listings.dropna(subset=["timestamp_observed"])

    sold_features = _rolling_bucket_features(sold)
    listing_features = _active_snapshot_features(listings)

    if sold_features.empty or listing_features.empty:
        listing_features["rolling_median_sold"] = listing_features.get("listed_price", 0.0)
        listing_features["dispersion_iqr_30d"] = 0.0
        listing_features["dispersion_std_log_30d"] = 0.0
        listing_features["momentum_30d"] = 0.0
        listing_features["returns_7d"] = 0.0
        listing_features["liquidity_7d"] = 0.0
        listing_features["liquidity_30d"] = 0.0
        listing_features["sell_through"] = 0.0
        listing_features["rarity"] = 1 / (1 + listing_features["active_depth"].fillna(0))
        return listing_features, sold_features

    sold_agg = (
        sold_features.groupby("bucket", as_index=False)
        .agg(
            rolling_median_sold=("rolling_median_sold", "last"),
            dispersion_iqr_30d=("dispersion_iqr_30d", "last"),
            dispersion_std_log_30d=("dispersion_std_log_30d", "last"),
            momentum_30d=("momentum_30d", "last"),
            returns_7d=("returns_7d", "last"),
            liquidity_7d=("sold_price", lambda s: s.tail(7).count()),
            liquidity_30d=("sold_price", lambda s: s.tail(30).count()),
        )
    )

    featured = listing_features.merge(sold_agg, on="bucket", how="left")
    featured["rolling_median_sold"] = featured["rolling_median_sold"].fillna(featured["listed_price"])
    featured["dispersion_iqr_30d"] = featured["dispersion_iqr_30d"].fillna(0.0)
    featured["dispersion_std_log_30d"] = featured["dispersion_std_log_30d"].fillna(0.0)
    featured["momentum_30d"] = featured["momentum_30d"].fillna(0.0)
    featured["returns_7d"] = featured["returns_7d"].fillna(0.0)
    featured["liquidity_7d"] = featured["liquidity_7d"].fillna(0.0)
    featured["liquidity_30d"] = featured["liquidity_30d"].fillna(0.0)
    featured["sell_through"] = featured["liquidity_30d"] / (featured["liquidity_30d"] + featured["active_depth"].clip(lower=0) + 1e-9)
    featured["rarity"] = 1 / (1 + featured["active_depth"].clip(lower=0))
    featured["price_ratio"] = np.where(
        featured["rolling_median_sold"].fillna(0) > 0,
        featured["listed_price"] / featured["rolling_median_sold"].replace(0, np.nan),
        1.0,
    )
    featured["z_score"] = (
        np.log(featured["listed_price"].clip(lower=1)) - np.log(featured["rolling_median_sold"].clip(lower=1))
    ) / featured["dispersion_std_log_30d"].replace(0, np.nan)
    featured["z_score"] = featured["z_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    featured["month"] = featured["timestamp_observed"].dt.month
    featured["day_of_week"] = featured["timestamp_observed"].dt.dayofweek
    return featured, sold_features


def build_training_frame(featured_listings: pd.DataFrame, sold_features: pd.DataFrame) -> pd.DataFrame:
    if sold_features.empty:
        base = featured_listings.copy()
        base["y_sale_within_horizon"] = (base["liquidity_7d"] > 0).astype(int)
        base["duration_days"] = base["listing_age_days"].clip(lower=1) + SETTINGS.horizon_days
        base["event_observed"] = 0
        return base

    history = sold_features.copy()
    history["y_sale_within_horizon"] = (history["time_to_sale_days"] <= SETTINGS.horizon_days).astype(int)
    history["duration_days"] = history["time_to_sale_days"].clip(lower=1)
    history["event_observed"] = 1
    history["price_ratio"] = history["listed_price"] / history["rolling_median_sold"].replace(0, np.nan)
    history["price_ratio"] = history["price_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    history["z_score"] = (
        np.log(history["listed_price"].clip(lower=1)) - np.log(history["rolling_median_sold"].clip(lower=1))
