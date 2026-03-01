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


def fit_survival_model(df: pd.DataFrame, report_path: Path) -> SurvivalArtifacts:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    columns = SURVIVAL_FEATURES + ["duration_days", "event_observed"]
    training = df.reindex(columns=columns, fill_value=0.0).fillna(0.0)
    feature_columns = [
        column
        for column in SURVIVAL_FEATURES
        if column in training and training[column].nunique(dropna=False) > 1 and training[column].std(ddof=0) > 1e-8
    ]
    training = training[feature_columns + ["duration_days", "event_observed"]]

    if len(training) < 8 or training["duration_days"].nunique() < 2 or not feature_columns:
        metrics = {"concordance_index": 0.5, "coefficients": {}}
        report_path.write_text(json.dumps(metrics, indent=2))
        return SurvivalArtifacts(model=None, metrics=metrics)

    model = CoxPHFitter(penalizer=0.20)
    try:
        model.fit(training, duration_col="duration_days", event_col="event_observed")
    except ConvergenceError:
        metrics = {"concordance_index": 0.5, "coefficients": {}}
        report_path.write_text(json.dumps(metrics, indent=2))
        return SurvivalArtifacts(model=None, metrics=metrics)
    metrics = {
        "concordance_index": float(model.concordance_index_),
        "coefficients": {key: float(value) for key, value in model.params_.to_dict().items()},
    }
    report_path.write_text(json.dumps(metrics, indent=2))
    return SurvivalArtifacts(model=model, metrics=metrics)


