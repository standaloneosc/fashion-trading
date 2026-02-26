from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import MODEL_FEATURES


@dataclass(slots=True)
class SaleProbabilityArtifacts:
    model: Pipeline
    metrics: dict[str, float | list[float]]


def fit_sale_probability_model(df: pd.DataFrame, report_path: Path) -> SaleProbabilityArtifacts:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    X = df.reindex(columns=MODEL_FEATURES, fill_value=0.0).fillna(0.0)
    y = df["y_sale_within_horizon"].astype(int)

    if y.nunique() < 2 or len(df) < 8:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        synthetic_X = pd.concat([X, X]).reset_index(drop=True)
        synthetic_y = pd.Series(([0] * len(X)) + ([1] * len(X)))
