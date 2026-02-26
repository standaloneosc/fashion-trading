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
        model.fit(synthetic_X, synthetic_y)
        metrics = {"auc": 0.5, "log_loss": 0.693, "calibration_pred": [], "calibration_true": []}
        report_path.write_text(json.dumps(metrics, indent=2))
        return SaleProbabilityArtifacts(model=model, metrics=metrics)

    test_size = 0.25 if len(df) >= 20 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    calib_true, calib_pred = calibration_curve(y_test, probs, n_bins=min(5, len(y_test)))
    metrics = {
        "auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else 0.5,
        "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
        "calibration_pred": [float(value) for value in calib_pred],
        "calibration_true": [float(value) for value in calib_true],
    }
    report_path.write_text(json.dumps(metrics, indent=2))
