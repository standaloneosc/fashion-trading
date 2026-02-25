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
