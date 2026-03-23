from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest.metrics import aggregate_rollout_summaries
from src.backtest.simulator import run_strategy_backtest
from src.backtest.stress import run_stress_tests
from src.collect.io import load_market_data
from src.config import PLOTS_DIR, REPORTS_DIR, SETTINGS, ensure_directories
from src.features.build_features import build_feature_store, build_training_frame
