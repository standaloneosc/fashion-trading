from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(
    strategy_comparison: pd.DataFrame,
    equity_curves: pd.DataFrame,
    trades: pd.DataFrame,
    survival_curve: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
