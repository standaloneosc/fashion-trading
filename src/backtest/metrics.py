from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(daily_pnl: pd.Series) -> float:
    pnl = daily_pnl.fillna(0.0)
    if pnl.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(252) * pnl.mean() / pnl.std(ddof=0))


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    return float(drawdown.min())

