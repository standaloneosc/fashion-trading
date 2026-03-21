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

    plt.figure(figsize=(10, 6))
    for strategy, group in equity_curves.groupby("strategy"):
        plt.plot(group["day"], group["equity"], label=strategy)
    plt.title("Cumulative PnL / Equity Curve")
    plt.xlabel("Day")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_pnl.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for strategy, group in equity_curves.groupby("strategy"):
        running_peak = group["equity"].cummax()
