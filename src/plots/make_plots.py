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
        drawdown = group["equity"] - running_peak
        plt.plot(group["day"], drawdown, label=strategy)
    plt.title("Drawdown")
    plt.xlabel("Day")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    if not trades.empty:
        for strategy, group in trades.groupby("strategy"):
            plt.hist(group["profit"], bins=15, alpha=0.45, label=strategy)
        plt.legend()
    plt.title("Profit Per Trade Distribution")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "profit_hist.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(survival_curve["days"], survival_curve["survival"])
    plt.title("Empirical Survival Curve")
