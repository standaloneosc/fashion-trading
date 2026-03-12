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


def summarize_trades(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "sim_trades": 0,
            "avg_profit_trade": 0.0,
            "median_profit_trade": 0.0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "median_time_to_sale": 0.0,
            "p90_time_to_sale": 0.0,
            "win_rate": 0.0,
            "avg_inventory": float(equity_curve.get("inventory_count", pd.Series([0])).mean()),
            "turnover": 0.0,
        }

    daily_pnl = equity_curve.set_index("day")["daily_pnl"]
    return {
        "sim_trades": int(len(trades)),
        "avg_profit_trade": float(trades["profit"].mean()),
        "median_profit_trade": float(trades["profit"].median()),
        "total_pnl": float(trades["profit"].sum()),
        "sharpe": sharpe_ratio(daily_pnl),
        "max_drawdown": max_drawdown(equity_curve.set_index("day")["equity"]),
        "median_time_to_sale": float(trades["holding_days"].median()),
        "p90_time_to_sale": float(trades["holding_days"].quantile(0.90)),
        "win_rate": float((trades["profit"] > 0).mean()),
        "avg_inventory": float(equity_curve["inventory_count"].mean()),
        "turnover": float(len(trades) / max(equity_curve["inventory_count"].mean(), 1)),
    }


def aggregate_rollout_summaries(trades_frames: list[pd.DataFrame], equity_frames: list[pd.DataFrame]) -> dict[str, float]:
    non_empty_trades = [frame for frame in trades_frames if not frame.empty]
    non_empty_curves = [frame for frame in equity_frames if not frame.empty]
    if not non_empty_trades:
        placeholder = pd.DataFrame(columns=["day", "daily_pnl", "equity", "inventory_count"])
        return summarize_trades(pd.DataFrame(), placeholder)

    combined = pd.concat(non_empty_trades, ignore_index=True)
    sharpes = [
        sharpe_ratio(frame.set_index("day")["daily_pnl"]) for frame in non_empty_curves if "daily_pnl" in frame.columns
    ]
    drawdowns = [
        max_drawdown(frame.set_index("day")["equity"]) for frame in non_empty_curves if "equity" in frame.columns
    ]
    inventories = [float(frame["inventory_count"].mean()) for frame in non_empty_curves]
    sharpe_value = float(np.mean(sharpes)) if sharpes else 0.0
    drawdown_value = float(np.mean(drawdowns)) if drawdowns else 0.0
    avg_inventory = float(np.mean(inventories)) if inventories else 0.0

    return {
        "sim_trades": int(len(combined)),
        "avg_profit_trade": float(combined["profit"].mean()),
        "median_profit_trade": float(combined["profit"].median()),
        "total_pnl": float(combined["profit"].sum()),
        "sharpe": sharpe_value,
        "max_drawdown": drawdown_value,
        "median_time_to_sale": float(combined["holding_days"].median()),
