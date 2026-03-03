from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(slots=True)
class InventoryItem:
    listing_id: str
    brand: str
    cost: float
    ask_price: float
    acquired_day: pd.Timestamp
    features: dict


class Strategy(Protocol):
    name: str

    def decide_buys(self, day: pd.Timestamp, candidates: pd.DataFrame, state: dict) -> list[InventoryItem]:
        ...
