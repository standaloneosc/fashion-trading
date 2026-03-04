from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.models.sale_prob import predict_sale_probability
from src.models.survival import predict_daily_hazard
from src.strategies.base import InventoryItem

