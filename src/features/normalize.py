from __future__ import annotations

import re

import pandas as pd

from src.config import SETTINGS


def canonicalize_brand(title: str | None, brand: str | None) -> str:
    raw = " ".join([str(title or ""), str(brand or "")]).lower()
    for candidate in SETTINGS.brands:
