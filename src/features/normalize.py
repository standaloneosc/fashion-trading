from __future__ import annotations

import re

import pandas as pd

from src.config import SETTINGS


def canonicalize_brand(title: str | None, brand: str | None) -> str:
    raw = " ".join([str(title or ""), str(brand or "")]).lower()
    for candidate in SETTINGS.brands:
        if candidate.lower() in raw:
            return candidate
    return str(brand or "Unknown").strip() or "Unknown"


def infer_category(title: str | None, category: str | None) -> str:
    category = str(category or "").strip().lower()
    if category:
        return category

    title = str(title or "").lower()
    for canonical, keywords in SETTINGS.categories.items():
        if any(keyword in title for keyword in keywords):
            return canonical
    return "other"


def infer_size(title: str | None, size: str | None) -> str:
    normalized = str(size or "").strip().upper()
    if normalized:
        return normalized

    title = str(title or "").lower()
    for pattern in SETTINGS.size_patterns:
        match = re.search(pattern, title)
        if match:
            return match.group(0).upper()
    return "UNK"


def normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if normalized.empty:
        return normalized

    normalized["brand"] = [
        canonicalize_brand(title, brand)
        for title, brand in zip(normalized.get("title", ""), normalized.get("brand", ""))
    ]
    normalized["category"] = [
        infer_category(title, category)
        for title, category in zip(normalized.get("title", ""), normalized.get("category", ""))
    ]
    normalized["size"] = [
        infer_size(title, size)
        for title, size in zip(normalized.get("title", ""), normalized.get("size", ""))
    ]
    normalized["currency"] = normalized.get("currency", "USD").fillna("USD")

    for column in ("listed_price", "shipping_price", "seller_score", "sold_price"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in ("timestamp_observed", "created_at", "sold_at"):
