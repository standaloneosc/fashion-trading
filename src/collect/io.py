from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, ensure_directories


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _concat_non_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load multi-venue panels from CSV files under ``data/``."""
    ensure_directories()

    listings_path = DATA_DIR / "listings_snapshot.csv"
    sold_paths = [
        DATA_DIR / "sold_items.csv",
        DATA_DIR / "depop_sold_items.csv",
        DATA_DIR / "grailed_sold_items.csv",
    ]

    listings = _load_csv(listings_path)
    if listings.empty:
        raise FileNotFoundError(
            f"No listing data at {listings_path}. "
            "Add a listings snapshot CSV under data/ (see README)."
        )

    supplemental_listings = [
        _load_csv(DATA_DIR / "depop_listings.csv"),
        _load_csv(DATA_DIR / "grailed_listings.csv"),
    ]
    listings = _concat_non_empty([listings, *supplemental_listings]).drop_duplicates(
        subset=["venue", "listing_id"], keep="last"
    )
    sold = _concat_non_empty([_load_csv(path) for path in sold_paths])
    if sold.empty:
        raise FileNotFoundError(
            "No sold history loaded. "
            "Ensure sold_items.csv (and optional depop_sold_items.csv, grailed_sold_items.csv) exist under data/."
        )
    return listings, sold
