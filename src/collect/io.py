from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, ensure_directories


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

