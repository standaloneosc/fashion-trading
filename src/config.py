from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


@dataclass(slots=True)
class Settings:
    brands: list[str] = field(
        default_factory=lambda: [
            "Supreme",
            "Rick Owens",
            "Stone Island",
            "Comme des Garcons",
            "Our Legacy",
            "Maison Margiela",
            "Acne Studios",
            "Aimé Leon Dore",
