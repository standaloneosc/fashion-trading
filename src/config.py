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
