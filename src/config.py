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
            "Alexander McQueen",
            "Amiri",
            "Alyx",
            "Arc'teryx",
            "Balenciaga",
            "Balmain",
            "BAPE",
            "Bode",
            "Boramy Viguier",
            "Bottega Veneta",
            "Brunello Cucinelli",
            "Burberry",
            "C.P. Company",
            "Canada Goose",
            "Carhartt WIP",
            "Cav Empt",
            "Celine",
            "Chrome Hearts",
            "Craig Green",
            "Daiwa Pier39",
            "Dries Van Noten",
            "Engineered Garments",
            "Erdem",
            "Essentials",
            "Fear of God",
            "Fendi",
            "Givenchy",
            "Guidi",
            "Helmut Lang",
            "Human Made",
            "Isabel Marant",
            "Issey Miyake",
            "Jacquemus",
            "Jean Paul Gaultier",
            "Jil Sander",
            "Junya Watanabe",
            "Kapital",
            "Kiko Kostadinov",
            "Lanvin",
            "Lemaire",
            "Louis Vuitton",
            "Maison Mihara Yasuhiro",
            "Marine Serre",
            "Marni",
            "Moncler",
            "Needles",
            "Neighborhood",
            "New Balance",
            "Noah",
            "Number (N)ine",
            "Off-White",
            "OAMC",
            "Palace",
            "Prada",
            "Raf Simons",
            "Red Wing",
            "Reigning Champ",
            "Represent",
            "Roa",
            "Sacai",
            "Salomon",
            "Sandy Liang",
            "Sasquatchfabrix",
            "Stella McCartney",
            "Stussy",
            "Sunnei",
            "Takahiromiyashita TheSoloist",
            "The North Face",
            "Thom Browne",
            "Tom Ford",
            "Undercover",
            "Uniqlo",
            "Vejas",
            "VETEMENTS",
            "Visvim",
            "Wacko Maria",
            "Wales Bonner",
            "Y-3",
            "Y/Project",
            "Yeezy",
            "Yohji Yamamoto",
            "A.P.C.",
            "And Wander",
            "Auralee",
            "Barbour",
            "Beams Plus",
            "Common Projects",
            "Danner",
            "Evisu",
            "Filson",
            "Golden Goose",
            "Hoka",
            "John Elliott",
            "Kith",
            "Mackage",
            "Miharayasuhiro",
            "N.Hoolywood",
            "Nanamica",
            "Norse Projects",
            "Porter Yoshida",
            "RRL",
            "Satisfy",
            "Snow Peak",
            "Taion",
            "Veilance",
            "WTAPS",
        ]
    )
    categories: dict[str, list[str]] = field(
        default_factory=lambda: {
            "outerwear": ["hoodie", "jacket", "coat", "parka", "fleece"],
            "top": ["shirt", "tee", "t-shirt", "sweater", "knit"],
            "bottoms": ["pants", "trousers", "jeans", "cargo", "shorts"],
            "footwear": ["boots", "sneakers", "shoes", "ramones", "loafer"],
            "accessories": ["bag", "cap", "hat", "wallet", "belt"],
        }
    )
    size_patterns: list[str] = field(
        default_factory=lambda: [
            r"\bxxs\b",
            r"\bxs\b",
            r"\bs\b",
            r"\bm\b",
            r"\bl\b",
            r"\bxl\b",
            r"\bxxl\b",
            r"\b\d{2}\b",
            r"\b\d{1,2}\.\d\b",
        ]
    )
    horizon_days: int = 14
    liquidity_windows: tuple[int, int] = (7, 30)
    momentum_window_days: int = 30
    dispersion_window_days: int = 30
    fee_rate: float = 0.12
    fixed_fee: float = 0.0
    default_shipping_cost: float = 12.0
    hold_cost_bps_day: float = 5.0
    slip_mean: float = -0.02
    slip_std: float = 0.03
    max_inventory: int = field(default_factory=lambda: int(os.getenv("MAX_INVENTORY", "48")))
    max_brand_inventory: int = field(default_factory=lambda: int(os.getenv("MAX_BRAND_INVENTORY", "8")))
    starting_cash: float = 25000.0
    z_buy: float = -1.0
    liquidity_min: float = 1.0
    p_target: float = 0.60
    demand_shock_scale: float = 0.60
    regime_shift_dispersion_scale: float = 1.50
    adversarial_price_scale: float = 1.75
