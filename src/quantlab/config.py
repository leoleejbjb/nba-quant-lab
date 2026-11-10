"""Paths, seasons and constants shared across NBA Quant Lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QUANTLAB_DATA_DIR", PROJECT_ROOT / "data"))

# Seasons with liveData play-by-play in the pinned mirror (2020-21 ... 2025-26).
ALL_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
# Seasons with official NBA.com box scores in the pinned box-score mirror.
BOX_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24")

# Experimental design (fixed before any model was evaluated):
#   history  : 2020-21 .. 2022-23 -> priors, training data
#   tuning   : 2023-24            -> hyper-parameters, calibration, model choice
#   test     : 2024-25, 2025-26   -> final walk-forward evaluation, no re-tuning
HISTORY_SEASONS = ("2020-21", "2021-22", "2022-23")
TUNING_SEASON = "2023-24"
TEST_SEASONS = ("2024-25", "2025-26")

REGULATION_PERIOD_S = 720.0
OVERTIME_PERIOD_S = 300.0
REGULATION_S = 4 * REGULATION_PERIOD_S
TEAM_ID_MIN = 1_610_612_700  # NBA team ids are 16106127xx; person ids are smaller
MINUTES_TOLERANCE_S = 3.0    # box scores report whole seconds


def period_length(period: int) -> float:
    return REGULATION_PERIOD_S if period <= 4 else OVERTIME_PERIOD_S


def period_start(period: int) -> float:
    """Game seconds elapsed when ``period`` (1-based) starts."""
    if period <= 4:
        return (period - 1) * REGULATION_PERIOD_S
    return REGULATION_S + (period - 5) * OVERTIME_PERIOD_S


@dataclass(frozen=True)
class Season:
    label: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"\d{4}-\d{2}", self.label):
            raise ValueError(f"Season must look like '2023-24', got {self.label!r}")
        if (int(self.label[:4]) + 1) % 100 != int(self.label[5:]):
            raise ValueError(f"Inconsistent season label {self.label!r}")

    @property
    def start_year(self) -> int:
        return int(self.label[:4])

    @property
    def game_prefix(self) -> int:
        """Regular-season game ids are 002YYxxxxx -> integer prefix 2YY (e.g. 223 for 2023-24)."""
        return 200 + self.start_year % 100

    @property
    def has_box(self) -> bool:
        return self.label in BOX_SEASONS


def season_of_game(game_id: int) -> str:
    yy = (int(game_id) // 100_000) % 100
    y = 2000 + yy
    return f"{y}-{(y + 1) % 100:02d}"


@dataclass(frozen=True)
class Paths:
    root: Path = DATA_DIR

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    def staged(self, season: str) -> Path:
        return self.root / "staged" / season

    def processed(self, season: str) -> Path:
        return self.root / "processed" / season

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def backtest(self) -> Path:
        return self.root / "backtest"

    @property
    def sims(self) -> Path:
        return self.root / "sims"

    def ensure(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
