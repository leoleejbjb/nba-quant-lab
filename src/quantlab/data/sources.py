"""Pinned data sources (exact upstream commits + SHA-256 checksums).

* Play-by-play: the NBA ``cdn.nba.com`` liveData feed, collected per game by
  shufinskiy/nba_data (Apache-2.0). Six regular seasons, 2020-21 .. 2025-26.
* Official box scores: NBA.com player box scores and team game totals,
  collected by NocturneBear/NBA-Data-2010-2024 (MIT), seasons through 2023-24.
  Used to *validate* the box scores derived from play-by-play, to identify
  players who dressed but did not play (DNP - Coach's Decision) and to fill
  invisible period starters. They are never used as same-game model inputs.

The underlying statistics belong to NBA.com (Terms of Use: personal,
non-commercial use). The current season is not pinned: ``data.cdn`` fetches it
from cdn.nba.com (``quantlab live-update``). That CDN refused the build
environment ("Access Denied"); nothing here attempts to bypass a refusal. The
same pinned files (and checksums for 2023-24 .. 2025-26) are used by the
FiveMan lineup project.
"""

from __future__ import annotations

from dataclasses import dataclass

SHUFINSKIY = ("shufinskiy/nba_data", "e829d4678be1e075f99e5d41a1c5f97089be446b")
NOCTURNE = ("NocturneBear/NBA-Data-2010-2024", "a5f108b5b1f08074d78b9e8e901926a9ce4c06c5")


def raw_url(repo: tuple[str, str], path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{path}"


@dataclass(frozen=True)
class RawFile:
    filename: str
    url: str
    sha256: str
    repo: str
    commit: str
    license: str
    member: str | None = None  # CSV inside a .tar.xz archive


PBP_SHA256 = {
    2020: "e45e1b4a70a33ce93525c58f04362b6c275f7e514789772d4a8360f22a13f4f2",
    2021: "0c116aff50b829de70902023f108ee78c0f44bec569521ebf1fa292e01bda28f",
    2022: "de895e3474365fc14b028f514eb0f8c9c1503b95d3f1a5aef5a408305ddfb41f",
    2023: "33d49fefc809f73d5d6cbad6d1d6690e0df6c89fe2b5a4d814ecc9ea4df6101b",
    2024: "35a82e6b29821adb22b8474dea8c51e08847d8e00690d1d5ffde7d000e0087d1",
    2025: "a5d1bed30778430816b755676649cb2e3690ea62df5d96f4d9d6e9a8b8839339",
}
BOX_SHA256 = {
    "regular_season_box_scores_2010_2024_part_1.csv": "ca9636d08c577e3d752f69e32f90b173f4c3374c15f4dc8494b655049fdfc557",
    "regular_season_box_scores_2010_2024_part_2.csv": "5643af8a9f15ce3540dced759375ca79e77c71644e23270136f23b7a57fc182d",
    "regular_season_box_scores_2010_2024_part_3.csv": "65a01fee3df865a272e578cf4b365dce7dbc62eef2ce6f15f0844cf6694ef375",
}
TEAM_TOTALS = ("regular_season_totals_2010_2024.csv",
               "b63367be8c58155f978c841bb966f5b19e334cec833a37ddae2915a29cd352d9")

LIC_PBP = "Mirror Apache-2.0; data NBA.com Terms of Use (personal, non-commercial)"
LIC_BOX = "Mirror MIT; data NBA.com Terms of Use (personal, non-commercial)"


def pbp_file(start_year: int) -> RawFile:
    name = f"cdnnba_{start_year}.tar.xz"
    return RawFile(name, raw_url(SHUFINSKIY, f"datasets/{name}"), PBP_SHA256[start_year],
                   SHUFINSKIY[0], SHUFINSKIY[1], LIC_PBP, member=f"cdnnba_{start_year}.csv")


def box_files() -> list[RawFile]:
    out = [RawFile(n, raw_url(NOCTURNE, n), h, NOCTURNE[0], NOCTURNE[1], LIC_BOX) for n, h in BOX_SHA256.items()]
    return out


def team_totals_file() -> RawFile:
    n, h = TEAM_TOTALS
    return RawFile(n, raw_url(NOCTURNE, n), h, NOCTURNE[0], NOCTURNE[1], LIC_BOX)


def all_files(seasons: list[str]) -> list[RawFile]:
    years = sorted({int(s[:4]) for s in seasons})
    files = [pbp_file(y) for y in years]
    if any(y <= 2023 for y in years):
        files += box_files() + [team_totals_file()]
    return files
