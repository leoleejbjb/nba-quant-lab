"""Rehearse the live season on a finished one, and measure how good truly pregame projections are.

    .venv/bin/python scripts/rehearse_live.py prepare  --cutoff 2026-01-15 --dir <folder>
    .venv/bin/python scripts/rehearse_live.py update   --dir <folder>
    .venv/bin/python scripts/rehearse_live.py evaluate --dir <folder> [--sims 2000] [--score-only]

``prepare`` makes a self-contained data folder in which "today" is ``--cutoff`` in the 2025-26 season: the pinned
seasons before it are linked from ``data/`` (read-only), the fitted parameters are copied, and 2025-26 is written in
the live format - a schedule in which games before the cutoff are final and the rest are scheduled, and a play-by-play
file per final game (rebuilt from the pinned mirror, exactly as cdn.nba.com serves it). ``update`` runs the real
``quantlab live-update --skip-fetch`` on that folder, with 2025-26 as the live season. ``evaluate`` projects every
unplayed game in the horizon pregame (projected starters and availability, no information from the game) and scores
the projections against what happened, next to the backtested as-of projections of the same games.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data"
SEASON = "2025-26"
PINNED = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
PARAMS = ("rate_params.json", "team_factor_params.json", "sim_constants.json", "integration.json")


def _link(src: Path, dst: Path) -> None:
    """Link a pinned file or season folder into the rehearsal folder (read-only use). Windows without the symlink
    privilege (no Developer Mode): a hard link for a file, a directory junction for a folder."""
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if os.name != "nt":
            raise
        if src.is_dir():
            import _winapi
            _winapi.CreateJunction(str(src), str(dst))
        else:
            try:
                os.link(src, dst)
            except OSError:                            # another drive: hard links cannot cross volumes
                shutil.copy2(src, dst)


def _check_folder(folder: Path) -> Path:
    """The rehearsal's data folder - never the project's own data, and an existing one only if a rehearsal made it."""
    data = (folder / "data").resolve()
    main = MAIN.resolve()
    if data == main or main in data.parents or data in main.parents:
        raise SystemExit(f"--dir {folder}: the rehearsal needs a folder of its own, not the project's data ({main})")
    if data.exists() and any(data.iterdir()) and not (folder / "rehearsal.json").exists():
        raise SystemExit(f"--dir {folder}: {data} exists and was not made by a rehearsal (no rehearsal.json); "
                         "choose a new or empty folder")
    return data


def env_for(folder: Path, horizon: int) -> dict:
    env = dict(os.environ, QUANTLAB_DATA_DIR=str(folder / "data"), QUANTLAB_LIVE_SEASON=SEASON,
               QUANTLAB_UPCOMING_DAYS=str(horizon), PYTHONUTF8="1")
    return env


def prepare(folder: Path, cutoff: pd.Timestamp) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    data = _check_folder(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rehearsal.json").write_text('{"complete": false}', encoding="utf-8")   # this folder is a rehearsal's
    if data.exists():
        shutil.rmtree(data)
    (data / "raw").mkdir(parents=True)
    for f in (MAIN / "raw").iterdir():                 # pinned files, linked (read-only use)
        if f.is_file():
            _link(f, data / "raw" / f.name)
    for s in PINNED:
        for sub in ("processed", "staged"):
            (data / sub).mkdir(exist_ok=True)
            _link(MAIN / sub / s, data / sub / s)
    (data / "models").mkdir()
    for f in PARAMS:
        if (MAIN / "models" / f).exists():
            shutil.copy2(MAIN / "models" / f, data / "models" / f)

    os.environ["QUANTLAB_DATA_DIR"] = str(data)       # before importing quantlab: PATHS points at the folder
    os.environ["QUANTLAB_LIVE_SEASON"] = SEASON
    from quantlab.data.cdn import doc_from_mirror_rows, pbp_path, schedule_doc, schedule_path, write_json_gz

    games = pd.read_parquet(MAIN / "processed" / SEASON / "games.parquet")
    final = set(games.loc[pd.to_datetime(games["game_date"]) < cutoff, "game_id"].astype(int))
    write_json_gz(schedule_doc(games, final), schedule_path(SEASON))
    with tarfile.open(MAIN / "raw" / "cdnnba_2025.tar.xz", "r:xz") as t:
        csv = pd.read_csv(io.BufferedReader(t.extractfile("cdnnba_2025.csv")), low_memory=False)
    csv = csv[csv["gameId"].isin(final)]
    for gid, rows in csv.groupby("gameId"):
        write_json_gz(doc_from_mirror_rows(rows), pbp_path(SEASON, gid))
    (folder / "rehearsal.json").write_text(json.dumps({"season": SEASON, "cutoff": str(cutoff.date()),
                                                       "final_games": len(final), "games": int(len(games)),
                                                       "complete": True}, indent=1), encoding="utf-8")
    print(f"[rehearse] {folder}: {len(final)} of {len(games)} {SEASON} games final before {cutoff.date()}")


def update(folder: Path, horizon: int) -> None:
    cmd = [sys.executable, "-m", "quantlab.cli", "live-update", "--skip-fetch"]
    subprocess.run(cmd, check=True, env=env_for(folder, horizon), cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "update", "evaluate", "all"])
    ap.add_argument("--dir", required=True, type=Path, help="a new folder (it gets a data/ of its own; never the project's data folder)")
    ap.add_argument("--cutoff", default="2026-01-15")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--score-only", action="store_true", help="evaluate: re-score the saved projections")
    a = ap.parse_args()
    if a.step in ("prepare", "all"):
        _check_folder(a.dir)                          # before anything is created
    else:
        a.dir.mkdir(parents=True, exist_ok=True)
    if a.step in ("prepare", "all"):
        prepare(a.dir, pd.Timestamp(a.cutoff))
    if a.step in ("update", "all"):
        update(a.dir, a.horizon)
    if a.step in ("evaluate", "all"):
        cmd = [sys.executable, "-m", "quantlab.eval.pregame", "--sims", str(a.sims),
               "--main-data", str(MAIN), "--out", str(a.dir / "pregame_eval.json")] + \
            (["--score-only"] if a.score_only else [])
        subprocess.run(cmd, check=True, env=env_for(a.dir, a.horizon), cwd=ROOT)


if __name__ == "__main__":
    main()
