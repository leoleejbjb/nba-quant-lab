"""Season bookkeeping for the live feed: a finished current season stays in the database when the next begins."""

from __future__ import annotations

from quantlab import config


def test_a_finished_feed_season_stays_after_the_rollover(monkeypatch, tmp_path):
    for s in ("2025-26", "2026-27", "2027-28"):
        (tmp_path / "processed" / s).mkdir(parents=True)
    for s in ("2025-26", "2026-27"):
        (tmp_path / "processed" / s / "games.parquet").write_bytes(b"")
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))
    monkeypatch.setattr(config, "LIVE_SEASON", "2027-28")
    assert config.db_seasons() == config.ALL_SEASONS + ("2026-27",)       # the new season has no games yet
    (tmp_path / "processed" / "2027-28" / "games.parquet").write_bytes(b"")
    assert config.db_seasons()[-2:] == ("2026-27", "2027-28")
    assert config.Season("2026-27").is_live and config.Season("2027-28").is_live   # built from the feed files
    assert not config.Season("2025-26").is_live                                     # pinned mirror


def test_the_rehearsal_treats_a_pinned_season_as_live(monkeypatch, tmp_path):
    (tmp_path / "processed" / "2025-26").mkdir(parents=True)
    (tmp_path / "processed" / "2025-26" / "games.parquet").write_bytes(b"")
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))
    monkeypatch.setattr(config, "LIVE_SEASON", "2025-26")
    monkeypatch.setattr(config, "ALL_SEASONS", tuple(s for s in config.PINNED_SEASONS if s != "2025-26"))
    assert config.db_seasons()[-1] == "2025-26" and config.db_seasons().count("2025-26") == 1
    assert config.Season("2025-26").is_live
