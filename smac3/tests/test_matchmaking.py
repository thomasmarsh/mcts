"""`matchmaking.play_trial`: closest-opponent selection, min/max game bounds,
sigma-threshold early stop -- with `target.play_game` monkeypatched to return
scripted win/loss/draw sequences so no real subprocess runs.
"""

from __future__ import annotations

import yaml

from smac3_cli import matchmaking
from smac3_cli.config import SearchConfig
from smac3_cli.pool import OpponentPool

_SPACE_YAML = """
parameters:
  family:
    type: categorical
    choices: [rave, ucb1_pn]
    default: rave
"""


def _cfg() -> SearchConfig:
    return SearchConfig._from_dict(yaml.safe_load(_SPACE_YAML))


def _pool() -> OpponentPool:
    return OpponentPool.bootstrap(_cfg())  # anchors: "default" mu=25.0, "random" mu=0.0


def test_min_games_floor_even_when_sigma_drops_early(monkeypatch):
    """5 lopsided wins would normally crash sigma below threshold fast, but
    the loop must still run at least `min_games` steps."""
    calls = []

    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        calls.append(opponent_config)
        return 20, 0, 0, None

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    mu, sigma, games = matchmaking.play_trial(
        _cfg(),
        "binary",
        {"family": "rave"},
        _pool(),
        seed_base=0,
        sigma_threshold=100.0,
    )

    assert len(calls) == matchmaking._MIN_GAMES
    assert len(games) == matchmaking._MIN_GAMES * 20
    assert all(g["outcome"] == "win" for g in games)


def test_max_games_ceiling_when_sigma_never_converges(monkeypatch):
    """Alternating win/loss keeps sigma high; the loop must still stop at
    `max_games` steps."""

    calls = []

    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        calls.append(seed)
        return 0, 0, 0, "crashed"

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    mu, sigma, games = matchmaking.play_trial(
        _cfg(), "binary", {"family": "rave"}, _pool(), seed_base=0
    )

    assert len(calls) == matchmaking._MAX_GAMES
    assert games == []


def test_sigma_threshold_stops_early_past_min_games(monkeypatch):
    """A big fixed win-count each step should converge sigma below the
    threshold well before `max_games`, so the loop stops in between."""

    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        return 20, 0, 0, None

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    mu, sigma, games = matchmaking.play_trial(
        _cfg(), "binary", {"family": "rave"}, _pool(), seed_base=0
    )

    assert sigma < matchmaking._SIGMA_THRESHOLD
    assert matchmaking._MIN_GAMES <= len(games) < matchmaking._MAX_GAMES * 20


def test_closest_opponent_changes_as_rating_moves(monkeypatch):
    """Winning repeatedly should raise the candidate's mu enough that
    closest() picks a higher-mu anchor than the one it started against."""
    opponents_seen = []

    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        opponents_seen.append(opponent_config["family"])
        return 1, 0, 0, None

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    pool = _pool()
    pool.maybe_insert({"family": "strong"}, mu=40.0, sigma=1.0)  # new champion anchor

    matchmaking.play_trial(_cfg(), "binary", {"family": "ucb1_pn"}, pool, seed_base=0)

    # An unrated candidate (mu=25.0) starts against "default" (also mu=25.0).
    assert opponents_seen[0] == "rave"
    # Enough wins should climb the candidate's mu past 25 and toward "strong".
    assert opponents_seen[-1] == "strong"


def test_crashed_step_counts_toward_bounds_but_logs_no_game(monkeypatch):
    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        return 0, 0, 0, "crashed"

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    mu, sigma, games = matchmaking.play_trial(
        _cfg(), "binary", {"family": "rave"}, _pool(), seed_base=0
    )

    assert games == []
    assert mu == matchmaking.trueskill.Rating().mu


def test_opponent_rating_never_mutates_pool_anchor(monkeypatch):
    def fake_play_game(cfg, binary, candidate_config, opponent_config, *, seed, trace_path=None):
        return 1, 0, 0, None

    monkeypatch.setattr(matchmaking, "play_game", fake_play_game)

    pool = _pool()
    before = [(a.id, a.mu, a.sigma) for a in pool.anchors]

    matchmaking.play_trial(_cfg(), "binary", {"family": "rave"}, pool, seed_base=0)

    after = [(a.id, a.mu, a.sigma) for a in pool.anchors]
    assert before == after
