import math

import pytest

from az_train import yardstick as ys

RULES = {
    "early_gen": 30,
    "early_min_mean_lag_score": 0.55,
    "rating_step": 20,
    "final_vs_half_min_score": 0.60,
    "last_quartile_min_lag_score": 0.55,
    "max_significant_cycles": 0,
    "max_capped_rate": 0.05,
}


def pairing(a, b, wins, losses, draws=0):
    return {"type": "pairing", "a": a, "b": b, "a_wins": wins, "b_wins": losses, "draws": draws}


def ladder_rows(gens, elo_of, games=100):
    """Every pair's expected result under a planted Elo, rounded to whole games (no noise)."""
    rows = []
    for i, a in enumerate(gens):
        for b in gens[i + 1 :]:
            p = 1 / (1 + 10 ** (-(elo_of(a) - elo_of(b)) / 400))
            wins = round(games * p)
            rows.append(pairing(f"gen{a}", f"gen{b}", wins, games - wins))
    return rows


def test_wilson_matches_the_known_interval_and_handles_extremes():
    lo, hi = ys.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert ys.wilson(0, 0) == (0.0, 1.0)
    lo, hi = ys.wilson(100, 100)
    assert 0.96 < lo < 1.0 and hi == pytest.approx(1.0)


def test_bradley_terry_recovers_a_planted_ladder():
    gens = [0, 10, 20, 30, 40, 50]
    planted = {g: 15.0 * g for g in gens}  # 0, 150, ..., 750 Elo
    report = ys.rating_report(ladder_rows(gens, planted.get), [f"gen{g}" for g in gens])
    elo = dict(zip(report["players"], report["elo"], strict=True))
    assert elo["gen0"] == 0.0
    for g in gens[1:]:
        # The pseudo-draw prior pulls extreme pairs in slightly; the ordering and spacing hold.
        assert elo[f"gen{g}"] == pytest.approx(planted[g], rel=0.12)
    ordered = [elo[f"gen{g}"] for g in gens]
    assert ordered == sorted(ordered)
    assert report["cycles"] == []
    assert report["deviance"] / max(1, report["df"]) < 1.0
    assert all(s >= 0 for s in report["se"]) and report["se"][0] == 0.0


def test_equal_players_rate_equally():
    rows = [pairing("gen0", "gen10", 50, 50), pairing("gen0", "gen20", 50, 50),
            pairing("gen10", "gen20", 50, 50)]  # fmt: skip
    report = ys.rating_report(rows, ["gen0", "gen10", "gen20"])
    assert report["elo"] == pytest.approx([0, 0, 0], abs=1e-6)
    assert report["cycles"] == []


def test_draws_count_half_and_a_perfect_score_stays_finite():
    p = ys.Pair("a", "b", 3, 1, 2)
    assert p.games == 6 and p.points == 4.0 and p.score == pytest.approx(4 / 6)
    fit = ys.fit_bradley_terry([ys.Pair("a", "b", 100, 0, 0)], order=["a", "b"])
    assert math.isfinite(fit["elo"][1]) and fit["elo"][1] < 0  # b is below a, and finite


def test_a_rock_paper_scissors_triple_is_a_cycle_and_a_poor_fit():
    rows = [pairing("gen0", "gen10", 80, 20), pairing("gen10", "gen20", 80, 20),
            pairing("gen20", "gen0", 80, 20)]  # fmt: skip
    report = ys.rating_report(rows, ["gen0", "gen10", "gen20"])
    assert len(report["cycles"]) == 1
    assert sorted(report["cycles"][0]) == ["gen0", "gen10", "gen20"]
    assert report["deviance"] > 20
    assert report["worst_residual"]["games"] == 100


def test_a_transitive_triple_with_one_insignificant_edge_has_no_cycle():
    rows = [pairing("gen0", "gen10", 70, 30), pairing("gen10", "gen20", 70, 30),
            pairing("gen0", "gen20", 45, 55)]  # fmt: skip
    assert ys.rating_report(rows, ["gen0", "gen10", "gen20"])["cycles"] == []


def test_repeated_pairing_rows_are_summed():
    rows = [pairing("a", "b", 10, 5), pairing("a", "b", 4, 1, 2)]
    (p,) = ys.pairs_from_rows(rows)
    assert (p.wins, p.losses, p.draws) == (14, 6, 2)


def test_rating_steps_pair_checkpoints_a_step_apart():
    steps = ys.rating_steps({10: 0.0, 20: 50.0, 30: 90.0, 40: 80.0}, 20)
    assert steps == [{"from": 10, "to": 30, "gain": 90.0}, {"from": 20, "to": 40, "gain": 30.0}]


def log_rows(n, lag_score, capped=0, black=0.5, value_std=0.4, ece=0.05):
    return [
        {
            "gen": g,
            "gate": {"lag": {"score": lag_score(g)}},
            "selfplay": {"games_started": 256, "games_finished": 256 - capped,
                         "games_capped": capped, "black_wins": int(black * (256 - capped))},
            "val_after": {"value_std": value_std},
            "diagnostics": {"calibration": {"ece": ece}},
        }
        for g in range(1, n + 1)
    ]  # fmt: skip


def evaluate(rows, ratings, total=100):
    return ys.evaluate_rules(RULES, 10, total, rows, ratings, 0.005)


def report_for(elo_of, gens=tuple(range(0, 101, 10)), games=100):
    return ys.rating_report(ladder_rows(list(gens), elo_of, games), [f"gen{g}" for g in gens])


def test_a_steadily_improving_run_passes():
    rows = log_rows(100, lambda g: 0.7)
    out = evaluate(rows, report_for(lambda g: 8.0 * g))
    assert out["verdict"] == "PASS", out["clauses"]
    assert all(c["status"] == "pass" for c in out["clauses"].values())


def test_an_early_stall_kills_the_run_at_generation_30():
    rows = log_rows(30, lambda g: 0.50)
    out = evaluate(rows, None)
    assert out["verdict"] == "KILL_EARLY"
    assert out["clauses"]["early"]["status"] == "fail"


def test_before_generation_30_the_early_clause_is_pending_and_the_verdict_pending():
    out = evaluate(log_rows(12, lambda g: 0.9), None)
    assert out["clauses"]["early"]["status"] == "pending"
    assert out["verdict"] == "PENDING"


def test_a_run_that_stops_improving_is_a_plateau_and_names_where():
    def elo(g):
        return 8.0 * min(g, 60) - 2.0 * max(0, g - 60)

    rows = log_rows(100, lambda g: 0.7 if g < 75 else 0.5)
    out = evaluate(rows, report_for(elo))
    assert out["verdict"] == "PLATEAU"
    assert out["plateau_from"] == 60  # 50 -> 70 still gains, 60 -> 80 does not
    assert out["clauses"]["final_vs_half"]["status"] == "fail"  # gen 100 is no better than gen 50
    assert out["clauses"]["late_lag"]["status"] == "fail"


def test_never_improving_ratings_fail_rather_than_plateau():
    out = evaluate(log_rows(100, lambda g: 0.7), report_for(lambda g: 0.0))
    assert out["verdict"] == "FAIL"


def test_a_cycle_or_capped_games_fail_the_run_even_with_rising_ratings():
    cyc = report_for(lambda g: 8.0 * g)
    cyc["cycles"] = [["gen10", "gen20", "gen30"]]
    assert evaluate(log_rows(100, lambda g: 0.7), cyc)["verdict"] == "FAIL"
    capped = evaluate(log_rows(100, lambda g: 0.7, capped=40), report_for(lambda g: 8.0 * g))
    assert capped["verdict"] == "FAIL"
    assert capped["clauses"]["health"]["capped_rate"] > 0.05


def test_health_warnings_are_reported_but_do_not_fail():
    rows = log_rows(100, lambda g: 0.7, black=0.9, ece=0.4)
    out = evaluate(rows, report_for(lambda g: 8.0 * g))
    assert out["verdict"] == "PASS"
    assert len(out["clauses"]["health"]["warnings"]) == 2
