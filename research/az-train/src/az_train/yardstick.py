"""Reference-free progress measurement for a self-play training run: rate checkpoints from nets
playing nets, find intransitivity, and judge a run against pre-registered rules.

Nothing here knows the game, an engine or an oracle. The inputs are pairing rows (what the paired
match harness writes: ``type = "pairing"`` with ``a``, ``b``, ``a_wins``, ``b_wins``, ``draws``) and
the coordinator's per-generation log rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def wilson(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``wins`` successes (draws count one half) in ``games`` games."""
    if games == 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return centre - half, centre + half


@dataclass(frozen=True)
class Pair:
    """One pairing's totals from ``a``'s side."""

    a: str
    b: str
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score(self) -> float:
        return self.points / self.games if self.games else 0.5

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        return wilson(self.points, self.games, z)

    def beats(self) -> bool:
        """``a`` is significantly ahead: the Wilson lower bound of its score exceeds one half."""
        return self.games > 0 and self.wilson()[0] > 0.5


def pairs_from_rows(rows: list[dict[str, Any]]) -> list[Pair]:
    """Pairing rows -> pairs; repeated ``(a, b)`` rows (a resumed run) are summed."""
    totals: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        if r.get("type") != "pairing":
            continue
        t = totals.setdefault((r["a"], r["b"]), [0, 0, 0])
        t[0] += r["a_wins"]
        t[1] += r["b_wins"]
        t[2] += r["draws"]
    return [Pair(a, b, *t) for (a, b), t in totals.items()]


def _players(pairs: list[Pair], order: list[str] | None) -> list[str]:
    names = list(order or [])
    for p in pairs:
        for n in (p.a, p.b):
            if n not in names:
                names.append(n)
    return names


def fit_bradley_terry(
    pairs: list[Pair], *, order: list[str] | None = None, prior_games: float = 1.0
) -> dict[str, Any]:
    """Maximum-likelihood Bradley-Terry ratings in Elo units, the first player of ``order`` at 0.

    ``P(i beats j) = 1 / (1 + exp(-(r_i - r_j)))``, a draw is half a win. ``prior_games`` adds that
    many virtual drawn games to every pair so ratings stay finite when a checkpoint wins all its
    games (a 100 percent score otherwise diverges); it is a tiny weight against 100-game pairs.
    ``se`` is the standard error of each rating *relative to the anchor* (from the inverse
    information matrix), so the anchor's is 0.

    Lack of fit is the binomial deviance ``G^2`` of the pair scores against the fit with its
    degrees of freedom (pairs minus rated players plus one); a large ratio means the results are
    not explained by one number per checkpoint, i.e. intransitivity.
    """
    names = _players(pairs, order)
    index = {n: i for i, n in enumerate(names)}
    m = len(names)
    if m < 2 or not pairs:
        return {"players": names, "elo": [0.0] * m, "se": [0.0] * m, "deviance": 0.0, "df": 0}
    n_eff = np.array([p.games + prior_games for p in pairs], dtype=float)
    s_eff = np.array([p.points + prior_games / 2 for p in pairs], dtype=float)
    ia = np.array([index[p.a] for p in pairs])
    ib = np.array([index[p.b] for p in pairs])
    r = np.zeros(m)
    for _ in range(100):
        p_a = 1 / (1 + np.exp(-(r[ia] - r[ib])))
        resid = s_eff - n_eff * p_a
        grad = np.zeros(m)
        np.add.at(grad, ia, resid)
        np.add.at(grad, ib, -resid)
        w = n_eff * p_a * (1 - p_a)
        hess = np.zeros((m, m))
        np.add.at(hess, (ia, ia), w)
        np.add.at(hess, (ib, ib), w)
        np.add.at(hess, (ia, ib), -w)
        np.add.at(hess, (ib, ia), -w)
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        r = r + step
        r -= r.mean()
        if np.abs(step).max() < 1e-10:
            break
    p_a = 1 / (1 + np.exp(-(r[ia] - r[ib])))
    w = n_eff * p_a * (1 - p_a)
    hess = np.zeros((m, m))
    np.add.at(hess, (ia, ia), w)
    np.add.at(hess, (ib, ib), w)
    np.add.at(hess, (ia, ib), -w)
    np.add.at(hess, (ib, ia), -w)
    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.maximum(cov.diagonal() + cov[0, 0] - 2 * cov[:, 0], 0.0))

    games = np.array([p.games for p in pairs], dtype=float)
    points = np.array([p.points for p in pairs], dtype=float)

    def xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, x * np.log(np.where(y > 0, y, 1.0)), 0.0)

    expected = games * p_a
    deviance = float(
        2 * np.sum(xlogy(points, points / np.maximum(expected, 1e-12))
                   + xlogy(games - points, (games - points) / np.maximum(games - expected, 1e-12)))
    )  # fmt: skip
    df = len(pairs) - (m - 1)
    residuals = [
        {
            "a": p.a,
            "b": p.b,
            "games": p.games,
            "score": p.score,
            "fitted": float(p_a[k]),
            "residual": p.score - float(p_a[k]),
        }
        for k, p in enumerate(pairs)
    ]
    r0 = r - r[0]
    return {
        "players": names,
        "elo": [float(x * ELO_PER_LOGIT) for x in r0],
        "se": [float(x * ELO_PER_LOGIT) for x in se],
        "deviance": deviance,
        "df": df,
        "worst_residual": max(residuals, key=lambda x: abs(x["residual"])) if residuals else None,
        "residuals": residuals,
    }


def significant_cycles(pairs: list[Pair]) -> list[tuple[str, str, str]]:
    """Directed 3-cycles ``a > b > c > a`` where every edge is significant (Wilson lower bound of
    the winner's score above one half). Each cycle is listed once, starting at its smallest name."""
    beats: set[tuple[str, str]] = set()
    for p in pairs:
        if p.beats():
            beats.add((p.a, p.b))
        else:
            flipped = Pair(p.b, p.a, p.losses, p.wins, p.draws)
            if flipped.beats():
                beats.add((p.b, p.a))
    names = sorted({n for e in beats for n in e})
    cycles = []
    for x, y, z in combinations(names, 3):
        for a, b, c in ((x, y, z), (x, z, y)):
            if (a, b) in beats and (b, c) in beats and (c, a) in beats:
                cycles.append((a, b, c))
    return cycles


def rating_report(
    rows: list[dict[str, Any]], order: list[str], prior_games: float = 1.0
) -> dict[str, Any]:
    """Pairing rows -> the round-robin report: ratings, standard errors, pair table, cycles."""
    pairs = pairs_from_rows(rows)
    fit = fit_bradley_terry(pairs, order=order, prior_games=prior_games)
    cycles = significant_cycles(pairs)
    table = []
    for p in pairs:
        lo, hi = p.wilson()
        table.append(
            {"a": p.a, "b": p.b, "games": p.games, "wins": p.wins, "losses": p.losses,
             "draws": p.draws, "score": p.score, "wilson_lo": lo, "wilson_hi": hi}
        )  # fmt: skip
    return {**fit, "pairs": table, "cycles": [list(c) for c in cycles]}


def rating_steps(elo_by_gen: dict[int, float], step: int) -> list[dict[str, Any]]:
    """For each checkpoint ``g`` that has a checkpoint ``g + step``: the rating gain."""
    gens = sorted(elo_by_gen)
    return [
        {"from": g, "to": g + step, "gain": elo_by_gen[g + step] - elo_by_gen[g]}
        for g in gens
        if g + step in elo_by_gen
    ]


# ------------------------------------------------------------------------------------- rules


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def lag_scores(rows: list[dict[str, Any]]) -> dict[int, float]:
    """Generation -> the lag gate's score, for the generations that have one."""
    return {r["gen"]: r["gate"]["lag"]["score"] for r in rows if "gate" in r}


def evaluate_rules(
    rules: dict[str, Any],
    lag: int,
    total_generations: int,
    rows: list[dict[str, Any]],
    ratings: dict[str, Any] | None,
    stall_min_value_std: float,
) -> dict[str, Any]:
    """Judge a run by the pre-registered rules (``[rules]`` of the run config; the numbers are
    data, the logic is here). Clauses that lack their inputs are ``pending``."""
    clauses: dict[str, dict[str, Any]] = {}
    gens_done = max((r["gen"] for r in rows), default=0)
    scores = lag_scores(rows)

    early_gen = rules["early_gen"]
    early = [s for g, s in scores.items() if lag < g <= early_gen]
    if gens_done < early_gen or not early:
        clauses["early"] = {"status": "pending", "needs": f"generation {early_gen}"}
    else:
        v = _mean(early)
        clauses["early"] = {
            "status": "pass" if v >= rules["early_min_mean_lag_score"] else "fail",
            "value": v,
            "threshold": rules["early_min_mean_lag_score"],
            "generations": len(early),
        }

    last_q = [s for g, s in scores.items() if g > 0.75 * total_generations]
    if gens_done < total_generations or not last_q:
        clauses["late_lag"] = {"status": "pending", "needs": f"generation {total_generations}"}
    else:
        v = _mean(last_q)
        clauses["late_lag"] = {
            "status": "pass" if v >= rules["last_quartile_min_lag_score"] else "fail",
            "value": v,
            "threshold": rules["last_quartile_min_lag_score"],
        }

    steps: list[dict[str, Any]] = []
    plateau_from: int | None = None
    if ratings is None:
        for k in ("rating_grows", "final_vs_half", "no_cycles"):
            clauses[k] = {"status": "pending", "needs": "the round robin"}
    else:
        elo = {int(n[3:]): e for n, e in zip(ratings["players"], ratings["elo"], strict=True)}
        first = min(g for g in elo if g > 0) if any(g > 0 for g in elo) else 0
        chain = {g: e for g, e in elo.items() if g >= first}
        steps = rating_steps(chain, rules["rating_step"])
        bad = [s for s in steps if s["gain"] <= 0]
        plateau_from = bad[0]["from"] if bad else None
        clauses["rating_grows"] = {
            "status": "pass" if steps and not bad else "fail",
            "steps": steps,
            "first_non_increasing_step": plateau_from,
        }
        final, half = f"gen{total_generations}", f"gen{total_generations // 2}"
        direct = next(
            (p for p in ratings["pairs"] if {p["a"], p["b"]} == {final, half}),
            None,
        )
        if direct is None:
            clauses["final_vs_half"] = {"status": "pending", "needs": f"{final} vs {half}"}
        else:
            score = direct["score"] if direct["a"] == final else 1 - direct["score"]
            lo = direct["wilson_lo"] if direct["a"] == final else 1 - direct["wilson_hi"]
            ok = score >= rules["final_vs_half_min_score"] and lo > 0.5
            clauses["final_vs_half"] = {
                "status": "pass" if ok else "fail",
                "score": score,
                "wilson_lo": lo,
                "threshold": rules["final_vs_half_min_score"],
            }
        cycles = ratings["cycles"]
        clauses["no_cycles"] = {
            "status": "pass" if len(cycles) <= rules["max_significant_cycles"] else "fail",
            "cycles": cycles,
            "threshold": rules["max_significant_cycles"],
        }

    recent = rows[-10:]
    capped = sum(r["selfplay"].get("games_capped", 0) for r in recent)
    started = sum(r["selfplay"].get("games_started", 0) for r in recent)
    value_std = [r["val_after"]["value_std"] for r in recent if "val_after" in r]
    health_ok = (
        started > 0
        and capped / started <= rules["max_capped_rate"]
        and all(v >= stall_min_value_std for v in value_std)
    )
    black = _mean(
        [r["selfplay"]["black_wins"] / max(1, r["selfplay"]["games_finished"]) for r in recent]
    )
    ece = _mean([r["diagnostics"]["calibration"]["ece"] for r in recent if "diagnostics" in r])
    warnings = []
    if black is not None and not 0.35 <= black <= 0.65:
        warnings.append(f"first-player win rate {black:.2f} outside [0.35, 0.65]")
    if ece is not None and ece > 0.15:
        warnings.append(f"value calibration ECE {ece:.2f} above 0.15")
    clauses["health"] = {
        "status": "pass" if health_ok else ("fail" if started > 0 else "pending"),
        "capped_rate": capped / started if started else None,
        "min_value_std": min(value_std) if value_std else None,
        "warnings": warnings,
    }

    by = {k: v["status"] for k, v in clauses.items()}
    if by["early"] == "fail":
        verdict = "KILL_EARLY"
    elif "pending" in by.values():
        verdict = "PENDING"
    elif by["no_cycles"] == "fail" or by["health"] == "fail":
        verdict = "FAIL"
    elif all(v == "pass" for v in by.values()):
        verdict = "PASS"
    elif steps and steps[0]["gain"] <= 0:
        verdict = "FAIL"
    else:
        verdict = "PLATEAU"
    return {"verdict": verdict, "plateau_from": plateau_from, "clauses": clauses}
