"""Hand-verifiable ledger arithmetic over strict evidence payloads."""

from __future__ import annotations

from tuner_cli.compute import fold_ledger
from tuner_cli.domain import ComputeLedger, PhaseCompute
from tuner_cli.event_payloads import (
    PairCompletedPayload,
    PairFailedPayload,
    PairIdentity,
    PairStartedPayload,
    RunInterruptedPayload,
)
from tuner_cli.evidence import EvidenceEvent


def _identity(phase: str, pair_id: str = "pair-1", candidate: str = "cand-1") -> PairIdentity:
    return PairIdentity(phase, candidate, "task-1", pair_id, "opponent-1", 1000)


def _game(seq: int, iterations: int = 7, elapsed: int = 13) -> dict[str, object]:
    return {
        "game_id": f"game-{seq}",
        "candidate_side": "first" if seq == 1 else "second",
        "outcome": "candidate_win" if seq == 1 else "draw",
        "derived_seed": 1,
        "round": 1,
        "seq": seq,
        "trace_game_seq": None,
        "plies": 2,
        "elapsed_ms": elapsed,
        "candidate_metrics": {
            "iterations_total": iterations,
            "iterations_first_half": 1,
            "move_time_ms": 1,
        },
        "opponent_metrics": {
            "iterations_total": iterations + 1,
            "iterations_first_half": 1,
            "move_time_ms": 1,
        },
        "raw_record": "{}",
    }


def _event(payload: object) -> EvidenceEvent:
    import typing

    from tuner_cli.event_payloads import EventPayload

    typed = typing.cast(EventPayload, payload)
    return EvidenceEvent(1, typed.event_type, typed)


def test_empty_evidence_yields_zeroed_ledger() -> None:
    assert fold_ledger([]) == ComputeLedger(PhaseCompute(), PhaseCompute())


def test_completed_pair_counts_attempt_games_iterations_and_time() -> None:
    events = [
        _event(PairStartedPayload(_identity("tuning"), task_seed=5)),
        _event(
            PairCompletedPayload(
                _identity("tuning"),
                (_game(1), _game(2, iterations=7, elapsed=13)),
                pair_utility=1.0,
            )
        ),
    ]
    ledger = fold_ledger(events)
    # One attempt, one completed pair, two games; each game contributes
    # candidate 7 + opponent 8 iterations and 13 ms.
    assert ledger.tuning == PhaseCompute(
        pair_attempts=1,
        completed_pairs=1,
        failed_attempts=0,
        censored_attempts=0,
        physical_games=2,
        search_iterations=(7 + 8) * 2,
        wall_time_ms=13 * 2,
    )
    assert ledger.validation == PhaseCompute()


def test_failed_pair_counts_attempt_and_complete_partial_games_only() -> None:
    partial = '{"type":"configured_match_result","seq":1,"elapsed_ms":9,"metrics":true}'
    events = [
        _event(PairStartedPayload(_identity("tuning"), task_seed=5)),
        _event(
            PairFailedPayload(
                _identity("tuning"),
                kind="pair_output",
                command=("game", "play"),
                returncode=1,
                stderr="boom",
                stdout="noise\n" + partial,
                partial_output=(partial,),
            )
        ),
    ]
    ledger = fold_ledger(events)
    # The one complete partial game record contributes its elapsed time but no
    # strategy metrics, so it adds one physical game with zero iterations.
    assert ledger.tuning == PhaseCompute(
        pair_attempts=1,
        completed_pairs=0,
        failed_attempts=1,
        censored_attempts=0,
        physical_games=1,
        search_iterations=0,
        wall_time_ms=9,
    )


def test_interrupted_start_is_censored_and_retry_adds_an_attempt() -> None:
    events = [
        _event(PairStartedPayload(_identity("validation"), task_seed=5)),
        _event(RunInterruptedPayload("pair_execution", "pair-1")),
        _event(PairStartedPayload(_identity("validation"), task_seed=5)),
        _event(
            PairCompletedPayload(
                _identity("validation"),
                (_game(1, iterations=1, elapsed=2), _game(2, iterations=1, elapsed=2)),
                0.5,
            )
        ),
    ]
    ledger = fold_ledger(events)
    # Two starts (the original plus the retry), one completion, one censored.
    assert ledger.validation == PhaseCompute(
        pair_attempts=2,
        completed_pairs=1,
        failed_attempts=0,
        censored_attempts=1,
        physical_games=2,
        search_iterations=(1 + 2) * 2,
        wall_time_ms=2 * 2,
    )


def test_phases_are_separated() -> None:
    events = [
        _event(PairStartedPayload(_identity("tuning", "pair-t"), task_seed=5)),
        _event(
            PairCompletedPayload(
                _identity("tuning", "pair-t"), (_game(1), _game(2, iterations=2, elapsed=3)), 1.0
            )
        ),
        _event(PairStartedPayload(_identity("validation", "pair-v"), task_seed=5)),
    ]
    ledger = fold_ledger(events)
    assert ledger.tuning.pair_attempts == 1
    assert ledger.tuning.completed_pairs == 1
    assert ledger.validation.pair_attempts == 1
    assert ledger.validation.censored_attempts == 1
    assert ledger.validation.completed_pairs == 0
