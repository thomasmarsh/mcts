"""Pure evidence-to-ledger calculation with no policy or JSON presentation."""

from __future__ import annotations

import json

from .codec import is_json_object
from .domain import ComputeLedger, PhaseCompute
from .event_payloads import (
    DiagnosticPairCompletedPayload,
    DiagnosticPairFailedPayload,
    DiagnosticPairStartedPayload,
    PairCompletedPayload,
    PairFailedPayload,
    PairStartedPayload,
)
from .evidence import EvidenceEvent


class LedgerBuilder:
    """Incremental evidence-to-ledger fold for one replay pass.

    Every ``pair_started`` event adds one attempt for its phase, including
    retried starts of the same pair after an interruption. Censored attempts
    are starts that never reached a completed or failed terminal event, so
    they are derived as ``attempts - completed - failed`` at read time.
    """

    def __init__(self) -> None:
        self._tuning = _PhaseAccumulator()
        self._validation = _PhaseAccumulator()
        self._diagnostic = _PhaseAccumulator()

    def apply(self, event: EvidenceEvent) -> None:
        payload = event.payload
        match payload:
            case PairStartedPayload():
                self._phase(payload.identity.phase).attempts += 1
            case PairCompletedPayload():
                phase = self._phase(payload.identity.phase)
                phase.completed_pairs += 1
                for game in payload.games:
                    phase.physical_games += 1
                    phase.search_iterations += game_iterations(game)
                    phase.wall_time_ms += game_elapsed(game)
            case PairFailedPayload():
                phase = self._phase(payload.identity.phase)
                phase.failed_attempts += 1
                for raw in payload.partial_output:
                    if (game := parse_partial_game(raw)) is not None:
                        phase.physical_games += 1
                        phase.search_iterations += game_iterations(game)
                        phase.wall_time_ms += game_elapsed(game)
            case DiagnosticPairStartedPayload():
                self._diagnostic.attempts += 1
            case DiagnosticPairCompletedPayload():
                self._diagnostic.completed_pairs += 1
                for game in payload.games:
                    self._diagnostic.physical_games += 1
                    self._diagnostic.search_iterations += game_iterations(game)
                    self._diagnostic.wall_time_ms += game_elapsed(game)
            case DiagnosticPairFailedPayload():
                self._diagnostic.failed_attempts += 1
            case _:
                pass

    def ledger(self) -> ComputeLedger:
        return ComputeLedger(
            self._tuning.compute(), self._validation.compute(), self._diagnostic.compute()
        )

    def _phase(self, phase: str) -> _PhaseAccumulator:
        return self._tuning if phase == "tuning" else self._validation


class _PhaseAccumulator:
    attempts: int
    completed_pairs: int
    failed_attempts: int
    physical_games: int
    search_iterations: int
    wall_time_ms: int

    def __init__(self) -> None:
        self.attempts = 0
        self.completed_pairs = 0
        self.failed_attempts = 0
        self.physical_games = 0
        self.search_iterations = 0
        self.wall_time_ms = 0

    def compute(self) -> PhaseCompute:
        return PhaseCompute(
            self.attempts,
            self.completed_pairs,
            self.failed_attempts,
            self.attempts - self.completed_pairs - self.failed_attempts,
            self.physical_games,
            self.search_iterations,
            self.wall_time_ms,
        )


def fold_ledger(events: list[EvidenceEvent]) -> ComputeLedger:
    """Derive one factual ledger from a complete evidence stream."""
    builder = LedgerBuilder()
    for event in events:
        builder.apply(event)
    return builder.ledger()


def game_iterations(game: object) -> int:
    """Candidate plus opponent ``iterations_total`` from one game record."""
    if not is_json_object(game):
        return 0
    candidate_value = game.get("candidate_metrics")
    opponent_value = game.get("opponent_metrics")
    if not isinstance(candidate_value, dict) or not isinstance(opponent_value, dict):
        return 0
    candidate_iter = candidate_value.get("iterations_total")
    opponent_iter = opponent_value.get("iterations_total")
    return (candidate_iter if isinstance(candidate_iter, int) else 0) + (
        opponent_iter if isinstance(opponent_iter, int) else 0
    )


def game_elapsed(game: object) -> int:
    """Recorded ``elapsed_ms`` from one game record."""
    if not is_json_object(game):
        return 0
    elapsed = game.get("elapsed_ms")
    return elapsed if isinstance(elapsed, int) else 0


def parse_partial_game(raw: str) -> object | None:
    """One complete ``configured_match_result`` record from failed-pair output."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if is_json_object(parsed) and parsed.get("type") == "configured_match_result":
        return parsed
    return None
