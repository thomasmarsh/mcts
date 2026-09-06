"""Deterministic weighted task corpora and selected-prefix policy."""

from __future__ import annotations

from collections import Counter

from .domain import OpponentPanel, Phase, TaskCorpus, TaskPrefix
from .identity import task_case, task_corpus, task_prefix


def weighted_schedule(panel: OpponentPanel, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("task count must be positive")
    assigned = [0] * len(panel.opponents)
    schedule: list[int] = []
    for ordinal in range(count):
        total = ordinal + 1
        chosen = max(
            range(len(panel.opponents)),
            key=lambda index: (
                panel.opponents[index].weight * total / panel.total_weight - assigned[index],
                -index,
            ),
        )
        assigned[chosen] += 1
        schedule.append(chosen)
    return tuple(schedule)


def validate_cycle_endpoint(panel: OpponentPanel, count: int, label: str) -> None:
    if count <= 0 or count % panel.total_weight:
        raise ValueError(
            f"{label} must be positive and divisible by panel total weight {panel.total_weight}"
        )


def build_corpus(
    phase: Phase, count: int, task_seed: int, panel: OpponentPanel, game_config_fingerprint: str
) -> TaskCorpus:
    schedule = weighted_schedule(panel, count)
    cases = tuple(
        task_case(phase, ordinal, task_seed, panel.opponents[index], panel, game_config_fingerprint)
        for ordinal, index in enumerate(schedule)
    )
    return task_corpus(phase, cases, panel)


def selected_prefix(corpus: TaskCorpus, count: int) -> TaskPrefix:
    return task_prefix(corpus, count)


def tuning_blocks(corpus: TaskCorpus, panel: OpponentPanel) -> tuple[TaskPrefix, ...]:
    """Return cumulative prefixes ending at each complete weighted panel cycle."""
    return tuple(
        selected_prefix(corpus, length)
        for length in range(panel.total_weight, len(corpus.cases) + 1, panel.total_weight)
    )


def verify_weighted_corpus(corpus: TaskCorpus, panel: OpponentPanel) -> None:
    expected = weighted_schedule(panel, len(corpus.cases))
    actual = tuple(
        next(
            index
            for index, item in enumerate(panel.opponents)
            if item.opponent_id == case.opponent_id
        )
        for case in corpus.cases
    )
    if actual != expected:
        raise ValueError("task corpus violates the weighted-fair schedule")
    counts = Counter(case.opponent_id for case in corpus.cases)
    if len(corpus.cases) % panel.total_weight == 0 and any(
        counts[item.opponent_id] != item.weight * len(corpus.cases) // panel.total_weight
        for item in panel.opponents
    ):
        raise ValueError("complete corpus does not have exact panel weights")
