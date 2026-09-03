from __future__ import annotations

import pytest

from tuner_cli.__main__ import _options, build_parser
from tuner_cli.domain import SearchEffort


def test_parser_requires_explicit_scientific_inputs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-binary", "game"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-binary", "game", "--run-dir", "run"])
    args = parser.parse_args(
        [
            "--game-binary",
            "game",
            "--objective-file",
            "objective.json",
            "--run-dir",
            "run",
            "--task-seed",
            "9",
            "--tuning-pair-budget",
            "24",
            "--validation-pair-budget",
            "6",
            "--production-validation-pairs",
            "3",
        ]
    )
    assert str(args.game_binary) == "game"
    assert str(args.run_dir) == "run"
    assert args.tuning_pair_budget == 24
    assert args.validation_pair_budget == 6
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--game-binary",
                "game",
                "--objective-file",
                "objective.json",
                "--run-dir",
                "run",
                "--task-seed",
                "9",
                "--production-validation-pairs",
                "3",
            ]
        )
    assert parser.parse_args(
        [
            "--game-binary",
            "game",
            "--objective-file",
            "objective.json",
            "--run-dir",
            "run",
            "--task-seed",
            "9",
            "--tuning-pair-budget",
            "24",
            "--validation-pair-budget",
            "6",
            "--production-validation-pairs",
            "3",
            "--resume",
        ]
    ).resume
    assert (
        parser.parse_args(
            [
                "--game-binary",
                "game",
                "--objective-file",
                "objective.json",
                "--run-dir",
                "run",
                "--task-seed",
                "9",
                "--tuning-pair-budget",
                "24",
                "--validation-pair-budget",
                "6",
                "--production-validation-pairs",
                "3",
                "--evaluator-workers",
                "2",
            ]
        ).evaluator_workers
        == 2
    )


def test_effort_flags_resolve_independently_and_are_exclusive() -> None:
    parser = build_parser()
    base = [
        "--game-binary",
        "game",
        "--objective-file",
        "objective.json",
        "--run-dir",
        "run",
        "--task-seed",
        "9",
        "--tuning-pair-budget",
        "24",
        "--validation-pair-budget",
        "6",
        "--production-validation-pairs",
        "3",
    ]
    assert _options(parser.parse_args(base)).tuning_effort == SearchEffort("iterations", 1_000)
    options = _options(
        parser.parse_args(
            base
            + [
                "--tuning-max-time-ms",
                "10",
                "--validation-max-iterations",
                "20",
                "--production-max-time-ms",
                "30",
            ]
        )
    )
    assert (options.tuning_effort, options.validation_effort, options.production_effort) == (
        SearchEffort("time_ms", 10),
        SearchEffort("iterations", 20),
        SearchEffort("time_ms", 30),
    )
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--tuning-max-iterations", "1", "--tuning-max-time-ms", "1"])


def test_shadow_halving_spare_margin_defaults_zero_and_threads_through() -> None:
    parser = build_parser()
    base = [
        "--game-binary",
        "game",
        "--objective-file",
        "objective.json",
        "--run-dir",
        "run",
        "--task-seed",
        "9",
        "--tuning-pair-budget",
        "24",
        "--validation-pair-budget",
        "6",
        "--production-validation-pairs",
        "3",
    ]
    assert _options(parser.parse_args(base)).shadow_halving_spare_margin == 0.0
    options = _options(
        parser.parse_args(
            base + ["--shadow-policy", "successive_halving", "--shadow-halving-spare-margin", "0.1"]
        )
    )
    assert options.shadow_halving_spare_margin == 0.1


def test_space_override_flags_assemble_into_the_overrides_map() -> None:
    parser = build_parser()
    base = [
        "--game-binary",
        "game",
        "--objective-file",
        "objective.json",
        "--run-dir",
        "run",
        "--task-seed",
        "9",
        "--tuning-pair-budget",
        "24",
        "--validation-pair-budget",
        "6",
        "--production-validation-pairs",
        "3",
    ]
    assert _options(parser.parse_args(base)).space_overrides == {}
    options = _options(
        parser.parse_args(
            base
            + [
                "--fix",
                "schedule=threshold",
                "--param-range",
                "c=1.2,1.8",
                "--param-choices",
                "q_init=Zero,Infinity",
            ]
        )
    )
    from tuner_cli.space_overrides import encode_space_overrides

    assert encode_space_overrides(options.space_overrides) == {
        "c": {"range": [1.2, 1.8]},
        "q_init": {"choices": ["Zero", "Infinity"]},
        "schedule": {"fix": "threshold"},
    }
    with pytest.raises(ValueError):
        _options(parser.parse_args(base + ["--fix", "bogus"]))
    with pytest.raises(ValueError):
        _options(parser.parse_args(base + ["--fix", "c=1", "--param-range", "c=1,2"]))


def test_constraint_flag_lowers_onto_the_space_overrides_map() -> None:
    parser = build_parser()
    base = [
        "--game-binary",
        "game",
        "--objective-file",
        "objective.json",
        "--run-dir",
        "run",
        "--task-seed",
        "9",
        "--tuning-pair-budget",
        "24",
        "--validation-pair-budget",
        "6",
        "--production-validation-pairs",
        "3",
    ]
    from tuner_cli.space_overrides import encode_space_overrides

    options = _options(
        parser.parse_args(
            base
            + [
                "--constraint",
                '{"set": {"c": {"range": [1.2, 1.8]}}}',
                "--constraint",
                '{"select": {"choices": ["ucb1", "rave"]}}',
            ]
        )
    )
    assert encode_space_overrides(options.space_overrides) == {
        "c": {"range": [1.2, 1.8]},
        "select": {"choices": ["ucb1", "rave"]},
    }
    # A `when`-predicated constraint is rejected until the full cutover wires it.
    with pytest.raises(ValueError):
        _options(
            parser.parse_args(
                base
                + ["--constraint", '{"when": {"select": ["ucb1"]}, "set": {"c": {"fix": 1.4}}}']
            )
        )
    # The same parameter cannot be constrained twice across the two surfaces.
    with pytest.raises(ValueError):
        _options(
            parser.parse_args(
                base + ["--param-range", "c=1,2", "--constraint", '{"set": {"c": {"fix": 1.5}}}']
            )
        )
