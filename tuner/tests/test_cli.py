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


_CLI_BASE = [
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


def test_space_control_flags_assemble_into_constraints() -> None:
    parser = build_parser()
    from tuner_cli.constraints import encode_constraints

    assert _options(parser.parse_args(_CLI_BASE)).constraints == ()
    options = _options(
        parser.parse_args(
            _CLI_BASE
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
    assert encode_constraints(options.constraints) == [
        {
            "set": {
                "c": {"range": [1.2, 1.8]},
                "q_init": {"choices": ["Zero", "Infinity"]},
                "schedule": {"fix": "threshold"},
            }
        }
    ]
    with pytest.raises(ValueError):
        _options(parser.parse_args(_CLI_BASE + ["--fix", "bogus"]))
    with pytest.raises(ValueError):
        _options(parser.parse_args(_CLI_BASE + ["--fix", "c=1", "--param-range", "c=1,2"]))


def test_constraint_flag_carries_the_full_wire_form() -> None:
    parser = build_parser()
    from tuner_cli.constraints import encode_constraints

    options = _options(
        parser.parse_args(
            _CLI_BASE
            + [
                "--constraint",
                '{"set": {"c": {"range": [1.2, 1.8]}}}',
                "--constraint",
                '{"select": {"choices": ["ucb1", "rave"]}}',
            ]
        )
    )
    assert encode_constraints(options.constraints) == [
        {"set": {"c": {"range": [1.2, 1.8]}}},
        {"set": {"select": {"choices": ["ucb1", "rave"]}}},
    ]
    # A `when`-predicated constraint now flows through end to end.
    predicated = _options(
        parser.parse_args(
            _CLI_BASE
            + ["--constraint", '{"when": {"select": ["ucb1"]}, "set": {"c": {"fix": 1.4}}}']
        )
    )
    assert predicated.constraints[0].when == (("select", ("ucb1",)),)
    # `--exclude-family` is threaded raw for schema-time folding.
    excluded = _options(parser.parse_args(_CLI_BASE + ["--exclude-family", "rave"]))
    assert excluded.exclude_family == ("rave",)
