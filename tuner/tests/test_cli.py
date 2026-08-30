from __future__ import annotations

import pytest

from tuner_cli.__main__ import build_parser


def test_parser_requires_explicit_game_binary_and_run_directory() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-binary", "game"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", "run"])
    args = parser.parse_args(["--game-binary", "game", "--run-dir", "run"])
    assert str(args.game_binary) == "game"
    assert str(args.run_dir) == "run"
