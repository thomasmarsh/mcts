"""Keep the tuner domain on public, typed module seams."""

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "tuner_cli"


def _modules() -> tuple[Path, ...]:
    return tuple(sorted(SOURCE.glob("*.py")))


def test_production_imports_do_not_reach_private_module_names() -> None:
    violations: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(f"{path.name}:{node.lineno}: {alias.name}")
    assert not violations, "cross-module private imports:\n" + "\n".join(violations)


def test_production_source_has_no_type_escape_comments() -> None:
    violations = [
        f"{path.name}:{line_number}"
        for path in _modules()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "type: ignore" in line
    ]
    assert not violations, "type escapes:\n" + "\n".join(violations)
