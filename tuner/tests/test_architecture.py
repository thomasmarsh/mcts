"""Keep the tuner domain on public, typed module seams."""

import ast
import re
from pathlib import Path

_SRC = Path(__file__).parents[1] / "src"
SOURCE = _SRC / "tuner_cli"
# The projection package is held to the same public-seam, annotation, and
# line-count discipline as the core tuner modules.
PROJECTION_SOURCE = _SRC / "tuner_projection"

# Roughly this many logical statements is the hard split point for a single
# function (see the session hardening contract). Nested definitions are counted
# against their own budget, not their enclosing function's.
MAX_LOGICAL_LINES = 40


def _modules() -> tuple[Path, ...]:
    return tuple(sorted([*SOURCE.glob("*.py"), *PROJECTION_SOURCE.glob("*.py")]))


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _qualified_names(tree: ast.Module) -> dict[int, str]:
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qualified = f"{prefix}{child.name}"
                names[id(child)] = qualified
                visit(child, f"{qualified}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def _logical_lines(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    def count(node: ast.stmt) -> int:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return 0
        total = 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                total += count(child)
        return total

    return sum(count(statement) for statement in func.body)


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


def test_production_source_has_no_untyped_dictionary_or_any_escapes() -> None:
    pattern = re.compile(r"\bdict\[\s*str\s*,\s*object\s*\]|(?<![\w.])Any\b|\bcast\(")
    violations: list[str] = []
    for path in _modules():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                violations.append(f"{path.name}:{line_number}: {line.strip()}")
    assert not violations, "untyped dictionary / Any / cast escapes:\n" + "\n".join(violations)


def test_every_production_function_is_fully_annotated() -> None:
    violations: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = _qualified_names(tree)
        for func in _functions(tree):
            qualified = names.get(id(func), func.name)
            if func.returns is None:
                violations.append(f"{path.name}: {qualified} lacks a return annotation")
            args = func.args
            positional = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if args.vararg is not None:
                positional.append(args.vararg)
            if args.kwarg is not None:
                positional.append(args.kwarg)
            for arg in positional:
                if arg.arg in {"self", "cls"}:
                    continue
                if arg.annotation is None:
                    violations.append(f"{path.name}: {qualified}({arg.arg}) lacks an annotation")
    assert not violations, "unannotated production definitions:\n" + "\n".join(violations)


def test_continuation_advance_one_holds_no_scheduling_policy() -> None:
    tree = ast.parse((SOURCE / "continuation.py").read_text(encoding="utf-8"))
    advance = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "advance_one"
    )
    assert [type(statement) for statement in advance.body] == [ast.Match], (
        "advance_one must be a single match on the allocation decision, with no scheduling branches"
    )


def test_no_production_function_exceeds_the_logical_line_gate() -> None:
    violations: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = _qualified_names(tree)
        for func in _functions(tree):
            length = _logical_lines(func)
            if length > MAX_LOGICAL_LINES:
                qualified = names.get(id(func), func.name)
                violations.append(f"{path.name}: {qualified} is {length} logical lines")
    assert not violations, f"functions over {MAX_LOGICAL_LINES} logical lines:\n" + "\n".join(
        violations
    )
