"""Dependency-purity gate for moss_tts_lite (COORD hard rule #2).

Statically scans every .py under moss_tts_lite/ (AST parse, function-level imports
included via ast.walk) and asserts every import resolves to:
  - the stdlib            (sys.stdlib_module_names)
  - {torch, numpy, soundfile, yaml}
  - the moss_tts_lite package itself (absolute or relative imports)

This test file itself must stay stdlib-only.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1] / "moss_tts_lite"
ALLOWED_THIRD_PARTY = {"torch", "numpy", "soundfile", "yaml"}


def _collect_imports(path: Path) -> list[tuple[str, int, str]]:
    """Return (top_level_module, lineno, shown_name) for every Import /
    ImportFrom node anywhere in the file (module body, functions, try/except)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # relative import: inside the package by definition
                found.append(("moss_tts_lite", node.lineno, "." * node.level + (node.module or "")))
            else:
                mod = node.module or ""
                found.append((mod.split(".")[0], node.lineno, mod))
    return found


def test_dep_purity():
    stdlib = sys.stdlib_module_names
    files = sorted(PKG_ROOT.rglob("*.py"))
    assert files, f"no .py files found under {PKG_ROOT}"

    violations: list[str] = []
    seen: dict[str, set[str]] = {}   # top-level module -> files importing it

    for f in files:
        rel = f.relative_to(PKG_ROOT.parent).as_posix()
        try:
            imports = _collect_imports(f)
        except SyntaxError as e:  # pragma: no cover
            raise AssertionError(f"{rel}: unparsable: {e}") from e
        for top, lineno, name in imports:
            seen.setdefault(top, set()).add(rel)
            if top not in stdlib and top not in ALLOWED_THIRD_PARTY and top != "moss_tts_lite":
                violations.append(f"{rel}:{lineno}: import {name!r} (top={top!r})")

    assert not violations, (
        "moss_tts_lite imports outside torch/numpy/soundfile/yaml + stdlib:\n"
        + "\n".join(violations))

    third = sorted(t for t in seen if t in ALLOWED_THIRD_PARTY)
    own = sorted(t for t in seen if t == "moss_tts_lite")
    other = sorted(t for t in seen if t not in stdlib
                   and t not in ALLOWED_THIRD_PARTY and t != "moss_tts_lite")
    std_used = sorted(t for t in seen if t in stdlib)
    print(f"  scanned {len(files)} files under {PKG_ROOT.name}/")
    print(f"  third-party: {third}")
    print(f"  own package: {own}")
    print(f"  stdlib used ({len(std_used)}): {std_used}")
    assert not other, f"unexpected modules: {other}"


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_dep_purity()
    print("ALL TESTS PASSED")
