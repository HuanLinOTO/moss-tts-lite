"""Consolidation checker for the merged test suite (one line of output)."""

from __future__ import annotations

import ast
import os
import sys

MAPPING = {
    "test_text.py": {
        "test_bpe.py": 3,
        "test_normalizer.py": 2,
        "test_prompt.py": 4,

        "test_prompt_continuation.py": 3,

    },
    "test_core.py": {
        "test_model.py": 0,
        "test_sampling.py": 0,
        "test_st_loader.py": 4,
    },
    "test_generation.py": {
        "test_generate.py": 0,
        "test_fast_watchdog.py": 4,
    },
    "test_audio.py": {
        "test_codec.py": 8,
    },
    "test_parity.py": {
        "test_golden_parity.py": 2,
        "test_tts_parity.py": 2,
        "test_e2e.py": 0,
    },
    "test_quant.py": {
        "test_gptq.py": 9,
        "test_fast_m4.py": 1,
    },
    "test_fast.py": {
        "test_fast.py": 1,
        "test_fast_native.py": 3,
    },
    "test_release.py": {
        "test_dep_purity.py": 1,
        "test_export.py": 6,
        "test_vram_budget.py": 2,
    },

    "test_cli.py": {
        "test_cli_defaults.py": 4,

    },
}

KEPT = {"__init__.py", "_mini_bpe.py", "_mini_loader.py"}

def counted(path: str) -> list[str]:
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    return sorted(n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and (n.name.startswith("test_") or n.name.startswith("phase")))

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    problems: list[str] = []

    total_want = 0
    total_got = 0
    merged_want = sum(sum(s.values()) for f, s in MAPPING.items()
                      if f != "test_cli.py")
    merged_got = sum(len(counted(os.path.join(here, f)))
                     for f in MAPPING if f != "test_cli.py")
    for out_name, sources in sorted(MAPPING.items()):
        n_want = sum(sources.values())
        names = counted(os.path.join(here, out_name))
        total_want += n_want
        total_got += len(names)
        if out_name == "test_cli.py":

            if not names:
                problems.append("test_cli.py: no test_* functions")
            continue
        if len(names) != n_want:
            problems.append(f"{out_name}: {len(names)} test/phase funcs, want {n_want}")

        for src in sources:
            if src != out_name and os.path.exists(os.path.join(here, src)):
                problems.append(f"{out_name}: original {src} still present")

    for name in sorted(KEPT):
        if not os.path.exists(os.path.join(here, name)):
            problems.append(f"missing preserved file {name}")

    left = sorted(p for p in os.listdir(here) if p.startswith("test_"))
    if len(left) != len(MAPPING):
        problems.append(f"test_*.py count {len(left)}, want {len(MAPPING)}: {left}")
    ok = not problems and merged_got == merged_want == 55
    print(f"_merge_check: {'PASS' if ok else 'FAIL'} "
          f"({merged_got}/{merged_want} test_*/phase* functions across "
          f"{len(MAPPING) - 1} merged files; "
          f"{len(counted(os.path.join(here, 'test_cli.py')))} in the new "
          f"test_cli.py)"
          + ("" if ok else "; " + "; ".join(problems)))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
