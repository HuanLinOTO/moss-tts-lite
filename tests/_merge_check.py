"""Consolidation checker for the merged test suite (one line of output).

Asserts that every ``test_*`` / ``phase*`` function from the 20 original files
still exists after the 8-file consolidation, and that the originals are gone.

The expected counts are stated as the original-file mapping (not re-derived
from the sources, which are deleted by the merge commit), so this script keeps
working on a clean checkout.

Run:  python3 tests/_merge_check.py
"""

from __future__ import annotations

import ast
import os
import sys

#: merged file -> the original files it was assembled from (source: n_funcs).
#: n_funcs counts ``def test_*`` + ``def phase*`` at any nesting depth.
MAPPING = {
    "test_text.py": {
        "test_bpe.py": 3,                 # test_tiny_bpe_merges, test_roundtrip_samples, test_parity_hf
        "test_normalizer.py": 2,          # test_builtin_cases, test_parity_with_reference
        "test_prompt.py": 4,              # test_template_rendering, test_tensor_contract,
                                          #   test_normalizer_applied, test_parity_reference_processor
        "test_prompt_continuation.py": 3,  # test_golden_parity, test_delay_pattern_helper,
                                          #   test_argument_guards
    },
    "test_core.py": {
        "test_model.py": 0,               # entry point is main() (checked as a whole file)
        "test_sampling.py": 0,
        "test_st_loader.py": 4,
    },
    "test_generation.py": {
        "test_generate.py": 0,
        "test_fast_watchdog.py": 4,       # phase_b, phase_long, phase_pause, phase_a
    },
    "test_audio.py": {
        "test_codec.py": 8,
    },
    "test_parity.py": {
        "test_golden_parity.py": 2,
        "test_tts_parity.py": 2,          # phase_a, phase_b
        "test_e2e.py": 0,
    },
    "test_quant.py": {
        "test_gptq.py": 9,                # phase_g1..phase_g9
        "test_fast_m4.py": 1,             # phase_a
    },
    "test_fast.py": {
        "test_fast.py": 1,                # phase0
        "test_fast_native.py": 3,         # phase_a, phase_b, phase_c
    },
    "test_release.py": {
        "test_dep_purity.py": 1,
        "test_export.py": 6,              # phase_e1..phase_e6
        "test_vram_budget.py": 2,         # phase_0_sizing, phase_1_peaks
    },
    # Added after the consolidation (v1.1.0 CLI default flip), not part of the
    # original 20 files: help rendering / flag matrix / GPU path smoke.  Its
    # count is not asserted against an original -- the file is a new gate.
    "test_cli.py": {
        "test_cli_defaults.py": 4,        # test_help_renders, test_flag_matrix,
                                          #   test_gpu_paths, test_gpu_quant_default
    },
}

#: files that must survive the consolidation untouched
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
            # scope note: this file is a *new* gate, not a merge target, so it
            # only has to exist and be non-trivial
            if not names:
                problems.append("test_cli.py: no test_* functions")
            continue
        if len(names) != n_want:
            problems.append(f"{out_name}: {len(names)} test/phase funcs, want {n_want}")
        # no original may still be present.  test_fast.py is the one name that
        # collides with its merged output: there the original *is* the file
        # that was rewritten in place, so it is covered by the count above.
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
