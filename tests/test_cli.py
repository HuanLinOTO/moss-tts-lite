"""CLI gates: the v1.1.0 default flip (fast is what a bare command line runs).

Until v1.1.0 the CLI default was the exact **eager** path and ``--fast`` was an
opt-in that enabled the CUDA-graph decoder.  The two decoders are bitwise
identical (M1), so the opt-in only ever cost speed: the default is now
``moss_tts_lite.fast`` and ``--fast`` is a no-op alias kept for the scripts and
docs that already pass it.  This file pins that contract.

Phases:
  A  help rendering (subprocess, no GPU): the usage line lists --eager, and the
     three path flags say what they now mean.
  B  flag matrix (no GPU, no weights): `cli.synthesize` is monkeypatched, so the
     kwargs the CLI would hand to the pipeline are asserted directly --
     default/--fast/--eager/--fast-native/--quant combinations, the one-line
     no-op notice, and the two rejected combinations.
  C  GPU smoke (flock, needs the bf16 base weights): a bare command line really
     decodes on the fast path, --eager really takes MossTTSModel.step, and
     --fast-native reports the native graph pool.
  D  GPU smoke for the quantized default (standalone export, if present):
     `--quant` with no fast flag stays on the fast path and prints no extra
     "--quant implies --fast" line.

Run:
  PYTHONPATH=. MOSS_TTS_ROOT=/root/MOSS-TTS python3 tests/test_cli.py

  # phase C is a GPU phase (bf16 base, ~17 GiB peak) -- share the card politely:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 tests/test_cli.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import contextlib

# `python3 tests/<this file>.py` straight from the repo root must import
# moss_tts_lite without an explicit PYTHONPATH.
LITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LITE_ROOT)

import numpy as np
import torch

from moss_tts_lite import cli
from moss_tts_lite.cli import _BUILTIN_DEFAULTS, main as cli_main

ROOT = os.environ.get("MOSS_TTS_ROOT", os.path.dirname(LITE_ROOT))
MODEL_DIR = os.environ.get("MOSS_TTS_MODEL_DIR",
                           os.path.join(ROOT, "models", "MOSS-TTS-v1.5"))
CODEC_DIR = os.environ.get("MOSS_AUDIO_MODEL_DIR",
                           os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer"))
OUT = os.path.join(LITE_ROOT, ".tmp", "cli_agent", "wav")

#: exact text of the compatibility notice (requirement: a *one-line* note)
NOOP_NOTE = "[--fast is now the default; this flag is a no-op]"
#: the pre-v1.1.0 notice; it must be gone now that fast is unconditional
OBSOLETE_NOTE = "--quant implies --fast"


# --------------------------------------------------------------------------- #
# Phase A -- help rendering (no GPU)                                          #
# --------------------------------------------------------------------------- #
def _help_text() -> str:
    proc = subprocess.run([sys.executable, "-m", "moss_tts_lite", "--help"],
                          cwd=LITE_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_help_renders() -> None:
    raw = _help_text()
    # argparse hard-wraps help strings at the terminal width, so compare on a
    # whitespace-normalized copy of the whole help text
    text = " ".join(raw.split())
    assert "python -m moss_tts_lite" in raw
    # the usage line advertises the new switch next to the no-op alias
    usage = text.split("positional arguments:")[0]
    assert "--eager" in usage and "--fast" in usage and "--fast-native" in usage
    # Each flag's help is anchored to its own option line (argparse prints
    # "--flag<spaces>help..."), which normalizes to "--flag help...".
    assert "--fast NO-OP (fast is already the default since v1.1.0)" in text, text[:200]
    assert "--eager reference slow path, for debugging" in text, text[:200]
    assert ("--fast-native whole-step CUDA graph tier "
            "(moss_tts_lite.fast_native, arm n2): FASTER than the default") in text
    assert "NOT bitwise-identical" in text
    print("  A: --help renders; usage lists --eager; --fast says NO-OP; "
          "--fast-native says 'faster, not bitwise' -> PASS")


# --------------------------------------------------------------------------- #
# Phase B -- flag matrix, no GPU (synthesize is monkeypatched)                #
# --------------------------------------------------------------------------- #
class _FakeRes:
    n_steps = 1
    finished = True
    audio_frames = None
    watchdog_triggered = False


def _fake_synthesize(text, output, **kw):
    """Stand-in for cli.synthesize: record the decode-path kwargs, write nothing."""
    calls.append(kw)
    kw["stats"].update({"steps": 1, "step_ms": [10.0], "prefill_ms": 5.0,
                        "l0": 66, "max_seq_len": 4226, "kv_mib": 566.0,
                        "max_graphs": None, "graph_pool_cap": None})
    return np.zeros(24000, dtype="float32"), 24000, _FakeRes(), "resident"


calls: list[dict] = []


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run main(argv) with the pipeline stubbed; return (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli_main(argv)
    return rc, out.getvalue(), err.getvalue()


def _run_expecting_exit(argv: list[str]) -> str:
    try:
        _run(argv)
    except SystemExit as exc:
        return str(exc)
    raise AssertionError(f"expected SystemExit for {argv}")


_ORIG_SYNTH = None


def test_flag_matrix() -> None:
    global _ORIG_SYNTH
    _ORIG_SYNTH = cli.synthesize
    cli.synthesize = _fake_synthesize
    ok = True
    try:
        # ---- 1) bare command line = fast, no notice of any kind ------------
        calls.clear()
        rc, out, err = _run(["hi", "-o", "out.wav"])
        assert rc == 0 and len(calls) == 1
        kw = calls[-1]
        good = (kw["fast"] is True and kw["fast_native"] is False
                and kw["quant"] is None and err == "")
        ok &= good
        print(f"  B1 bare (no fast flags)      -> fast={kw['fast']} "
              f"native={kw['fast_native']} stderr={err.strip()!r} "
              f"{'ok' if good else 'FAIL'}")
        assert "path=fast(cuda-graphs" in out and "path=eager" not in out

        # ---- 2) --fast is a no-op alias: same path + one notice line -------
        calls.clear()
        rc, out2, err2 = _run(["hi", "-o", "out.wav", "--fast"])
        kw = calls[-1]
        good = (kw["fast"] is True and kw["fast_native"] is False
                and err2.strip() == NOOP_NOTE and OBSOLETE_NOTE not in err2
                and "path=fast(cuda-graphs" in out2)
        ok &= good
        print(f"  B2 --fast (no-op alias)      -> fast={kw['fast']} "
              f"stderr={err2.strip()!r} {'ok' if good else 'FAIL'}")

        # ---- 3) --eager opts back into the reference path ------------------
        calls.clear()
        rc, out3, err3 = _run(["hi", "-o", "out.wav", "--eager"])
        kw = calls[-1]
        good = (kw["fast"] is False and kw["fast_native"] is False
                and err3 == "" and "path=eager(reference)" in out3)
        ok &= good
        print(f"  B3 --eager                   -> fast={kw['fast']} "
              f"path=eager printed={'path=eager(reference)' in out3} "
              f"{'ok' if good else 'FAIL'}")

        # ---- 4) --fast-native stays the faster non-bitwise tier ------------
        calls.clear()
        rc, out4, err4 = _run(["hi", "-o", "out.wav", "--fast-native"])
        kw = calls[-1]
        good = (kw["fast"] is True and kw["fast_native"] is True
                and "path=fast-native(n2" in out4)
        ok &= good
        print(f"  B4 --fast-native             -> fast={kw['fast']} "
              f"native={kw['fast_native']} {'ok' if good else 'FAIL'}")

        # ---- 5) --quant alone: still fast, and the obsolete notice is gone --
        calls.clear()
        rc, out5, err5 = _run(["hi", "-o", "out.wav", "--quant", "w4"])
        kw = calls[-1]
        good = (kw["fast"] is True and kw["quant"] == "w4"
                and kw["w4_group_size"] == 128
                and OBSOLETE_NOTE not in err5 and err5 == "")
        ok &= good
        print(f"  B5 --quant w4 (no fast flag) -> fast={kw['fast']} "
              f"quant={kw['quant']} stderr={err5.strip()!r} "
              f"{'ok' if good else 'FAIL'}")
        assert "path=fast(cuda-graphs" in out5

        # ---- 6) --fast together with --quant: no duplicate notice ----------
        calls.clear()
        rc, out6, err6 = _run(["hi", "-o", "out.wav", "--fast", "--quant", "w4"])
        kw = calls[-1]
        good = (kw["fast"] is True and err6.strip() == NOOP_NOTE
                and err6.count(NOOP_NOTE) == 1)
        ok &= good
        print(f"  B6 --fast --quant w4         -> notice printed exactly once "
              f"{'ok' if good else 'FAIL'}")

        # ---- 7) rejected combinations --------------------------------------
        e_eager_native = _run_expecting_exit(
            ["hi", "-o", "out.wav", "--eager", "--fast-native"])
        e_eager_quant = _run_expecting_exit(
            ["hi", "-o", "out.wav", "--eager", "--quant", "w4"])
        e_native_greedy = _run_expecting_exit(
            ["hi", "-o", "out.wav", "--fast-native", "--greedy"])
        good = ("mutually exclusive" in e_eager_native
                and "cannot be combined with --eager" in e_eager_quant
                and "cannot be combined with --greedy" in e_native_greedy)
        ok &= good
        print(f"  B7 --eager+--fast-native / --eager+--quant / native+greedy all "
              f"refused {'ok' if good else 'FAIL'}")

        # ---- 8) a standalone export must refuse --eager (offline guard) -----
        import tempfile
        from moss_tts_lite.export import FORMAT_TAG, META_FILE
        import json
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, META_FILE), "w", encoding="utf-8") as f:
                json.dump({"format": FORMAT_TAG, "presets": ["w1"]}, f)
            msg = _run_expecting_exit(
                ["hi", "-o", "out.wav", "--model-dir", tmp, "--eager"])
        good = "standalone quantized export" in msg and "--eager" in msg
        ok &= good
        print(f"  B8 standalone export + --eager refused "
              f"{'ok' if good else 'FAIL'}")

        # ---- 9) the built-in default itself is fast ------------------------
        good = _BUILTIN_DEFAULTS["fast"] is True
        ok &= good
        print(f"  B9 _BUILTIN_DEFAULTS['fast']={_BUILTIN_DEFAULTS['fast']} "
              f"{'ok' if good else 'FAIL'}")

        # ---- 10) --device cpu must not try CUDA graphs ---------------------
        calls.clear()
        rc, out10, err10 = _run(["hi", "-o", "out.wav", "--device", "cpu"])
        kw = calls[-1]
        good = (kw["fast"] is False and "--device cpu" in err10
                and "path=eager(reference)" in out10)
        ok &= good
        print(f"  B10 --device cpu -> fast={kw['fast']} (eager fallback) "
              f"{'ok' if good else 'FAIL'}")
        msg = _run_expecting_exit(
            ["hi", "-o", "out.wav", "--device", "cpu", "--quant", "w4"])
        good = "requires CUDA" in msg
        ok &= good
        print(f"  B11 --device cpu + --quant refused {'ok' if good else 'FAIL'}")
    finally:
        cli.synthesize = _ORIG_SYNTH
    assert ok, "flag matrix FAILED"
    print("  B: flag matrix -> PASS")


# --------------------------------------------------------------------------- #
# Phase C -- GPU smoke: the real pipeline picks the path the flags promise     #
# --------------------------------------------------------------------------- #
#: short input: the point is *which* decoder ran, not how long the utterance is
TEXT = "你好。"
BUDGET = 96


def _cli_notes(err: str) -> list[str]:
    """CLI's own stderr lines (torch warnings and the like are not ours)."""
    return [l for l in err.splitlines()
            if "moss_tts_lite]" in l or l.startswith("[--fast")]


def _cli_once(flags: list[str]) -> tuple[int, str, str, dict[str, int]]:
    """Run the real pipeline and count which top-level decoder entry fired.

    The printed `path=` line is derived from the parsed flags, so on its own it
    would only prove the flags were read, not that the pipeline honoured them.
    Wrapping the three decode entry points proves the second half too -- and in
    particular that a bare command line really calls `generate_fast`.
    """
    hits = {"fast": 0, "native": 0, "eager": 0}
    orig = (cli.generate_fast, cli.generate_native, cli.generate)

    def wrap(name, fn):
        def inner(*a, **k):
            hits[name] += 1
            return fn(*a, **k)
        return inner

    cli.generate_fast = wrap("fast", orig[0])
    cli.generate_native = wrap("native", orig[1])
    cli.generate = wrap("eager", orig[2])
    argv = [TEXT, "-o", os.path.join(OUT, f"cli{'_'.join(flags) or '_default'}.wav"),
            "--model-dir", MODEL_DIR, "--codec-dir", CODEC_DIR,
            "--seed", "1234", "--max-new-tokens", str(BUDGET), *flags]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(argv)
    finally:
        cli.generate_fast, cli.generate_native, cli.generate = orig
    return rc, out.getvalue(), err.getvalue(), hits


def test_gpu_paths() -> None:
    if not torch.cuda.is_available():
        print("  C: no CUDA -- skipped")
        return
    if not os.path.isdir(MODEL_DIR):
        print(f"  C: SKIP (no bf16 base at {MODEL_DIR})")
        return
    os.makedirs(OUT, exist_ok=True)
    ok = True

    # 1) bare command line: the fast graph decoder, no notice of any kind
    rc, out, err, hits = _cli_once([])
    good = (rc == 0 and hits == {"fast": 1, "native": 0, "eager": 0}
            and "path=fast(cuda-graphs,bitwise-vs-eager)" in out
            and _cli_notes(err) == [])
    ok &= good
    print(f"  C1 bare -> {_summary(out)} generate_fast={hits['fast']} "
          f"generate={hits['eager']} {'ok' if good else 'FAIL'}")

    # 2) --fast: same decoder, plus the single notice line on stderr
    rc, out2, err2, hits2 = _cli_once(["--fast"])
    good = (rc == 0 and hits2 == {"fast": 1, "native": 0, "eager": 0}
            and "path=fast(cuda-graphs,bitwise-vs-eager)" in out2
            and err2.strip() == NOOP_NOTE)
    ok &= good
    print(f"  C2 --fast -> {_summary(out2)} generate_fast={hits2['fast']} "
          f"stderr={err2.strip()!r} {'ok' if good else 'FAIL'}")

    # 3) --eager: the slow reference path (MossTTSModel.step via generate)
    rc, out3, err3, hits3 = _cli_once(["--eager"])
    good = (rc == 0 and hits3 == {"fast": 0, "native": 0, "eager": 1}
            and "path=eager(reference)" in out3 and _cli_notes(err3) == [])
    ok &= good
    print(f"  C3 --eager -> {_summary(out3)} generate={hits3['eager']} "
          f"{'ok' if good else 'FAIL'}")

    # 4) --fast-native: whole-step graph tier, reports its captured graph pool
    rc, out4, err4, hits4 = _cli_once(["--fast-native"])
    good = (rc == 0 and hits4 == {"fast": 0, "native": 1, "eager": 0}
            and "path=fast-native(n2,native-graphs)" in out4
            and "native graphs:" in out4)
    ok &= good
    print(f"  C4 --fast-native -> {_summary(out4)} generate_native={hits4['native']} "
          f"{'ok' if good else 'FAIL'}")

    assert ok, "GPU path smoke FAILED"
    print("  C: GPU smoke -> PASS")


def _summary(out: str) -> str:
    line = [l for l in out.splitlines() if "wrote " in l]
    if not line:
        return "(no summary line)"
    l = line[0]
    steps = l.split("steps=")[1].split(",")[0]
    if "decode avg=" in l:                      # fast.py / native report steps/s
        timing = l.split("decode avg=")[1].split(",")[0]
    else:                                       # eager prints no graph timing
        timing = "prefill=" + l.split("prefill=")[1].split(",")[0]
    return f"steps={steps} {timing}"


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def test_gpu_quant_default() -> None:
    """`--quant` with no fast flag: unchanged behaviour, no extra printing.

    Before v1.1.0 a bare `--quant` printed "--quant implies --fast"; now that
    fast is the default that line would be noise, so it must be gone -- and the
    quantized decode must still land on the fast path.
    """
    export = os.environ.get("MOSS_TTS_8GB_DIR",
                            os.path.join(ROOT, "models_export",
                                         "MOSS-TTS-v1.5-W4GPTQ-w1"))
    if not torch.cuda.is_available():
        print("  D: no CUDA -- skipped")
        return
    if not os.path.isdir(export):
        print(f"  D: SKIP (no standalone export at {export})")
        return
    os.makedirs(OUT, exist_ok=True)
    hits = {"fast": 0, "native": 0, "eager": 0}
    orig = (cli.generate_fast, cli.generate_native, cli.generate)

    def wrap(name, fn):
        def inner(*a, **k):
            hits[name] += 1
            return fn(*a, **k)
        return inner

    cli.generate_fast = wrap("fast", orig[0])
    cli.generate_native = wrap("native", orig[1])
    cli.generate = wrap("eager", orig[2])
    argv = [TEXT, "-o", os.path.join(OUT, "cli_quant_default.wav"),
            "--model-dir", export, "--codec-dir", CODEC_DIR,
            "--seed", "1234", "--max-new-tokens", str(BUDGET),
            "--quant", "w4gptq"]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli_main(argv)
    finally:
        cli.generate_fast, cli.generate_native, cli.generate = orig
    notes = _cli_notes(err.getvalue())
    good = (rc == 0 and hits == {"fast": 1, "native": 0, "eager": 0}
            and "--quant implies" not in err.getvalue()
            and "path=fast(cuda-graphs,bitwise-vs-eager)" in out.getvalue()
            and all("standalone" in l or "GPTQ state" in l for l in notes))
    print(f"  D --quant w4gptq (no fast flag) -> {_summary(out.getvalue())} "
          f"generate_fast={hits['fast']} notes={len(notes)} {'ok' if good else 'FAIL'}")
    assert good, "quant-default smoke FAILED"
    print("  D: quant default path -> PASS")


def main() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_help_renders()
    test_flag_matrix()
    test_gpu_paths()
    test_gpu_quant_default()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
