"""CLI gates:"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import contextlib

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

NOOP_NOTE = "[--fast is now the default; this flag is a no-op]"

OBSOLETE_NOTE = "--quant implies --fast"

def _help_text() -> str:
    proc = subprocess.run([sys.executable, "-m", "moss_tts_lite", "--help"],
                          cwd=LITE_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout

def test_help_renders() -> None:
    raw = _help_text()

    text = " ".join(raw.split())
    assert "python -m moss_tts_lite" in raw

    usage = text.split("positional arguments:")[0]
    assert "--eager" in usage and "--fast" in usage and "--fast-native" in usage

    assert "--fast no-op: fast is the default" in text, text[:200]
    assert "--eager eager reference path" in text, text[:200]
    assert "--fast-native faster than the default" in text
    assert "not bitwise-identical" in text
    print("  A: --help renders; usage lists --eager; --fast says no-op; "
          "--fast-native says 'faster, not bitwise' -> PASS")

class _FakeRes:
    n_steps = 1
    finished = True
    audio_frames = None
    watchdog_triggered = False

def _fake_synthesize(text, output, **kw):
    """Stand-in for cli."""
    calls.append(kw)
    kw["stats"].update({"steps": 1, "step_ms": [10.0], "prefill_ms": 5.0,
                        "l0": 66, "max_seq_len": 4226, "kv_mib": 566.0,
                        "max_graphs": None, "graph_pool_cap": None})
    return np.zeros(24000, dtype="float32"), 24000, _FakeRes(), "resident"

calls: list[dict] = []

def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run main(argv) with the pipeline stubbed;"""
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

        calls.clear()
        rc, out2, err2 = _run(["hi", "-o", "out.wav", "--fast"])
        kw = calls[-1]
        good = (kw["fast"] is True and kw["fast_native"] is False
                and err2.strip() == NOOP_NOTE and OBSOLETE_NOTE not in err2
                and "path=fast(cuda-graphs" in out2)
        ok &= good
        print(f"  B2 --fast (no-op alias)      -> fast={kw['fast']} "
              f"stderr={err2.strip()!r} {'ok' if good else 'FAIL'}")

        calls.clear()
        rc, out3, err3 = _run(["hi", "-o", "out.wav", "--eager"])
        kw = calls[-1]
        good = (kw["fast"] is False and kw["fast_native"] is False
                and err3 == "" and "path=eager(reference)" in out3)
        ok &= good
        print(f"  B3 --eager                   -> fast={kw['fast']} "
              f"path=eager printed={'path=eager(reference)' in out3} "
              f"{'ok' if good else 'FAIL'}")

        calls.clear()
        rc, out4, err4 = _run(["hi", "-o", "out.wav", "--fast-native"])
        kw = calls[-1]
        good = (kw["fast"] is True and kw["fast_native"] is True
                and "path=fast-native(n2" in out4)
        ok &= good
        print(f"  B4 --fast-native             -> fast={kw['fast']} "
              f"native={kw['fast_native']} {'ok' if good else 'FAIL'}")

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

        calls.clear()
        rc, out6, err6 = _run(["hi", "-o", "out.wav", "--fast", "--quant", "w4"])
        kw = calls[-1]
        good = (kw["fast"] is True and err6.strip() == NOOP_NOTE
                and err6.count(NOOP_NOTE) == 1)
        ok &= good
        print(f"  B6 --fast --quant w4         -> notice printed exactly once "
              f"{'ok' if good else 'FAIL'}")

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

        good = _BUILTIN_DEFAULTS["fast"] is True
        ok &= good
        print(f"  B9 _BUILTIN_DEFAULTS['fast']={_BUILTIN_DEFAULTS['fast']} "
              f"{'ok' if good else 'FAIL'}")

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

TEXT = "你好。"
BUDGET = 96

def _cli_notes(err: str) -> list[str]:
    """CLI's own stderr lines (torch warnings and the like are not ours)."""
    return [l for l in err.splitlines()
            if "moss_tts_lite]" in l or l.startswith("[--fast")]

def _cli_once(flags: list[str]) -> tuple[int, str, str, dict[str, int]]:
    """Run the real pipeline and count which top-level decoder entry fired."""
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

    rc, out, err, hits = _cli_once([])
    good = (rc == 0 and hits == {"fast": 1, "native": 0, "eager": 0}
            and "path=fast(cuda-graphs,bitwise-vs-eager)" in out
            and _cli_notes(err) == [])
    ok &= good
    print(f"  C1 bare -> {_summary(out)} generate_fast={hits['fast']} "
          f"generate={hits['eager']} {'ok' if good else 'FAIL'}")

    rc, out2, err2, hits2 = _cli_once(["--fast"])
    good = (rc == 0 and hits2 == {"fast": 1, "native": 0, "eager": 0}
            and "path=fast(cuda-graphs,bitwise-vs-eager)" in out2
            and err2.strip() == NOOP_NOTE)
    ok &= good
    print(f"  C2 --fast -> {_summary(out2)} generate_fast={hits2['fast']} "
          f"stderr={err2.strip()!r} {'ok' if good else 'FAIL'}")

    rc, out3, err3, hits3 = _cli_once(["--eager"])
    good = (rc == 0 and hits3 == {"fast": 0, "native": 0, "eager": 1}
            and "path=eager(reference)" in out3 and _cli_notes(err3) == [])
    ok &= good
    print(f"  C3 --eager -> {_summary(out3)} generate={hits3['eager']} "
          f"{'ok' if good else 'FAIL'}")

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
    if "decode avg=" in l:
        timing = l.split("decode avg=")[1].split(",")[0]
    else:
        timing = "prefill=" + l.split("prefill=")[1].split(",")[0]
    return f"steps={steps} {timing}"

def test_gpu_quant_default() -> None:
    """`--quant` with no fast flag:"""
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
