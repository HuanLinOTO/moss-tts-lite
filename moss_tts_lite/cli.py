"""CLI: text -> TTS -> codec -> 24 kHz wav.

Pipeline: build_tts_prompt -> MossTTSModel -> generate ->
delayed_rows_to_segments -> MossCodecDecoder.decode -> soundfile.write.
Decode path: fast (default, bitwise) / --fast-native (faster, not bitwise) /
--eager (reference). Precedence: CLI > moss_tts_lite/config.yaml > built-ins.
VRAM: codec+TTS co-resident by default; on CUDA OOM, sequential retry.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch

from .bpe import QwenBPE
from .codec import MossCodecDecoder, delayed_rows_to_segments
from .export import is_standalone_dir, load_standalone_model, standalone_presets
from .fast import FastMossTTS, generate_fast
from .fast_native import FastNativeTTS, generate_native
from .gptq import load_gptq_fast
from .generate import generate
from .model import MossTTSModel
from .prompt import build_tts_prompt
from .st_loader import read_safetensors

MODEL_DIR = os.environ.get("MOSS_TTS_MODEL_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "models", "MOSS-TTS-v1.5")
CODEC_DIR = os.environ.get("MOSS_TTS_CODEC_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "models", "MOSS-Audio-Tokenizer")

# Resolution: --gptq-state > $MOSS_TTS_GPTQ_DIR > <model_dir>/gptq/
# (offline artifacts, not shipped; standalone exports on HF/ModelScope).
GPTQ_PRESETS = {"w1": "w1.pt", "w1p": "w1p.pt", "w2": "w2.pt"}
GPTQ_DEFAULT_PRESET = "w1p"
GPTQ_REGEN_HINT = (
    "download a standalone quantized export from HF/ModelScope (see README), "
    "or re-run the GPTQ pipeline in the upstream development workspace")

SR = 24000

#: The delay-ramp flush needs positions beyond max_new_tokens; 64 covers the
#: worst case (normal end: 34 ramp-out rows).
KV_RAMP_MARGIN = 64

#: Library default; the CLI overrides it with the on-demand size below.
DEFAULT_MAX_SEQ_LEN = 8192

_BUILTIN_DEFAULTS = {
    "language": None,
    "seed": 1234,
    "greedy": False,
    "max_new_tokens": 4096,
    "max_seq_len": None,
    "max_graphs": 64,
    "device": "cuda",
    "chunk_duration": 8.0,
    "text_temperature": 1.5,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
    "fast": True,  # default; --eager selects the slow path
    "fast_native": False,
    "watchdog": True,
    "watchdog_silence_frames": None,
    "watchdog_max_segment_frames": None,
}

# [pause Ns] markers are requested silence, not runaway: excluded below.
_PAUSE_RE = re.compile(r"\[pause\s*(\d+(?:\.\d+)?)s\]", re.IGNORECASE)


def _max_pause_s(text: str) -> float | None:
    vals = [float(m) for m in _PAUSE_RE.findall(text or "")]
    return max(vals) if vals else None


def _kv_mib(n_seq: int) -> float:
    """KV bytes for `n_seq` positions: layers * 2 (K,V) * kv_heads * head_dim * bf16."""
    return n_seq * 36 * 2 * 8 * 128 * 2 / 2**20


def _resolve_max_seq_len(requested: int | None, l0: int, max_new_tokens: int) -> int:
    """KV positions to allocate: on-demand when `requested` is None.

    On-demand means exactly what this single call needs:
    ``L0 + max_new_tokens + KV_RAMP_MARGIN`` (see KV_RAMP_MARGIN).  An explicit
    `requested` is a **lower bound**: the allocation is never smaller than the
    on-demand size, so a user asking for a huge cache gets it, and a user asking
    for a smaller one still gets a working run instead of a "prompt exceeds KV
    cache" error.  The CLI does not expose a way to undersize the cache on
    purpose (that is what --max-new-tokens is for).
    """
    need = max(1, int(l0)) + max(0, int(max_new_tokens)) + KV_RAMP_MARGIN
    return max(need, int(requested)) if requested else need


def _kv_advice(exc: ValueError, l0: int, max_new_tokens: int) -> ValueError:
    """Re-raise a KV-cache ValueError with an actionable suggestion appended.

    The generation loops raise a bare "prompt L + max_new_tokens N exceeds KV
    cache S"; on an 8 GB card that is the one error a user can actually fix, so
    the message names both remedies (and says which one this code path uses).
    """
    msg = str(exc)
    if "KV cache" not in msg and "exceeds" not in msg:
        return exc
    return type(exc)(
        f"{msg}\n[moss_tts_lite] the KV cache is sized on demand "
        f"(L0={l0} + --max-new-tokens={max_new_tokens} + {KV_RAMP_MARGIN} ramp), "
        f"so this only happens when the cache was pinned smaller than the run "
        f"needs. Two ways out:\n"
        f"  * lower the budget: --max-new-tokens N (1 step ~ 1/12.5 s of audio), or\n"
        f"  * split the text into segments and synthesize them separately.\n"
        f"  Raising --max-seq-len does not help by itself -- it is a lower bound, "
        f"and the on-demand size already follows --max-new-tokens.")


def _graph_kwargs(max_graphs: int | None) -> dict:
    """FastNativeTTS kwargs; `None` keeps the class default (256)."""
    return {} if max_graphs is None else {"max_graphs": int(max_graphs)}


def _oom_advice(exc: torch.cuda.OutOfMemoryError, text: str,
                max_new_tokens: int) -> torch.cuda.OutOfMemoryError:
    """Wrap a terminal OOM with the two remedies that actually work here.

    A terminal OOM (both the resident and the sequential strategy failed) on a
    small card is almost always a generation budget too large for the card, not
    a broken configuration: the decode peak is weights + graph pool + KV, and of
    those only the KV half scales with the user's request.  Say so.
    """
    return torch.cuda.OutOfMemoryError(
        f"{exc}\n[moss_tts_lite] both VRAM strategies failed (text {len(text)} chars, "
        f"--max-new-tokens {max_new_tokens}).\n"
        f"  * lower the budget: --max-new-tokens N (e.g. 1024 ~ 82 s of audio; "
        f"1 step ~ 1/12.5 s), or\n"
        f"  * split the text into segments and synthesize them separately.\n"
        f"  Auxiliary levers: --fast-native --max-graphs 16 (smaller graph pool), "
        f"or --quant w4 for the smallest shipped weights (7.48 GiB class).")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _yaml_defaults(path: str | None) -> dict:
    """Load optional YAML overrides; never fail the CLI because of them."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - config is strictly optional
        print(f"[moss_tts_lite] ignoring config {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in _BUILTIN_DEFAULTS}


def _load_tts(model_dir: str, dev: torch.device, max_seq_len: int = 8192) -> MossTTSModel:
    weights = read_safetensors(model_dir)
    model = MossTTSModel(weights, device=dev, dtype=torch.bfloat16,
                         max_seq_len=max_seq_len)
    del weights
    torch.cuda.empty_cache()
    return model


def _load_tts_standalone(model_dir: str, dev: torch.device,
                         max_seq_len: int = 8192):
    """Assemble a self-contained quantized export (meta.json; no base ckpt)."""
    model, fast = load_standalone_model(model_dir, device=dev,
                                        max_seq_len=max_seq_len)
    torch.cuda.empty_cache()
    return model, fast


def _standalone_tokenizer(model_dir: str):
    """Tokenizer taken from the export dir itself (QwenBPE on its own files)."""
    tok_json = os.path.join(model_dir, "tokenizer.json")
    return QwenBPE(os.path.join(model_dir, "vocab.json"),
                   os.path.join(model_dir, "merges.txt"),
                   tok_json if os.path.exists(tok_json)
                   else os.path.join(model_dir, "added_tokens.json"))


def _resolve_gptq_state(spec: str, gptq_state: str | None,
                        model_dir: str) -> str:
    """Resolve a `--quant w4gptq[...]` request to a state file path.

    Order: ``--gptq-state`` > ``$MOSS_TTS_GPTQ_DIR`` > ``<model_dir>/gptq/``.
    ``spec`` is a preset name ("w1"/"w2") or a literal path.  Never falls back
    silently to another quantization: a missing state is a hard error (the
    error text carries the regeneration commands).
    """
    fname = GPTQ_PRESETS.get(spec, spec if spec.endswith(".pt") else f"{spec}.pt")
    if gptq_state:
        cands = [gptq_state]
    elif os.path.isabs(spec) or os.sep in spec:
        cands = [spec]
    else:
        cands = []
        env = os.environ.get("MOSS_TTS_GPTQ_DIR")
        if env:
            cands.append(os.path.join(env, fname))
        cands.append(os.path.join(model_dir, "gptq", fname))
    for c in cands:
        if os.path.isfile(c):
            if not os.path.isfile(c + ".meta.json"):
                raise SystemExit(
                    f"[moss_tts_lite] GPTQ state {c} is missing {c}.meta.json "
                    f"(per-linear group sizes / bf16 keeps live there); "
                    f"refusing to load it with guessed parameters.")
            return c
    raise SystemExit(
        f"[moss_tts_lite] GPTQ state not found for --quant {spec!r}; looked at:\n  "
        + "\n  ".join(cands)
        + "\n(a silent fallback to a different quantization would change the "
          "weight\nsemantics, so this is a hard error). Use --gptq-state PATH, "
          "or set\n$MOSS_TTS_GPTQ_DIR, or build the state into "
          "<model_dir>/gptq/, or pass\n--quant w4gptq:auto to fall back to the "
          "non-GPTQ w4 tier.\n" + GPTQ_REGEN_HINT)


def _pipeline(text: str, output: str, *, language, seed, greedy, max_new_tokens,
              device, model_dir, codec_dir, chunk_duration, sampling, resident,
              fast=True, quant=None, w4_group_size=None, watchdog=True,
              watchdog_silence_frames=None, watchdog_max_segment_frames=None,
              max_requested_pause_s=None, gptq_state=None, fast_native=False,
              max_seq_len=None, max_graphs=None):
    """One full pass; returns (wav float32 np, sampling rate, GenResult, stats).

    resident=True  : codec loaded first and kept co-resident during TTS.
    resident=False : TTS -> generate -> free -> codec -> decode.
    fast=True      : CUDA-graph decode path (moss_tts_lite.fast); numerics of the
                     exact path are preserved (M1: bitwise-identical decode).
                     This is the DEFAULT for the CLI; fast=False is the eager
                     reference (MossTTSModel.step, `--eager`).
    fast_native=True: whole-step CUDA graph tier (moss_tts_lite.fast_native,
                     arm n2).  NOT bitwise: it swaps kernels (`F.rms_norm`, fused
                     int4 GEMMs, fused heads) to cut the step's kernel count, so
                     it decodes a different-but-valid utterance at ~97 steps/s
                     on W4 (~81 for fast.py); measured on A10G.
                     Incompatible with --greedy (the native loop samples).
    max_seq_len    : KV cache positions.  ``None`` (the CLI default) sizes it on
                     demand: the prompt is built first, L0 is known before the
                     weights are loaded, and the cache is exactly
                     ``L0 + max_new_tokens + KV_RAMP_MARGIN``.  An explicit
                     integer is a *lower bound* (a small request is still rounded
                     up to the on-demand size, which the run actually needs).
    max_graphs     : fast_native's whole-step-graph pool cap (see FastNativeTTS).
    Either way every large object is released in ``finally``.
    """
    dev = torch.device(device)
    model = None
    fast_model = None
    native = None
    codec = None
    try:
        if resident:
            codec = MossCodecDecoder(codec_dir, device=dev)
        standalone = is_standalone_dir(model_dir)
        tokenizer = _standalone_tokenizer(model_dir) if standalone else None
        # prompt first: L0 fixes the on-demand KV size
        prompt = build_tts_prompt(text, language=language, tokenizer=tokenizer)
        want_seq = _resolve_max_seq_len(max_seq_len, int(prompt["input_ids"].shape[1]),
                                        max_new_tokens)
        if standalone:
            if gptq_state:
                raise SystemExit(
                    "[moss_tts_lite] --gptq-state cannot be combined with a "
                    "standalone model dir (the quantization is inside it)")
            if not fast:
                raise SystemExit(
                    "[moss_tts_lite] --model-dir is a standalone quantized "
                    "export: it only runs the fast/quant decode path, so "
                    "--eager (or fast=False) cannot be honoured here")
            model, fast_model = _load_tts_standalone(model_dir, dev, want_seq)
        else:
            model = _load_tts(model_dir, dev, want_seq)
        stats: dict = {}
        if fast:
            if standalone:
                pass          # quantized weights already installed by the loader
            elif gptq_state:
                fast_model = load_gptq_fast(model, gptq_state)
            else:
                _fkw = {} if w4_group_size is None else {"w4_group_size": w4_group_size}
                fast_model = FastMossTTS(model, quant=quant, **_fkw)
            if fast_native:
                # built once: per-length graphs must survive across calls
                native = FastNativeTTS(fast_model, arm="n2",
                                       **_graph_kwargs(max_graphs))
            try:
                res = generate_fast(
                    fast_model, prompt,
                    max_new_tokens=max_new_tokens, seed=seed, greedy=greedy, **sampling,
                    stats=stats, watchdog=watchdog,
                    watchdog_silence_frames=watchdog_silence_frames,
                    watchdog_max_segment_frames=watchdog_max_segment_frames,
                    max_requested_pause_s=max_requested_pause_s,
                ) if not fast_native else generate_native(
                    native, prompt,
                    max_new_tokens=max_new_tokens, seed=seed, greedy=greedy, **sampling,
                    stats=stats, watchdog=watchdog,
                    watchdog_silence_frames=watchdog_silence_frames,
                    watchdog_max_segment_frames=watchdog_max_segment_frames,
                    max_requested_pause_s=max_requested_pause_s,
                )
            except ValueError as exc:
                raise _kv_advice(exc, int(prompt["input_ids"].shape[1]),
                                 max_new_tokens) from None
        else:
            res = generate(
                model, prompt,
                max_new_tokens=max_new_tokens, seed=seed, greedy=greedy, **sampling,
                stats=stats,
            )
        stats["max_seq_len"] = int(model.max_seq_len)
        stats["l0"] = int(prompt["input_ids"].shape[1])
        stats["kv_mib"] = _kv_mib(int(model.max_seq_len))
        stats["max_graphs"] = (int(native.graph_count())
                               if native is not None else None)
        stats["graph_pool_cap"] = (int(native.max_graphs)
                                   if native is not None else None)
        segments = delayed_rows_to_segments(res.audio_frames)
        del fast_model
        fast_model = None
        del native
        native = None
        del model
        model = None
        torch.cuda.empty_cache()
        if not resident:
            codec = MossCodecDecoder(codec_dir, device=dev)
        wavs = [codec.decode(seg, chunk_duration=chunk_duration) for seg in segments]
        wav = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        sf.write(output, wav, SR, subtype="FLOAT")
        return wav, SR, res, stats
    finally:
        if fast_model is not None:
            del fast_model
        if native is not None:
            del native
        if model is not None:
            del model
        if codec is not None:
            del codec
        torch.cuda.empty_cache()


def synthesize(text: str, output: str, *, language=None, seed=1234, greedy=False,
               max_new_tokens=4096, device="cuda", model_dir=MODEL_DIR,
               codec_dir=CODEC_DIR, chunk_duration=8.0,
               sampling: dict | None = None, stats: dict | None = None,
               fast=True, quant=None, w4_group_size=None, watchdog=True,
               watchdog_silence_frames=None, watchdog_max_segment_frames=None,
               gptq_state=None, fast_native=False, max_seq_len=None,
               max_graphs=None):
    """End-to-end text -> wav.  Returns (wav, sr, res, strategy).

    Tries the co-resident strategy (codec + TTS on one card) first and falls
    back to sequential (TTS freed before the codec loads) on CUDA OOM.

    `fast=True` (the default, matching the CLI) selects the CUDA-graph path;
    `fast=False` is the eager reference.  This mirrors the CLI's behaviour flip
    (v1.1.0): a caller that wants the old eager default must now say so, and
    `test_parity`/library callers that exercise `MossTTSModel.step` directly are
    unaffected either way.

    `max_seq_len=None` (the CLI default) sizes the KV cache on demand; see
    `_resolve_max_seq_len`.  `max_graphs=None` keeps FastNativeTTS's own default.
    """
    sampling = {k: v for k, v in sampling.items() if v is not None} if sampling else {}
    max_pause = _max_pause_s(text)
    strategy = "resident"
    while True:
        try:
            wav, sr, res, tstats = _pipeline(
                text, output, language=language, seed=seed, greedy=greedy,
                max_new_tokens=max_new_tokens, device=device, model_dir=model_dir,
                codec_dir=codec_dir, chunk_duration=chunk_duration,
                sampling=sampling, resident=(strategy == "resident"), fast=fast,
                quant=quant, w4_group_size=w4_group_size, watchdog=watchdog,
                watchdog_silence_frames=watchdog_silence_frames,
                watchdog_max_segment_frames=watchdog_max_segment_frames,
                max_requested_pause_s=max_pause, gptq_state=gptq_state,
                fast_native=fast_native, max_seq_len=max_seq_len,
                max_graphs=max_graphs)
            break
        except torch.cuda.OutOfMemoryError as exc:
            if strategy != "resident":
                raise _oom_advice(exc, text, max_new_tokens) from None
            print("[moss_tts_lite] CUDA OOM with codec co-resident; "
                  "retrying with sequential load (TTS freed before codec)",
                  file=sys.stderr)
            strategy = "sequential"
    if stats is not None:
        stats.update(tstats)
    return wav, SR, res, strategy


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m moss_tts_lite",
        description="MOSS-TTS-v1.5 minimal inference: text -> 24 kHz wav.")
    p.add_argument("text", help="text to synthesize")
    p.add_argument("-o", "--output", required=True, help="output .wav path")
    p.add_argument("--language", default=None,
                   help="optional language tag (e.g. 'English')")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--greedy", action="store_true",
                   help="argmax decoding instead of seeded sampling")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="generation step budget (default 4096)")
    p.add_argument("--max-seq-len", type=int, default=None, metavar="N",
                   help="KV cache size in positions. Default: on demand = "
                        "L0 + --max-new-tokens + 64 (L0 is the prompt length, "
                        "known after tokenization; ~0.14 MiB per position). "
                        "An explicit N is a LOWER BOUND: the cache is never "
                        "smaller than the run needs, so raising it only matters "
                        "if you also raise --max-new-tokens. Use it to pin a "
                        "known cache size (e.g. a server reusing one process)")
    p.add_argument("--max-graphs", type=int, default=None, metavar="N",
                   help="--fast-native's whole-step graph pool cap (default 64; "
                        "One graph per decoded cache "
                        "length, ~6-7.5 MiB each; beyond the cap the oldest is "
                        "evicted and re-captured (~40-80 ms), so a one-shot CLI "
                        "call is unaffected (it captures every length once "
                        "anyway) while a long utterance in a long-lived process "
                        "degrades if N is too small")
    p.add_argument("--device", default=None,
                   help="'cuda' (default) or 'cpu' (eager only, no "
                        "quantization)")
    p.add_argument("--fast", dest="fast", action="store_true",
                   help="no-op: fast is the default (kept for compatibility)")
    p.add_argument("--eager", dest="eager", action="store_true",
                   help="eager reference path (slow; debugging only)")
    p.add_argument("--fast-native", dest="fast_native", action="store_true",
                   help="faster than the default (~97 vs ~82 steps/s on W4) "
                        "but not bitwise-identical; per-length graph capture "
                        "makes one-shot calls slower -- best in a long-lived "
                        "process. Incompatible with --greedy")
    p.add_argument("--quant", default=None,
                   choices=["w4", "w4g32", "w8", "w4gptq", "w4gptq:w1p",
                            "w4gptq:w2", "w4gptq:auto"],
                   help="weight quantization tier: w4gptq (W4 GPTQ, default "
                        "tier w1p), w4gptq:w2 (quality tier), w4gptq:auto "
                        "(cached state else w4), w4 (RTN g128), w4g32, w8 "
                        "(int8); default bf16")
    p.add_argument("--gptq-state", default=None, metavar="PATH",
                   help="offline GPTQ state file for w4gptq* (overrides "
                        "$MOSS_TTS_GPTQ_DIR and <model_dir>/gptq/)")
    p.add_argument("--no-watchdog", dest="watchdog", action="store_false",
                   help="disable the runaway-generation watchdog (default: on)")
    p.add_argument("--watchdog-silence-frames", type=int, default=None,
                   help="watchdog consecutive-low-energy frame threshold "
                        "(default: auto)")
    p.add_argument("--watchdog-max-segment-frames", type=int, default=None,
                   help="watchdog per-segment length floor (default: auto)")
    p.add_argument("--model-dir", default=None,
                   help="MOSS-TTS-v1.5 checkpoint directory")
    p.add_argument("--codec-dir", default=None,
                   help="MOSS-Audio-Tokenizer checkpoint directory")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="optional YAML defaults file "
                        "(default: moss_tts_lite/config.yaml if present)")
    args = p.parse_args(argv)

    cfg = {**_BUILTIN_DEFAULTS, **_yaml_defaults(args.config)}
    # CLI flags (non-None) win over YAML
    if args.language is not None:
        cfg["language"] = args.language
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.greedy:
        cfg["greedy"] = True
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.max_seq_len is not None:
        cfg["max_seq_len"] = args.max_seq_len
    if args.max_graphs is not None:
        cfg["max_graphs"] = args.max_graphs
    if args.device is not None:
        cfg["device"] = args.device
    if args.eager:
        cfg["fast"] = False
        cfg["fast_native"] = False
        if args.fast:
            # contradiction: refuse instead of silent last-wins
            print("[moss_tts_lite] --eager wins over --fast (--fast is a no-op "
                  "alias anyway); running the eager reference path",
                  file=sys.stderr)
    if args.fast:
        # no-op alias (fast is the default), kept for compatibility
        if not args.eager:
            print("[--fast is now the default; this flag is a no-op]",
                  file=sys.stderr)
    if args.fast_native:
        cfg["fast"] = True
        cfg["fast_native"] = True
    if args.eager and args.fast_native:
        raise SystemExit("[moss_tts_lite] --eager and --fast-native are mutually "
                         "exclusive (--eager is the reference slow path, "
                         "--fast-native the fastest CUDA-graph tier)")
    if cfg["device"] == "cpu":
        if cfg["fast_native"]:
            raise SystemExit("[moss_tts_lite] --fast-native requires CUDA "
                             "(whole-step CUDA graphs)")
        if cfg["fast"]:
            print("[moss_tts_lite] --device cpu: CUDA graphs need a GPU, so the "
                  "eager reference path runs instead of the default fast path "
                  "(quantization is likewise unavailable on cpu)",
                  file=sys.stderr)
            cfg["fast"] = False
    if args.fast_native and cfg["greedy"]:
        # the native tier samples; no argmax path exists
        raise SystemExit("[moss_tts_lite] --fast-native cannot be combined with "
                         "--greedy (the native tier samples)")
    # standalone dirs are self-describing (meta.json); no separate state file
    standalone_preset = None
    if args.model_dir and is_standalone_dir(args.model_dir):
        standalone_preset = standalone_presets(args.model_dir)[0]
        if args.eager:
            raise SystemExit(
                f"[moss_tts_lite] {args.model_dir} is a standalone quantized "
                f"export (preset {standalone_preset!r}); it has no bf16 weights, "
                f"so --eager cannot run it -- drop --eager")
        cfg["fast"] = True
        if args.gptq_state:
            raise SystemExit(
                "[moss_tts_lite] --gptq-state cannot be combined with a standalone "
                "model dir (the quantization is inside it)")
        if args.quant is not None:
            want = args.quant.split(":", 1)[1] if ":" in args.quant else "w1"
            if want not in (standalone_preset, "auto"):
                raise SystemExit(
                    f"[moss_tts_lite] {args.model_dir} provides preset "
                    f"{standalone_preset!r} only, not {want!r}; drop --quant "
                    f"or point --model-dir at the {want!r} export")
        print(f"[moss_tts_lite] standalone quantized export detected: using its "
              f"built-in preset {standalone_preset!r}")
    # --quant needs the fast path; --eager cannot carry it
    if args.quant is not None and args.eager:
        raise SystemExit("[moss_tts_lite] --quant cannot be combined with --eager: "
                         "weight quantization is only implemented on the fast "
                         "path (drop --eager and let the default run)")
    if args.quant is not None and cfg["device"] == "cpu":
        raise SystemExit("[moss_tts_lite] --quant requires CUDA (the int4 "
                         "kernels have no cpu fallback)")
    quant, w4_group, gptq_state = None, None, None
    if args.quant is not None and standalone_preset is None:
        if args.quant == "w4":
            quant, w4_group = "w4", 128
        elif args.quant == "w4g32":
            quant, w4_group = "w4", 32
        elif args.quant == "w8":
            quant, w4_group = "w8", None
        else:                                   # w4gptq / w4gptq:w2 / *:auto
            preset = (args.quant.split(":", 1)[1] if ":" in args.quant
                      else GPTQ_DEFAULT_PRESET)
            want_auto = preset == "auto"
            if want_auto:
                preset = GPTQ_DEFAULT_PRESET
            try:
                gptq_state = _resolve_gptq_state(
                    preset, args.gptq_state,
                    args.model_dir or os.environ.get("MOSS_TTS_MODEL_DIR")
                    or MODEL_DIR)
                print(f"[moss_tts_lite] --quant {args.quant}: GPTQ state {gptq_state}")
            except SystemExit as e:
                if not want_auto:
                    raise
                print(f"[moss_tts_lite] {e}\n[moss_tts_lite] --quant w4gptq:auto -> "
                      f"falling back to 'w4' (RTN group-128)")
                quant, w4_group = "w4", 128
    if not args.watchdog:
        cfg["watchdog"] = False
    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = os.environ.get("MOSS_TTS_MODEL_DIR") or MODEL_DIR
    if args.codec_dir is not None:
        codec_dir = args.codec_dir
    else:
        codec_dir = os.environ.get("MOSS_AUDIO_MODEL_DIR") or CODEC_DIR

    sampling = {
        "text_temperature": cfg["text_temperature"],
        "text_top_p": cfg["text_top_p"],
        "text_top_k": cfg["text_top_k"],
        "audio_temperature": cfg["audio_temperature"],
        "audio_top_p": cfg["audio_top_p"],
        "audio_top_k": cfg["audio_top_k"],
        "audio_repetition_penalty": cfg["audio_repetition_penalty"],
    }
    stats: dict = {}
    wav, sr, res, strategy = synthesize(
        args.text, args.output,
        language=cfg["language"], seed=cfg["seed"], greedy=cfg["greedy"],
        max_new_tokens=cfg["max_new_tokens"], device=cfg["device"],
        model_dir=model_dir, codec_dir=codec_dir,
        chunk_duration=cfg["chunk_duration"],
        sampling=sampling, stats=stats, fast=bool(cfg["fast"]),
        quant=quant, w4_group_size=w4_group, watchdog=cfg["watchdog"],
        gptq_state=gptq_state,
        watchdog_silence_frames=cfg["watchdog_silence_frames"],
        watchdog_max_segment_frames=cfg["watchdog_max_segment_frames"],
        fast_native=bool(cfg.get("fast_native")),
        max_seq_len=cfg.get("max_seq_len"),
        max_graphs=cfg.get("max_graphs"))

    steps = stats.get("steps") or res.n_steps
    step_ms = stats.get("step_ms") or []
    avg = sum(step_ms) / len(step_ms) if step_ms else float("nan")
    graphs = stats.get("max_graphs")
    if cfg.get("fast_native"):
        path = "fast-native(n2,native-graphs)"
    elif cfg["fast"]:
        path = "fast(cuda-graphs,bitwise-vs-eager)"
    else:
        path = "eager(reference)"
    print(f"[moss_tts_lite] wrote {args.output}: {len(wav)} samples "
          f"({len(wav) / sr:.2f}s @ {sr} Hz), steps={res.n_steps}, "
          f"finished={res.finished}, path={path}, vram_strategy={strategy}, "
          f"prefill={stats.get('prefill_ms', float('nan')):.0f}ms, "
          f"decode avg={avg:.1f}ms/step ({1000 / avg:.1f} steps/s)")
    print(f"[moss_tts_lite] KV: L0={stats.get('l0')} + "
          f"--max-new-tokens={cfg['max_new_tokens']} + {KV_RAMP_MARGIN} ramp "
          f"-> max_seq_len={stats.get('max_seq_len')} "
          f"({stats.get('kv_mib', float('nan')):.0f} MiB"
          + (f", user floor {cfg['max_seq_len']}" if cfg.get("max_seq_len") else "")
          + ")")
    if graphs is not None:
        print(f"[moss_tts_lite] native graphs: {graphs} captured "
              f"(cap {stats.get('graph_pool_cap')})")
    if getattr(res, "watchdog_triggered", False):
        ws = getattr(res, "watchdog_stats", {})
        print(f"[moss_tts_lite] WARNING: production watchdog triggered "
              f"({res.watchdog_reason}); forced audio end. stats={ws}")
    return 0
