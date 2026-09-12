"""Command-line entry point: text -> TTS -> MOSS-Audio codec -> 24 kHz wav.

Usage:
    python -m moss_tts_lite "text" -o out.wav [--language X] [--seed 1234]
        [--greedy] [--max-new-tokens 4096] [--device cuda]
        [--fast] [--fast-native] [--quant TIER]
        [--model-dir DIR] [--config PATH]

Decode paths:
    (default)      exact eager path; bitwise-parity reference
    --fast         CUDA-graph path (moss_tts_lite.fast); bitwise-identical
                   decode to the exact path (M1), ~81 steps/s on W4
    --fast-native  whole-step CUDA graph tier (moss_tts_lite.fast_native,
                   arm n2); NOT bitwise -- swaps kernels for kernel-count
                   reduction, decodes a different-but-valid utterance at
                   ~97 steps/s on W4 (text argmax 100%, audio top-25 97.5%)

Pipeline: build_tts_prompt -> MossTTSModel -> generate -> 
delayed_rows_to_segments -> MossCodecDecoder.decode -> soundfile.write
(24 kHz mono, float32).

Dependencies: torch / numpy / soundfile / pyyaml + stdlib only.

Optional YAML defaults are read from ``moss_tts_lite/config.yaml`` next to the
package when it exists (keys: language, seed, greedy, max_new_tokens, device,
chunk_duration, text_temperature, text_top_p, text_top_k, audio_temperature,
audio_top_p, audio_top_k, audio_repetition_penalty).  Precedence:
command line > YAML > built-ins.  A missing file or missing pyyaml simply
falls back to built-ins.

VRAM strategy (A10G-24G: TTS bf16 peak ~17 GiB + codec fp32 ~3.6 GiB):
    resident   load the codec FIRST, then TTS, keep both co-resident
               (~20.6 GiB peak) — the default;
    sequential on CUDA OOM, retry: TTS -> generate -> free TTS -> codec ->
               decode (~17 GiB peak).  The strategy actually used is printed.
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

# ---- GPTQ states (offline artifacts; see .tmp/reports/gptq-2-final.md) -----
# Resolution order: --gptq-state > $MOSS_TTS_GPTQ_DIR > <model_dir>/gptq/.
# The state file is *not* shipped in git (4 GiB); regenerate with
#   python3 .tmp/gptq_agent/run_gptq.py --group-size 32 --damp 0.1 ...
#   python3 .tmp/gptq_agent/merge_states.py --base ... --out <path>
GPTQ_PRESETS = {"w1": "w1.pt", "w1p": "w1p.pt", "w2": "w2.pt"}
GPTQ_DEFAULT_PRESET = "w1p"
GPTQ_REGEN_HINT = (
    "regenerate it (GPU, ~35 min, needs .tmp/gptq_agent/calib/):\n"
    "  python3 .tmp/gptq_agent/run_gptq.py --group-size 32 --damp 0.1 \\\n"
    "      --scale-from compensated --inverse-device cpu --layers 0:17 --tag g32_a\n"
    "  python3 .tmp/gptq_agent/run_gptq.py ... --layers 18:35 --tag g32_b\n"
    "  python3 .tmp/gptq_agent/merge_states.py --union \\\n"
    "      gptq_state_g32_a.pt,gptq_state_g32_b.pt --union-group 32 \\\n"
    "      --out gptq_state_g32.pt\n"
    "  python3 .tmp/gptq_agent/merge_states.py --base gptq_state_g32.pt \\\n"
    "      --base-group 32 --bf16-linears <layers> --out <state path>") 

SR = 24000

_BUILTIN_DEFAULTS = {
    "language": None,
    "seed": 1234,
    "greedy": False,
    "max_new_tokens": 4096,
    "device": "cuda",
    "chunk_duration": 8.0,
    "text_temperature": 1.5,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
    "fast": False,
    "fast_native": False,
    "watchdog": True,
    "watchdog_silence_frames": None,
    "watchdog_max_segment_frames": None,
}

# [pause 8s] / [pause 1.5s] markers in the raw input text; the watchdog must
# not mistake a requested pause for runaway silence.
_PAUSE_RE = re.compile(r"\[pause\s*(\d+(?:\.\d+)?)s\]", re.IGNORECASE)


def _max_pause_s(text: str) -> float | None:
    vals = [float(m) for m in _PAUSE_RE.findall(text or "")]
    return max(vals) if vals else None

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


def _load_tts(model_dir: str, dev: torch.device) -> MossTTSModel:
    weights = read_safetensors(model_dir)
    model = MossTTSModel(weights, device=dev, dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()
    return model


def _load_tts_standalone(model_dir: str, dev: torch.device):
    """Assemble a self-contained quantized export (meta.json; no base ckpt)."""
    model, fast = load_standalone_model(model_dir, device=dev)
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
              fast=False, quant=None, w4_group_size=None, watchdog=True,
              watchdog_silence_frames=None, watchdog_max_segment_frames=None,
              max_requested_pause_s=None, gptq_state=None, fast_native=False):
    """One full pass; returns (wav float32 np, sampling rate, GenResult).

    resident=True  : codec loaded first and kept co-resident during TTS.
    resident=False : TTS -> generate -> free -> codec -> decode.
    fast=True      : CUDA-graph decode path (moss_tts_lite.fast); numerics of the
                     exact path are preserved (M1: bitwise-identical decode).
    fast_native=True: whole-step CUDA graph tier (moss_tts_lite.fast_native,
                     arm n2).  NOT bitwise: it swaps kernels (`F.rms_norm`, fused
                     int4 GEMMs, fused heads) to cut the step's kernel count, so
                     it decodes a different-but-valid utterance at ~97 steps/s
                     on W4 (~81 for fast.py); see .tmp/reports/native-1.md.
                     Incompatible with --greedy (the native loop samples).
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
        if standalone:
            # self-contained quantized export: the int4 payloads AND the
            # tokenizer files live inside the model dir, so nothing is read
            # from the default models/ layout.
            if gptq_state:
                raise SystemExit(
                    "[moss_tts_lite] --gptq-state cannot be combined with a "
                    "standalone model dir (the quantization is inside it)")
            if not fast:
                print("[moss_tts_lite] --model-dir is a standalone quantized "
                      "export; using the fast/quant decode path",
                      file=sys.stderr)
                fast = True
            tokenizer = _standalone_tokenizer(model_dir)
            model, fast_model = _load_tts_standalone(model_dir, dev)
        else:
            tokenizer = None
            model = _load_tts(model_dir, dev)
        prompt = build_tts_prompt(text, language=language, tokenizer=tokenizer)
        stats: dict = {}
        if fast:
            if standalone:
                pass          # quantized weights already installed by the loader
            elif gptq_state:
                fast_model = load_gptq_fast(model, gptq_state)
            else:
                # `w4_group_size=None` must NOT be passed through: it would
                # override FastMossTTS's own default (128) and trip its
                # 32/64/128/256 validation, which is what made bare `--fast`
                # fail with "w4_group_size must be 32/64/128/256" (pre-existing
                # at bc5415f; only the --quant tiers set it explicitly).
                _fkw = {} if w4_group_size is None else {"w4_group_size": w4_group_size}
                fast_model = FastMossTTS(model, quant=quant, **_fkw)
            if fast_native:
                # arm n2: whole-step graph + fused kernels; constructed once so
                # its captured graphs survive across calls in a long-lived
                # process (one graph is captured per decoded cache length)
                native = FastNativeTTS(fast_model, arm="n2")
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
        else:
            res = generate(
                model, prompt,
                max_new_tokens=max_new_tokens, seed=seed, greedy=greedy, **sampling,
                stats=stats,
            )
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
               fast=False, quant=None, w4_group_size=None, watchdog=True,
               watchdog_silence_frames=None, watchdog_max_segment_frames=None,
               gptq_state=None, fast_native=False):
    """End-to-end text -> wav.  Returns (wav, sr, res, strategy).

    Tries the co-resident strategy (codec + TTS on one card) first and falls
    back to sequential (TTS freed before the codec loads) on CUDA OOM.
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
                fast_native=fast_native)
            break
        except torch.cuda.OutOfMemoryError:
            if strategy != "resident":
                raise
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
    p.add_argument("--device", default=None, help="'cuda' (default) or 'cpu'")
    p.add_argument("--fast", action="store_true",
                   help="CUDA-graph decode path (moss_tts_lite.fast); "
                        "bitwise-identical output, faster steps")
    p.add_argument("--fast-native", dest="fast_native", action="store_true",
                   help="whole-step CUDA graph tier (moss_tts_lite.fast_native, "
                        "arm n2); NOT bitwise -- it swaps kernels (F.rms_norm, "
                        "fused int4 GEMMs/heads) to cut the step's kernel "
                        "count and decodes a different-but-valid utterance: "
                        "text argmax 100%%, audio top-25 97.5%% (W4 baseline "
                        "75.4%%). WARNING: it needs one graph captured per "
                        "decoded cache length, so a ONE-SHOT CLI CALL IS "
                        "SLOWER than --fast (~105 vs ~28 ms/step here: 143 "
                        "captures for a 144-step utterance). It reaches ~97 "
                        "steps/s only once the graphs are warm, i.e. in a "
                        "long-lived process that synthesizes repeatedly "
                        "(fast.py's sub-graphs are length-independent and so "
                        "pay this only once). Implies --fast; incompatible "
                        "with --greedy. Default: off")
    p.add_argument("--quant", default=None,
                   choices=["w4", "w4g32", "w8", "w4gptq", "w4gptq:w1p",
                            "w4gptq:w2", "w4gptq:auto"],
                   help="weight quantization (implies --fast). GPTQ tiers "
                        "(calibration-quantized, offline state): 'w4gptq' = "
                        "W1, the default tier (~8.3 GiB, ~81 steps/s, audio "
                        "top-25 89.30%% [CUDA topk] / 98.32%% [tie-robust], "
                        "runaway 0/12); 'w4gptq:w2' = W2 quality tier (~8.5 GiB, "
                        "~78 steps/s, top-25 91.25%% [CUDA topk] / 99.30%% "
                        "[tie-robust], 0/12); 'w4gptq:w1p' = W1p (same shape and "
                        "speed class as W1, quantized against the ten-language "
                        "calibration v3, see .tmp/reports/w1plus-1.md) - the "
                        "top-25 figure depends on the "
                        "tie convention, see moss_tts_lite/README.md; "
                        "'w4gptq:auto' = cached state if present, else fall "
                        "back to 'w4' with a notice. Non-GPTQ tiers: 'w4' = "
                        "RTN int4 group-128 (perf-m4 semantics; ~7.4 GiB, ~91 "
                        "steps/s, top-25 75.4%%, runaway 3/12), 'w4g32' = RTN "
                        "int4 group-32 (~8.1 GiB, ~84 steps/s), 'w8' = int8 "
                        "per-channel (memory-saving, slow decode); default: "
                        "bf16")
    p.add_argument("--gptq-state", default=None, metavar="PATH",
                   help="offline GPTQ state file for --quant w4gptq* (needs a "
                        "sibling <PATH>.meta.json). Overrides "
                        "$MOSS_TTS_GPTQ_DIR and <model_dir>/gptq/")
    p.add_argument("--no-watchdog", dest="watchdog", action="store_false",
                   help="disable the runaway-generation watchdog (default: on; "
                        "forces an audio end after ~5.1 s of consecutive "
                        "low-energy frames or an over-long segment)")
    p.add_argument("--watchdog-silence-frames", type=int, default=None,
                   help="override the watchdog's consecutive-low-energy "
                        "threshold (default: auto = 64 frames, raised to cover "
                        "[pause Xs] markers in the text)")
    p.add_argument("--watchdog-max-segment-frames", type=int, default=None,
                   help="override the watchdog's per-segment length floor "
                        "(default: auto = 640 frames)")
    p.add_argument("--model-dir", default=None,
                   help="MOSS-TTS-v1.5 checkpoint directory")
    p.add_argument("--codec-dir", default=None,
                   help="MOSS-Audio-Tokenizer checkpoint directory")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="optional YAML defaults file "
                        "(default: moss_tts_lite/config.yaml if present)")
    args = p.parse_args(argv)

    cfg = {**_BUILTIN_DEFAULTS, **_yaml_defaults(args.config)}
    # CLI flags win over YAML; a flag left at None was not given.
    if args.language is not None:
        cfg["language"] = args.language
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.greedy:
        cfg["greedy"] = True
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.device is not None:
        cfg["device"] = args.device
    if args.fast:
        cfg["fast"] = True
    if args.fast_native:
        # implies the fast path; the native tier needs the quantized container
        cfg["fast"] = True
        cfg["fast_native"] = True
    if args.fast_native and cfg["greedy"]:
        # generate_native() samples; greedy would need an argmax path that the
        # whole-step graph does not implement (the reference's greedy branch is
        # a different sampling call sequence)
        raise SystemExit("[moss_tts_lite] --fast-native cannot be combined with "
                         "--greedy (the native tier samples)")
    # A standalone export is self-describing (meta.json): --quant then only
    # selects *which* preset, and the offline-state resolution below must be
    # skipped (there is no separate state file).
    standalone_preset = None
    if args.model_dir and is_standalone_dir(args.model_dir):
        standalone_preset = standalone_presets(args.model_dir)[0]
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
    # --quant only exists on the fast path: enable it instead of silently
    # running an invalid combination (quant would be ignored on eager).
    quant, w4_group, gptq_state = None, None, None
    if args.quant is not None and standalone_preset is None:
        cfg["fast"] = True
        if not args.fast:
            print("[moss_tts_lite] --quant implies --fast; CUDA-graph fast path "
                  "enabled")
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
        fast_native=bool(cfg.get("fast_native")))

    steps = stats.get("steps") or res.n_steps
    step_ms = stats.get("step_ms") or []
    avg = sum(step_ms) / len(step_ms) if step_ms else float("nan")
    print(f"[moss_tts_lite] wrote {args.output}: {len(wav)} samples "
          f"({len(wav) / sr:.2f}s @ {sr} Hz), steps={res.n_steps}, "
          f"finished={res.finished}, vram_strategy={strategy}, "
          f"prefill={stats.get('prefill_ms', float('nan')):.0f}ms, "
          f"decode avg={avg:.1f}ms/step ({1000 / avg:.1f} steps/s)")
    if getattr(res, "watchdog_triggered", False):
        ws = getattr(res, "watchdog_stats", {})
        print(f"[moss_tts_lite] WARNING: production watchdog triggered "
              f"({res.watchdog_reason}); forced audio end. stats={ws}")
    return 0
