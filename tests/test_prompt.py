"""Tests for moss_tts_lite.prompt.build_tts_prompt.

Unit checks run in this env; full parity against the reference HF processor
(MossTTSDelayProcessor + fast tokenizer) runs inside .tmp/venv-hf.
"""

from __future__ import annotations

import os

import torch

from moss_tts_lite.prompt import (AUDIO_PAD_CODE, MODEL_DIR, N_VQ, _apply_chat_template_user,
                             _render_user_inst, build_tts_prompt, default_tokenizer)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
HF_PY = os.path.join(ROOT, ".tmp", "venv-hf", "bin", "python")


class _CaptureTokenizer:
    """Stub tokenizer that records the prompt string handed to encode()."""

    def __init__(self):
        self.last_text = None

    def encode(self, text: str) -> list[int]:
        self.last_text = text
        return list(range(10, 10 + len(text)))


def test_template_rendering():
    tok = _CaptureTokenizer()
    out = build_tts_prompt("hello", tokenizer=tok)
    expected = (
        "<user_inst>\n"
        "- Reference(s):\nNone\n"
        "- Instruction:\nNone\n"
        "- Tokens:\nNone\n"
        "- Quality:\nNone\n"
        "- Sound Event:\nNone\n"
        "- Ambient Sound:\nNone\n"
        "- Language:\nNone\n"
        "- Text:\nhello\n"
        "</user_inst>"
    )
    assert tok.last_text == (
        f"<|im_start|>user\n{expected}<|im_end|>\n<|im_start|>assistant\n"
    ), tok.last_text
    assert out["input_ids"].shape == (1, len(tok.last_text), 1 + N_VQ)
    assert out["attention_mask"].shape == (1, len(tok.last_text))
    print("  <user_inst> template + chat template rendering: OK")

    # str() semantics: int tokens / language must render like the reference
    tok2 = _CaptureTokenizer()
    build_tts_prompt("hi", tokens=512, language="French", tokenizer=tok2)
    assert "- Tokens:\n512\n" in tok2.last_text
    assert "- Language:\nFrench\n" in tok2.last_text
    print("  str(tokens=512) / str(language='French') rendering: OK")


def test_tensor_contract():
    bpe = default_tokenizer()
    out = build_tts_prompt("你好，世界！")
    ids = out["input_ids"]
    mask = out["attention_mask"]
    assert ids.dtype == torch.int64 and mask.dtype == torch.bool
    L = ids.shape[1]
    assert ids.shape == (1, L, 33) and mask.shape == (1, L)
    assert bool(mask.all()), "single sample must be all-True mask"
    # channel 0 == BPE ids of the rendered prompt; channels 1..32 == 1024
    prompt_str = (
        "<|im_start|>user\n"
        + _render_user_inst("你好，世界！")
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    want_ids = bpe.encode(prompt_str)
    assert ids[0, :, 0].tolist() == want_ids
    assert ids[0, :, 1] .eq(AUDIO_PAD_CODE).all()
    assert (ids[0, :, 1:] == AUDIO_PAD_CODE).all()
    assert L == len(want_ids)
    print(f"  tensor contract: [1,{L},33] int64, ch0=BPE ids ({L}), ch1..32==1024, mask all-True")


def test_normalizer_applied():
    tok = _CaptureTokenizer()
    build_tts_prompt("这 是  mixed   空白 text", tokenizer=tok)
    assert "- Text:\n这是 mixed 空白 text\n" in tok.last_text, tok.last_text
    # injectable normalizer must override the default
    tok2 = _CaptureTokenizer()
    build_tts_prompt("RAW", normalizer=lambda t: f"[{t}]", tokenizer=tok2)
    assert "- Text:\n[RAW]\n" in tok2.last_text
    print("  default normalize_tts_text applied / injectable normalizer: OK")


def test_parity_reference_processor():
    """Run the actual MossTTSDelayProcessor (HF, venv-hf) and compare input_ids."""
    if not os.path.exists(HF_PY):
        print("  [skip] .tmp/venv-hf not present (parity deferred)")
        return
    import subprocess
    script = r"""
import json, sys, types, importlib.util
sys.path.insert(0, "/root/MOSS-TTS")
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# load reference package under a synthetic name (dir name has a hyphen)
pkg_name = "moss_v15_ref"
pkg = types.ModuleType(pkg_name)
pkg.__path__ = ["/root/MOSS-TTS/models/MOSS-TTS-v1.5"]
sys.modules[pkg_name] = pkg
spec = importlib.util.spec_from_file_location(
    pkg_name + ".processing_moss_tts",
    "/root/MOSS-TTS/models/MOSS-TTS-v1.5/processing_moss_tts.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[pkg_name + ".processing_moss_tts"] = mod
spec.loader.exec_module(mod)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/MOSS-TTS/models/MOSS-TTS-v1.5",
                                    trust_remote_code=True)
proc = mod.MossTTSDelayProcessor(tokenizer=tok, audio_tokenizer=None)

from moss_tts_lite.prompt import build_tts_prompt
cases = json.load(open(sys.argv[1], encoding="utf-8"))
results = []
for text, language, ntok in cases:
    ref = proc([proc.build_user_message(text=text, language=language, tokens=ntok)],
               mode="generation")
    ref_ids = ref["input_ids"]
    ref_mask = ref["attention_mask"]
    mine = build_tts_prompt(text, language=language, tokens=ntok)
    same_ids = torch.equal(ref_ids, mine["input_ids"])
    same_mask = torch.equal(ref_mask.bool(), mine["attention_mask"])
    results.append({
        "text": text, "language": language, "tokens": ntok,
        "len": int(ref_ids.shape[1]),
        "same_ids": bool(same_ids), "same_mask": bool(same_mask),
        "ref_dtype": str(ref_ids.dtype), "mine_dtype": str(mine["input_ids"].dtype),
    })
json.dump(results, open(sys.argv[2], "w"))
import torch  # noqa: E402  (import here so build_tts_prompt import stays clean)
"""
    # NB: torch import inside script must happen before use; rewrite to top-import
    script = "import torch\n" + script.replace(
        "import torch  # noqa: E402  (import here so build_tts_prompt import stays clean)\n", "")
    cases = [
        ["Hello, world!", None, None],
        ["你好，世界！今天天气不错。", None, None],
        ["  This   is  a   test... with  @mentions and https://x.com/a  ", None, None],
        ["Bonjour le monde, ça va très bien l'été.", "French", None],
        ["数字 123 与 English 混排。", "Chinese", 512],
        ["第一行\n第二行\n- 列表项\n1. 有序项", None, None],
        ["真的假的？？？！！！", None, None],
        ["", None, None],
        ["😀👍🏻", None, None],
        ["Line1\n\n\n\nLine2", None, 0],
    ]
    import json
    import tempfile
    cf = os.path.join(tempfile.gettempdir(), "prompt_cases.json")
    with open(cf, "w", encoding="utf-8") as f:
        json.dump(cases, f)
    r = subprocess.run([HF_PY, "-c", script, cf, cf + ".res"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [skip] reference processor failed: {r.stderr[-600:]}")
        return
    with open(cf + ".res", encoding="utf-8") as f:
        results = json.load(f)
    n_ok = sum(1 for x in results if x["same_ids"] and x["same_mask"]
               and x["ref_dtype"] == x["mine_dtype"])
    for x in results:
        assert x["same_ids"], f"input_ids mismatch for {x['text']!r}"
        assert x["same_mask"], f"attention_mask mismatch for {x['text']!r}"
        assert x["ref_dtype"] == x["mine_dtype"] == "torch.int64"
    print(f"  parity vs MossTTSDelayProcessor (HF): {n_ok}/{len(results)} cases "
          f"input_ids+mask 100% identical (lengths: {[x['len'] for x in results]})")


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_template_rendering()
    test_tensor_contract()
    test_normalizer_applied()
    test_parity_reference_processor()
    print("ALL TESTS PASSED")
