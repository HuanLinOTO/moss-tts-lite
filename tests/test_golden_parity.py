"""Golden parity: moss_tts_lite BPE + prompt vs asset-produced golden files.

  .tmp/golden/tok_golden.json    {"lines": [...], "ids": [[int,...],...]}
      produced with HF fast tokenizer `tok.encode(line)` on tokenizer_cases.txt
  .tmp/golden/prompt_golden.pt   {"cases": [...], "input_ids": [T[1,L,33]],
                                  "attention_mask": [T[1,L]]}
      produced with the reference MossTTSDelayProcessor (audio_tokenizer=None)

Acceptance: token ids 100% identical; prompt input_ids 100% identical.
"""

from __future__ import annotations

import json
import os

import torch

from moss_tts_lite.bpe import QwenBPE
from moss_tts_lite.prompt import build_tts_prompt

ROOT = os.environ.get("MOSS_TTS_ROOT", os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")


def test_tok_golden():
    path = os.path.join(GOLDEN, "tok_golden.json")
    if not os.path.exists(path):
        print("  [pending] .tmp/golden/tok_golden.json not present")
        return
    with open(path, encoding="utf-8") as f:
        golden = json.load(f)
    lines, ref_ids = golden["lines"], golden["ids"]
    bpe = QwenBPE(os.path.join(MODEL_DIR, "vocab.json"),
                  os.path.join(MODEL_DIR, "merges.txt"),
                  os.path.join(MODEL_DIR, "tokenizer.json"))
    n_bad = 0
    for i, (line, want) in enumerate(zip(lines, ref_ids)):
        got = bpe.encode(line)
        if got != want:
            n_bad += 1
            print(f"  tok MISMATCH line {i}: {line[:50]!r}")
            if n_bad <= 5:
                for j, (a, b) in enumerate(zip(got, want)):
                    if a != b:
                        print(f"    first diff @{j}: mine={a} hf={b}")
                        break
                else:
                    print(f"    length mine={len(got)} hf={len(want)}")
    assert n_bad == 0, f"{n_bad}/{len(lines)} tokenizer golden cases mismatch"
    print(f"  tok_golden: {len(lines)}/{len(lines)} lines 100% identical "
          f"({sum(len(x) for x in ref_ids)} tokens)")


def test_prompt_golden():
    path = os.path.join(GOLDEN, "prompt_golden.pt")
    if not os.path.exists(path):
        print("  [pending] .tmp/golden/prompt_golden.pt not present")
        return
    golden = torch.load(path, map_location="cpu", weights_only=True)
    n_ok = 0
    for case, ref_ids, ref_mask in zip(golden["cases"], golden["input_ids"],
                                       golden["attention_mask"]):
        kwargs = case["kwargs"]
        mine = build_tts_prompt(kwargs["text"],
                                language=kwargs.get("language"),
                                tokens=kwargs.get("tokens"))
        same_ids = torch.equal(ref_ids, mine["input_ids"])
        same_mask = torch.equal(ref_mask, mine["attention_mask"])
        assert same_ids, (
            f"prompt input_ids mismatch for case {case['name']}: "
            f"L_ref={ref_ids.shape[1]} L_mine={mine['input_ids'].shape[1]}")
        assert same_mask, f"attention_mask mismatch for case {case['name']}"
        n_ok += 1
        print(f"  prompt[{case['name']}]: L={ref_ids.shape[1]} "
              f"input_ids identical, mask identical")
    print(f"  prompt_golden: {n_ok}/{len(golden['cases'])} cases 100% identical")


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_tok_golden()
    test_prompt_golden()
    print("ALL TESTS PASSED")
