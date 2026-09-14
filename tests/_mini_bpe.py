"""TEMPORARY prompt builder for tts-agent self-tests (smoke only).

Byte-level BPE (Qwen2 pre-tokenizer semantics, ASCII classes) over
vocab.json + merges.txt, wrapping the exact UserMessage template + chat
template used by processing_moss_tts.build_user_message + apply_chat_template
(generation prompt).  Only valid for pure-ASCII smoke texts; the real
moss_tts_lite.prompt (tok) + golden assets replace this.
"""

import json
import os
import re
from pathlib import Path

import torch

MODEL_DIR = os.environ.get(
    "MOSS_TTS_ROOT", "/root/MOSS-TTS") + "/models/MOSS-TTS-v1.5"

IM_START = 151644
IM_END = 151645
AUDIO_PAD_CODE = 1024

# Qwen2 pre-tokenizer, ASCII classes (no unicode letters/digits in smoke texts)
_PRETOK = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\nA-Za-z0-9]?[A-Za-z]+"
    r"|[0-9]{1,3}"
    r"| ?[^\sA-Za-z0-9]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class _MiniBPE:
    def __init__(self, model_dir=MODEL_DIR):
        self.vocab = json.loads((Path(model_dir) / "vocab.json").read_text())
        ranks = {}
        for i, line in enumerate((Path(model_dir) / "merges.txt").read_text().splitlines()):
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            ranks[(a, b)] = i
        self.ranks = ranks
        self.byte_enc = _bytes_to_unicode()

    def _bpe(self, piece: str):
        word = tuple(self.byte_enc[b] for b in piece.encode("utf-8"))
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            first, second = best
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        return [self.vocab[p] for p in word]

    def encode(self, text: str):
        ids = []
        parts = re.split(r"(<\|im_start\|>|<\|im_end\|>)", text)
        for part in parts:
            if not part:
                continue
            if part == "<|im_start|>":
                ids.append(IM_START)
            elif part == "<|im_end|>":
                ids.append(IM_END)
            else:
                for m in _PRETOK.findall(part):
                    ids.extend(self._bpe(m))
        return ids


def build_tts_prompt_dev(text: str, model_dir=MODEL_DIR):
    """Direct-TTS generation prompt: UserMessage(None fields) + chat template
    + <|im_start|>assistant\\n.  Returns {"input_ids": [1,L,33], "attention_mask": [1,L]}."""
    bpe = _MiniBPE(model_dir)
    user_inst = ("<user_inst>\n- Reference(s):\nNone\n- Instruction:\nNone\n- Tokens:\nNone\n"
                 "- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n"
                 f"- Language:\nNone\n- Text:\n{text}\n</user_inst>")
    full = "<|im_start|>user\n" + user_inst + "<|im_end|>\n<|im_start|>assistant\n"
    ids = bpe.encode(full)
    L = len(ids)
    input_ids = torch.empty(1, L, 33, dtype=torch.long)
    input_ids[0, :, 0] = torch.tensor(ids, dtype=torch.long)
    input_ids[0, :, 1:] = AUDIO_PAD_CODE
    return {"input_ids": input_ids,
            "attention_mask": torch.ones(1, L, dtype=torch.bool)}


if __name__ == "__main__":
    bpe = _MiniBPE()
    assert bpe.encode("Hello") == [9707], bpe.encode("Hello")
    assert bpe.encode(" world") == [1879], bpe.encode(" world")
    assert bpe.encode("assistant") == [77091]
    p = build_tts_prompt_dev("Hello world, this is a smoke test.")
    ids = p["input_ids"][0, :, 0].tolist()
    assert ids[0] == IM_START and ids[1] == 872 and ids[2] == 198
    assert ids[-1] == 198 and ids[-3] == IM_START
    print("mini-BPE self-check OK; prompt L =", len(ids))
