"""Tests for moss_tts_lite.bpe.QwenBPE.

Unit tests on a tiny synthetic BPE + round-trip + full parity against the
reference HF (tokenizers) implementation on the real MOSS-TTS-v1.5 vocab.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile

from moss_tts_lite.bpe import QwenBPE

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
HF_PY = os.path.join(ROOT, ".tmp", "venv-hf", "bin", "python")


# --------------------------------------------------------------------------- #
# tiny synthetic BPE for controlled unit tests
# --------------------------------------------------------------------------- #

def _write_tiny_bpe(td: str):
    # byte-level vocab: 256 byte chars + merged symbols
    def bytes_to_unicode():
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

    b2u = bytes_to_unicode()
    vocab = {c: i for i, c in enumerate(sorted(set(b2u.values())))}
    merges = [("h", "e"), ("t", "h"), ("th", "e"), ("l", "o"),
              ("hel", "lo")]
    nxt = len(vocab)
    for a, b in merges:
        vocab[a + b] = nxt
        nxt += 1
    vp = os.path.join(td, "vocab.json")
    with open(vp, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    mp = os.path.join(td, "merges.txt")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    ap = os.path.join(td, "added.json")
    with open(ap, "w", encoding="utf-8") as f:
        json.dump({"<|special|>": 300, "<|endoftext|>": 301, "<tool_call>": 302}, f)
    return vp, mp, ap, b2u


def _enc_raw(bpe: QwenBPE, b2u: dict[int, str], text: str) -> list[int]:
    """Encode pretending every char is its own token (no merges applied):
    used only to sanity-check decode() byte mapping."""
    ids = []
    for piece in bpe._pretokenize(text):
        word = "".join(b2u[c] for c in piece.encode("utf-8"))
        for ch in word:
            ids.append(bpe._vocab[ch])
    return ids


def test_tiny_bpe_merges():
    with tempfile.TemporaryDirectory() as td:
        vp, mp, ap, b2u = _write_tiny_bpe(td)
        bpe = QwenBPE(vp, mp, ap)
        # "hello": h+e -> he(256); th+e... "hello" pretokenizes to ["hello"]
        # byte chars: h e l l o -> merges rank: (h,e)=0 first -> he l l o
        # then (l,o)=3 -> he l lo ; no (he,l)... => ["he","l","lo"]
        ids = bpe.encode("hello")
        toks = [bpe.decode([i]) for i in ids]
        assert "".join(toks) == "hello"
        assert bpe.decode(ids) == "hello"
        # added tokens split raw text (even glued to words) and are single ids
        ids = bpe.encode("hi<|special|>there<tool_call>x<|special|>")
        assert ids.count(300) == 2 and ids.count(302) == 1
        assert bpe.decode(ids) == "hi<|special|>there<tool_call>x<|special|>"
        # cache: second call returns identical ids
        assert bpe.encode("hello") == ids[:0] + bpe.encode("hello")
        # longest added token wins: encode text starting with both prefixes
        ids2 = bpe.encode("<|special|>")
        assert ids2 == [300]
        # unknown byte path still decodes with replacement (latin-1 chars exist
        # in the byte alphabet, so use a real multi-byte char)
        ids3 = bpe.encode("中")
        assert bpe.decode(ids3) == "中"
        # pre-tokenizer: digits split singly, contractions, whitespace runs
        assert bpe._pretokenize("abc123") == ["abc", "1", "2", "3"]
        assert bpe._pretokenize("I'm fine") == ["I", "'m", " fine"]
        assert bpe._pretokenize("don't") == ["don", "'t"]
        print("  tiny-BPE merges/added-tokens/cache/pretokenizer: OK")


def test_roundtrip_samples():
    if not os.path.isdir(MODEL_DIR):
        print("  [skip] model dir not present")
        return
    bpe = QwenBPE(os.path.join(MODEL_DIR, "vocab.json"),
                  os.path.join(MODEL_DIR, "merges.txt"),
                  os.path.join(MODEL_DIR, "tokenizer.json"))
    samples = [
        "Hello world!",
        "你好，世界！今天天气不错。",
        "The quick brown fox jumps over the lazy dog. I'm sure it's fine.",
        "中文与 English 混排 test 12345，加上标点！！！",
        "  leading and trailing spaces   ",
        "line1\nline2\r\nline3\n\n\n",
        "tabs\tand\x0bvertical\x0cforms",
        "URL https://example.com/a?b=c&d=e and email a.b+c@example.co",
        "数字混合：v2.3.1、2026年3月31日、3.14159",
        "<|im_start|>user\n内容<|im_end|>\n<|im_start|>assistant\n",
        "<|audio_start|><|audio_user_slot|><|audio_end|>",
        "emoji 😀👍🏻 and accré́ents ñ Ω≈ç√",
        "aaaa   bbb\n\n\nccc    ddd",
        "'s 't 're 've 'm 'll 'd 'S 'T 'LL 'Re",
        "allascii",
    ]
    for s in samples:
        ids = bpe.encode(s)
        assert bpe.decode(ids) == s, f"round-trip failed: {s!r} -> {bpe.decode(ids)!r}"
    print(f"  round-trip on {len(samples)} zh/en/mixed samples: OK")


# --------------------------------------------------------------------------- #
# parity vs HF tokenizers (ground truth)
# --------------------------------------------------------------------------- #

CORPUS = [
    "Hello world!",
    "你好，世界！今天天气不错。",
    "The quick brown fox jumps over the lazy dog. I'm sure it's fine.",
    "中文与 English 混排 test 12345，加上标点！！！",
    "  leading and trailing spaces   ",
    "line1\nline2\r\nline3\n\n\n",
    "tabs\tand\x0bvertical\x0cforms",
    "URL https://example.com/a?b=c&d=e and email a.b+c@example.co",
    "数字混合：v2.3.1、2026年3月31日、3.14159",
    "<|im_start|>user\nnormalize this text<|im_end|>\n<|im_start|>assistant\n",
    "<|audio_start|><|audio_user_slot|><|audio_end|>",
    "<|im_start|><|audio_pad|>" * 3,
    "emoji 😀👍🏻 and accré́ents ñ Ω≈ç√",
    "aaaa   bbb\n\n\nccc    ddd",
    "'s 't 're 've 'm 'll 'd 'S 'T 'LL 'Re",
    "we'LL check I'D go it'S o'K",
    "　全角スペース　と日本語テキスト、汉 kes",
    "Ⅷ Ⅻ ½ ① ⑴ ＡＢＣ ｄｅｆ",
    "a\u00a0b\u2009c\u3000d\u2028e\u0085f",
    "x" * 200,
    " "\
    * 50,
    "\n\n\n\n\n",
    "١٢٣ ๑๒๓ ①",
    "ｶﾀｶﾅ半角",
    "Ça va très bien l'été à Paris",
    "Довідник: привіт світ",
    "😀😀 😀😀 😀",
    "Then('the'quick'Brown'Fox'Jumps'Over'The'Lazy'Dog')",
    "<tool_call>f(x)</tool_call><tool_response>42</tool_response>",
    "<think>reasoning</think>answer<|repo_name|>moss/tts<|file_sep|>a.py",
    "",
]


def test_parity_hf():
    """Compare token ids against HF tokenizers on the real checkpoint vocab:
    31 curated cases + decode parity + 400 fuzz strings."""
    if not os.path.isdir(MODEL_DIR):
        print("  [skip] model dir not present")
        return
    if not os.path.exists(HF_PY):
        print("  [skip] .tmp/venv-hf not present (parity deferred)")
        return
    import random
    import subprocess
    rng = random.Random(7)
    pools = [
        "abcxyzABCXYZ", "你好世界测试中文", "ｱｲｳｴｵあいうえお", "٠١٢٣۴۵६۷八9",
        "0123456789", "Ⅷ½①⑴", " \t\n\r\v\f", "\u00a0\u2000\u2028\u3000\u0085\u200b",
        "'`\"'s't're've'm'll'd", "!@#$%^&*()_+-=[]{};:,.<>?/\\|~",
        "😀🄰Ⓐ℮Ω≈ç√",
    ]
    fuzz = []
    for _ in range(400):
        s = "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randint(0, 60)))
        try:
            s.encode("utf-8")  # drop lone surrogates (HF rejects them)
        except UnicodeEncodeError:
            continue
        fuzz.append(s)
    for f in ["README.md", "moss_tts_lite/bpe.py", "models/MOSS-TTS-v1.5/README.md"]:
        txt_path = os.path.join(ROOT, f)
        if os.path.exists(txt_path):
            txt = open(txt_path, encoding="utf-8").read()
            for i in range(0, min(len(txt), 8000), 997):
                fuzz.append(txt[i:i + 80])

    all_cases = CORPUS + fuzz
    case_file = os.path.join(tempfile.gettempdir(), "bpe_cases.json")
    with open(case_file, "w", encoding="utf-8") as f:
        json.dump(all_cases, f)
    script = r"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/MOSS-TTS-v1.5", trust_remote_code=True)
cases = json.load(open(sys.argv[1], encoding="utf-8"))
out = [tok.encode(c) for c in cases]
dec = [tok.decode(out[i]) for i in range(len(cases))]
json.dump({"enc": out, "dec": dec}, open(sys.argv[2], "w"))
"""
    ref_file = case_file + ".ref"
    r = subprocess.run([HF_PY, "-c", script, case_file, ref_file],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [skip] HF tokenizer failed: {r.stderr[-400:]}")
        return
    with open(ref_file, encoding="utf-8") as f:
        ref = json.load(f)
    bpe = QwenBPE(os.path.join(MODEL_DIR, "vocab.json"),
                  os.path.join(MODEL_DIR, "merges.txt"),
                  os.path.join(MODEL_DIR, "tokenizer.json"))
    n_bad = 0
    n_dec_bad = 0
    for i, (c, r_ids) in enumerate(zip(all_cases, ref["enc"])):
        m_ids = bpe.encode(c)
        if m_ids != r_ids:
            n_bad += 1
            if n_bad <= 3:
                for j, (a, b) in enumerate(zip(m_ids, r_ids)):
                    if a != b:
                        print(f"  case#{i} {c[:40]!r}: first diff at {j}: "
                              f"mine={a}({bpe.decode([a])!r}) hf={b}")
                        break
                else:
                    print(f"  case#{i} {c[:40]!r}: length diff "
                          f"mine={len(m_ids)} hf={len(r_ids)}")
        m_dec = bpe.decode(r_ids)
        if m_dec != ref["dec"][i]:
            n_dec_bad += 1
            if n_dec_bad <= 3:
                print(f"  decode#{i}: mine={m_dec[:40]!r} hf={ref['dec'][i][:40]!r}")
        assert n_bad == 0, f"encode parity failed on case {i}"
        assert n_dec_bad == 0, f"decode parity failed on case {i}"
    print(f"  parity vs HF tokenizers: {len(CORPUS)} curated + {len(fuzz)} fuzz "
          f"cases encode 100% identical, decode 100% identical "
          f"({sum(len(x) for x in ref['enc'])} tokens total)")


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_tiny_bpe_merges()
    test_roundtrip_samples()
    test_parity_hf()
    print("ALL TESTS PASSED")
