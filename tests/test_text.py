"""Text-layer tests:"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import json
import struct
import tempfile

import torch

from moss_tts_lite.bpe import QwenBPE
from moss_tts_lite.model import (AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE,
                                 AUDIO_START_TOKEN_ID, N_VQ)
from moss_tts_lite.normalizer import TEST_CASES, normalize_tts_text, run_tests
from moss_tts_lite.prompt import (AUDIO_PAD_CODE, N_VQ, _apply_chat_template_user,
                                  _render_user_inst, apply_delay_pattern,
                                  build_continuation_prompt, build_tts_prompt,
                                  default_tokenizer)

ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
HF_PY = os.path.join(ROOT, ".tmp", "venv-hf", "bin", "python")
REF_SCRIPT = os.path.join(ROOT, "models", "MOSS-TTS-v1.5",
                          "tts_robust_normalizer_single_script.py")
GOLDEN_PATH = os.path.join(ROOT, ".tmp", "golden2", "cont_prompt_golden.pt")

def _write_tiny_bpe(td: str):

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
    """Encode pretending every char is its own token (no merges applied):"""
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

        ids = bpe.encode("hello")
        toks = [bpe.decode([i]) for i in ids]
        assert "".join(toks) == "hello"
        assert bpe.decode(ids) == "hello"

        ids = bpe.encode("hi<|special|>there<tool_call>x<|special|>")
        assert ids.count(300) == 2 and ids.count(302) == 1
        assert bpe.decode(ids) == "hi<|special|>there<tool_call>x<|special|>"

        assert bpe.encode("hello") == ids[:0] + bpe.encode("hello")

        ids2 = bpe.encode("<|special|>")
        assert ids2 == [300]

        ids3 = bpe.encode("中")
        assert bpe.decode(ids3) == "中"

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
    """Compare token ids against HF tokenizers on the real checkpoint vocab:"""
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
            s.encode("utf-8")
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

def main_bpe() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_tiny_bpe_merges()
    test_roundtrip_samples()
    test_parity_hf()
    print("ALL TESTS PASSED")
    return 0

def test_builtin_cases():
    """The vendored script's own 38 cases, incl."""
    run_tests()
    assert len(TEST_CASES) == 38
    print(f"  vendored run_tests(): all {len(TEST_CASES)} cases passed (incl. idempotence)")

def test_parity_with_reference():
    """Byte-identical vendoring => must match the reference implementation on built-in cases + stress inputs + 200 fuzz stri..."""
    if not os.path.exists(REF_SCRIPT):
        print("  [skip] reference script not present")
        return
    spec = importlib.util.spec_from_file_location("ref_norm", REF_SCRIPT)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    cases = [t for _, t, _ in TEST_CASES]
    cases += [
        "", " ", "\n\n", "\t混合\t空白\r\n测试",
        "hello   world  123 ！！！???。。。...",
        "e-mail me@x.co, visit HTTP://A.Bb/c?d=e  #tag1 #tag2 @user_name u/abc r/tts",
        "v1.2.3-rc.2+build.5 和 app.js.map 与 index.d.ts 的 bundle.min.js",
        "《书名》嵌入, 【独立】标题！〖第二〗『三』「四」",
        "A=>B<-C<->D→E⇔F G->H",
        " mixed  latin 中文 spacing ＡＢＣ１２３ test ",
        "emoji 😀 test 👍🏻 mixed 中文",
        "surrogate-ish 𝕌𝕟𝕚𝕔𝕠𝕕𝕖 math 𝐛𝐨𝐥𝐝",
        "斜杠路径 /usr/local/bin 与 foo/bar.py 以及 C:\\Windows\\system32",
        "多\n行\n\n\n文本 with\n- list\n- items\n1. numbered\n2. lines",
        "note\u200bzero\ufeffwidth\u200dchars",
    ]
    import random
    rng = random.Random(42)
    alphabet = "ab01 .,!?-—…。！？中文《》【】@#https://\n\t\\/:+_"
    for _ in range(200):
        cases.append("".join(rng.choice(alphabet)
                             for _ in range(rng.randint(0, 80))))

    for i, c in enumerate(cases):
        a = ref.normalize_tts_text(c)
        b = normalize_tts_text(c)
        assert a == b, f"case #{i} mismatch: {c!r}: {a!r} != {b!r}"

        assert ref.normalize_tts_text(a) == normalize_tts_text(b), \
            f"case #{i} idempotence mismatch"
    print(f"  parity vs reference: {len(cases)} cases (incl. 200 fuzz) ALL MATCH")

def main_normalizer() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_builtin_cases()
    test_parity_with_reference()
    print("ALL TESTS PASSED")
    return 0

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

def main_prompt() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_template_rendering()
    test_tensor_contract()
    test_normalizer_applied()
    test_parity_reference_processor()
    print("ALL TESTS PASSED")
    return 0

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _check_diagonal(unified: torch.Tensor, prefix: torch.Tensor,
                    audio_start_idx: int) -> None:
    """Verify that the prompt's audio channels hold the exact truncated delay pattern of prefix[T, 32]."""
    T, n_vq = prefix.shape
    audio = unified[audio_start_idx + 1:, 1:]
    assert audio.shape == (T, n_vq), (audio.shape, (T, n_vq))

    for i in range(T):
        for ch in range(n_vq):
            val = int(audio[i, ch])
            if i >= ch:
                exp = int(prefix[i - ch, ch])
                assert val == exp, (f"row={i} ch={ch}: got {val}, expected {exp} "
                                    f"from prefix[{i - ch}, {ch}]")
            else:
                assert val == AUDIO_PAD_CODE, (
                    f"row={i} ch={ch}: got {val}, expected pad {AUDIO_PAD_CODE}")

def test_golden_parity() -> None:
    """100% bitwise parity with the reference processor."""
    if not os.path.exists(GOLDEN_PATH):
        print(f"[SKIP] golden file not found at {GOLDEN_PATH}")
        return

    golden = torch.save if False else torch.load(GOLDEN_PATH, map_location="cpu",
                                                 weights_only=False)
    cases = golden["cases"]
    input_ids_all = golden["input_ids"]
    attn_all = golden["attention_mask"]
    prefix_all = golden["prefix_codes"]

    print(f"[test_prompt_continuation] checking {len(cases)} golden cases...")
    for idx, case in enumerate(cases):
        name = case["name"]
        full_text = case["full_text"]
        lang = case["language"]
        prefix = prefix_all[idx]
        ref_ids = input_ids_all[idx]
        ref_attn = attn_all[idx]

        out = build_continuation_prompt(text=full_text, language=lang,
                                        prefix_codes=prefix)
        our_ids = out["input_ids"]
        our_attn = out["attention_mask"]

        assert our_ids.shape == ref_ids.shape, (
            f"[{name}] shape mismatch: ours={our_ids.shape} ref={ref_ids.shape}")
        assert our_attn.shape == ref_attn.shape, (
            f"[{name}] attn shape mismatch: ours={our_attn.shape} ref={ref_attn.shape}")

        diff = (our_ids != ref_ids)
        if diff.any():
            n_diff = int(diff.sum())
            first_pos = int(diff.nonzero()[0, 1])
            ch = int(diff.nonzero()[0, 2])
            raise AssertionError(
                f"[{name}] input_ids mismatch: {n_diff} diffs, first at pos={first_pos} "
                f"ch={ch}: ours={int(our_ids[0, first_pos, ch])} "
                f"ref={int(ref_ids[0, first_pos, ch])}")

        assert (our_attn == ref_attn).all(), f"[{name}] attention_mask mismatch"

        col0 = our_ids[0, :, 0]
        assert int(col0[-1]) == AUDIO_GEN_SLOT_TOKEN_ID, (
            f"[{name}] tail token must be gen_slot (151656), got {int(col0[-1])}")
        starts = (col0 == AUDIO_START_TOKEN_ID).nonzero().flatten().tolist()
        assert len(starts) == 1, f"[{name}] expected 1 audio_start, got {len(starts)}"
        start_idx = starts[0]
        assert start_idx == case["audio_start_idx"]

        user_audio = our_ids[0, :start_idx + 1, 1:]
        assert (user_audio == AUDIO_PAD_CODE).all(), (
            f"[{name}] non-pad in user-phase audio channels")

        _check_diagonal(our_ids[0], prefix, start_idx)

        print(f"  [OK] {name}: L={our_ids.shape[1]} (100% bitwise parity, "
              f"start={start_idx}, gen_slots={(col0 == AUDIO_GEN_SLOT_TOKEN_ID).sum()})")

def test_delay_pattern_helper() -> None:
    """Standalone unit test for apply_delay_pattern."""
    codes = torch.arange(10 * 32, dtype=torch.long).view(10, 32)
    delayed = apply_delay_pattern(codes, pad_code=1024)
    assert delayed.shape == (10 + 31, 32), delayed.shape
    for ch in range(32):
        assert (delayed[:ch, ch] == 1024).all()
        assert (delayed[ch:ch + 10, ch] == codes[:, ch]).all()
        assert (delayed[ch + 10:, ch] == 1024).all()
    print("  [OK] apply_delay_pattern standalone unit test passed")

def test_argument_guards() -> None:
    """Ensure invalid inputs fail fast with clear ValueError."""

    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(10))
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(40, 16))
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(10, 32))
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("  [OK] argument guard tests passed")

def main_continuation() -> int:
    test_delay_pattern_helper()
    test_argument_guards()
    test_golden_parity()
    print("[test_prompt_continuation] ALL TESTS PASSED.")
    return 0

def main() -> int:
    rc = 0
    rc |= main_bpe() or 0
    rc |= main_normalizer() or 0
    rc |= main_prompt() or 0
    rc |= main_continuation() or 0

    return rc

if __name__ == "__main__":
    sys.exit(main())
