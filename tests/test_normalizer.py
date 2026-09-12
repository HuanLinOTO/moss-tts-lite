"""Tests for moss_tts_lite.normalizer (vendored TTS robust normalizer)."""

from __future__ import annotations

import importlib.util
import os

from moss_tts_lite.normalizer import TEST_CASES, normalize_tts_text, run_tests

REF_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..",
                          "models", "MOSS-TTS-v1.5",
                          "tts_robust_normalizer_single_script.py")


def test_builtin_cases():
    """The vendored script's own 38 cases, incl. idempotence checks."""
    run_tests()
    assert len(TEST_CASES) == 38
    print(f"  vendored run_tests(): all {len(TEST_CASES)} cases passed (incl. idempotence)")


def test_parity_with_reference():
    """Byte-identical vendoring => must match the reference implementation on
    built-in cases + stress inputs + 200 fuzz strings (and their re-normalization)."""
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
        # idempotence parity
        assert ref.normalize_tts_text(a) == normalize_tts_text(b), \
            f"case #{i} idempotence mismatch"
    print(f"  parity vs reference: {len(cases)} cases (incl. 200 fuzz) ALL MATCH")


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_builtin_cases()
    test_parity_with_reference()
    print("ALL TESTS PASSED")
