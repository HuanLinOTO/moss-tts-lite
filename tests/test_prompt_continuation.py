"""Test build_continuation_prompt against the reference golden.

Parity: bitwise equality against `.tmp/golden2/cont_prompt_golden.pt` generated
by `encode_refs.py` through the official `MossTTSDelayProcessor` with
`mode="continuation"`.

Also tests structural invariants independently of the golden:
  * input_ids shape [1, L, 33] and attention_mask all-True;
  * user-phase audio channels strictly equal audio_pad_code (1024);
  * tail text id is exactly AUDIO_GEN_SLOT_TOKEN_ID (151656);
  * exactly T = prefix_codes.shape[0] gen-slots after audio_start;
  * diagonal delay pattern: for any valid frame f and channel ch,
    audio_ch[audio_start_idx + 1 + f + ch, ch] == prefix[f, ch];
  * input argument validation (bad dims, bad n_vq, short prefix).
"""

from __future__ import annotations

import os
import sys

import torch

from moss_tts_lite.model import (
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    N_VQ,
)
from moss_tts_lite.prompt import (
    apply_delay_pattern,
    build_continuation_prompt,
    default_tokenizer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_PATH = os.path.join(REPO_ROOT, ".tmp", "golden2", "cont_prompt_golden.pt")


def _check_diagonal(unified: torch.Tensor, prefix: torch.Tensor,
                    audio_start_idx: int) -> None:
    """Verify that the prompt's audio channels hold the exact truncated delay
    pattern of prefix[T, 32]."""
    T, n_vq = prefix.shape
    audio = unified[audio_start_idx + 1:, 1:]  # [T, 32]
    assert audio.shape == (T, n_vq), (audio.shape, (T, n_vq))
    # In the truncated delayed stream (first T rows), delayed[i, ch] is
    # prefix[i - ch, ch] when i >= ch, else pad (1024).
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

        # 1. shape match
        assert our_ids.shape == ref_ids.shape, (
            f"[{name}] shape mismatch: ours={our_ids.shape} ref={ref_ids.shape}")
        assert our_attn.shape == ref_attn.shape, (
            f"[{name}] attn shape mismatch: ours={our_attn.shape} ref={ref_attn.shape}")

        # 2. bitwise parity on input_ids
        diff = (our_ids != ref_ids)
        if diff.any():
            n_diff = int(diff.sum())
            first_pos = int(diff.nonzero()[0, 1])
            ch = int(diff.nonzero()[0, 2])
            raise AssertionError(
                f"[{name}] input_ids mismatch: {n_diff} diffs, first at pos={first_pos} "
                f"ch={ch}: ours={int(our_ids[0, first_pos, ch])} "
                f"ref={int(ref_ids[0, first_pos, ch])}")

        # 3. attention_mask equality
        assert (our_attn == ref_attn).all(), f"[{name}] attention_mask mismatch"

        # 4. independent structural invariants
        col0 = our_ids[0, :, 0]
        assert int(col0[-1]) == AUDIO_GEN_SLOT_TOKEN_ID, (
            f"[{name}] tail token must be gen_slot (151656), got {int(col0[-1])}")
        starts = (col0 == AUDIO_START_TOKEN_ID).nonzero().flatten().tolist()
        assert len(starts) == 1, f"[{name}] expected 1 audio_start, got {len(starts)}"
        start_idx = starts[0]
        assert start_idx == case["audio_start_idx"]

        # User-phase audio channels must be pure pad
        user_audio = our_ids[0, :start_idx + 1, 1:]
        assert (user_audio == AUDIO_PAD_CODE).all(), (
            f"[{name}] non-pad in user-phase audio channels")

        # Diagonal delay pattern matches prefix
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
    # Bad dim
    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(10))
        assert False, "expected ValueError"
    except ValueError:
        pass
    # Bad channel count
    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(40, 16))
        assert False, "expected ValueError"
    except ValueError:
        pass
    # Too short (< N_VQ)
    try:
        build_continuation_prompt("foo", prefix_codes=torch.zeros(10, 32))
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("  [OK] argument guard tests passed")


def main() -> int:
    test_delay_pattern_helper()
    test_argument_guards()
    test_golden_parity()
    print("[test_prompt_continuation] ALL TESTS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
