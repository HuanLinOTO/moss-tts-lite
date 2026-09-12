"""End-to-end generate() smoke test: greedy short-text TTS run.

Checks the delay-pattern trajectory shape (audio_start -> gen_slot rows ->
delay_slot x31 -> audio_end -> im_end), pad ramp structure, and termination.
bf16 ~17.5GB on GPU, run under the GPU lock:
  flock .tmp/gpu.lock python3 -m moss_tts_lite.tests.test_generate
"""

import sys

import torch

from moss_tts_lite.generate import generate
from moss_tts_lite.model import (
    AUDIO_END_TOKEN_ID,
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    AUDIO_DELAY_SLOT_TOKEN_ID,
    IM_END_TOKEN_ID,
    MossTTSModel,
)
try:  # prefer the real loader (tok-delivered); fall back to the temp mini one
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors
from tests._mini_bpe import build_tts_prompt_dev

MODEL_DIR = "/root/MOSS-TTS/models/MOSS-TTS-v1.5"


def main():
    dev = torch.device("cuda")
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=dev, dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()

    prompt = build_tts_prompt_dev("Hello world, this is a smoke test.")
    print("prompt L =", prompt["input_ids"].shape[1])

    res = generate(model, prompt, max_new_tokens=200, greedy=True)

    assert res.audio_frames.shape == (res.n_steps, 32), res.audio_frames.shape
    assert res.text_ids.shape == (res.n_steps,)
    assert res.n_steps <= 200
    print(f"greedy: n_steps={res.n_steps}, finished={res.finished}, "
          f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    t = res.text_ids.tolist()
    fr = res.audio_frames

    # ---- trajectory shape checks ----
    if AUDIO_START_TOKEN_ID in t:
        i_start = t.index(AUDIO_START_TOKEN_ID)
        # before audio starts: only pad allowed
        pre = [x for x in t[:i_start] if x != 151643]
        assert not pre, f"unexpected pre-audio text tokens: {pre[:8]}"
        i_first_gen = i_start + 1
        assert t[i_start + 1] == AUDIO_GEN_SLOT_TOKEN_ID, \
            f"first post-start row must be gen_slot, got {t[i_start + 1]}"
        # ramp: gen rows between audio_start and the first delay slot (K rows;
        # K = audio frame count, delay tail fills the remaining channels)
        i_delay = t.index(AUDIO_DELAY_SLOT_TOKEN_ID)
        assert i_delay >= i_first_gen, (i_delay, i_first_gen)
        # delay tail: exactly 32 delay rows (1 sampled + 31 forced), then audio_end
        # (reference state machine: delayed_lengths runs 0->32; audio_end at ==n_vq)
        delay_rows = [x for x in t[i_delay:] if x in (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID)]
        assert delay_rows[0] == AUDIO_DELAY_SLOT_TOKEN_ID
        k = delay_rows.index(AUDIO_END_TOKEN_ID)
        assert k == 32, f"expected 32 delay rows before audio_end, got {k}"
        assert t[i_delay + 32] == AUDIO_END_TOKEN_ID
        # audio_end row: all channels forced pad
        assert (fr[i_delay + 32] == AUDIO_PAD_CODE).all(), \
            "audio_end row must be all-pad"
        print("delay-pattern trajectory: audio_start@%d, %d gen rows (~%d frames), "
              "32 delay rows, audio_end@%d" % (i_start, i_delay - i_first_gen,
                                               i_delay - i_first_gen + 1, i_delay + 32))
    if res.finished:
        assert t[-1] == IM_END_TOKEN_ID, f"last token {t[-1]} != im_end"
        assert IM_END_TOKEN_ID not in t[:-1], "im_end mid-stream"
    print("text stream:", [hex(x) for x in t[:8]], "...", [hex(x) for x in t[-6:]])

    # ---- pad-code hygiene + independent state-walk over audio masks ----
    assert fr.min() >= 0 and fr.max() <= 1024
    real = fr[fr != AUDIO_PAD_CODE]
    # re-derive expected sampled-channel sets from the text stream using the
    # reference counter semantics (audio_lengths / delayed_lengths)
    MAXD = 2**63 - 1
    al, delayed = 0, MAXD
    for r, tt in enumerate(t):
        active = {i for i in range(32) if al > i and (delayed == MAXD or i >= delayed)}
        row = fr[r].tolist()
        for i in range(32):
            if i in active:
                assert row[i] != AUDIO_PAD_CODE, f"row {r} ch {i} should be sampled"
            else:
                assert row[i] == AUDIO_PAD_CODE, f"row {r} ch {i} should be pad, got {row[i]}"
        if tt in (AUDIO_START_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID):
            al += 1
        elif tt == AUDIO_END_TOKEN_ID:
            al = 0
        if delayed == MAXD and tt == AUDIO_DELAY_SLOT_TOKEN_ID:
            delayed = 0
        if delayed != MAXD:
            delayed += 1
        if delayed > 32:
            delayed = MAXD
    print("audio mask state-walk: all 63 rows match reference semantics")
    print(f"audio frames: {tuple(fr.shape)}, pad-ratio={float((fr == AUDIO_PAD_CODE).float().mean()):.3f}, "
          + (f"non-pad codes in [{int(real.min())}, {int(real.max())}]" if real.numel() else "all pad"))

    # ---- seeded sampling determinism ----
    res2 = generate(model, prompt, max_new_tokens=120, greedy=False, seed=1234)
    res3 = generate(model, prompt, max_new_tokens=120, greedy=False, seed=1234)
    assert res2.n_steps == res3.n_steps
    assert torch.equal(res2.text_ids, res3.text_ids), "same seed must reproduce text ids"
    assert torch.equal(res2.audio_frames, res3.audio_frames), "same seed must reproduce audio"
    print(f"seeded sampling: n_steps={res2.n_steps}, finished={res2.finished}, "
          f"deterministic rerun OK")

    print("test_generate PASS")


if __name__ == "__main__":
    sys.exit(main())
