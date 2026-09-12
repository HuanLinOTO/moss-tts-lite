"""Tests for moss_tts_lite.codec (MossCodecDecoder).

Run:  python3 -m moss_tts_lite.tests.test_codec
GPU required (~4 GB fp32); wrap with flock .tmp/gpu.lock when sharing.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from moss_tts_lite.codec import MossCodecDecoder  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer")
GOLDEN_WAV = os.path.join(ROOT, ".tmp", "golden", "codec_golden.wav")
GOLDEN_CODES = os.path.join(ROOT, ".tmp", "golden", "codec_golden_codes.pt")
GEN_GOLDEN = os.path.join(ROOT, ".tmp", "golden", "gen_golden.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _decoder() -> MossCodecDecoder:
    return MossCodecDecoder(MODEL_DIR, device=DEVICE, dtype=torch.float32)


def test_weight_norm_matches_torch():
    """Our WN-conv weight reconstruction must equal torch's parametrization."""
    import torch.nn as nn

    conv = nn.Conv1d(8, 512, kernel_size=1, bias=True)
    nn.utils.parametrizations.weight_norm(conv)
    with torch.no_grad():
        orig0 = conv.parametrizations.weight.original0.clone()
        orig1 = conv.parametrizations.weight.original1.clone()
    from moss_tts_lite.codec import _denorm_wn_conv

    w = _denorm_wn_conv(orig0.cpu(), orig1.cpu())
    ref = conv.weight.detach().cpu()
    assert torch.allclose(w, ref, atol=0, rtol=1e-6), (w - ref).abs().max().item()
    print("test_weight_norm_matches_torch: OK (max|d| = %.2e)" % (w - ref).abs().max().item())


def test_random_codes_shapes_and_sanity():
    """Random codes [T,32] for T in {50,125,500} -> length T*1920, finite, sane amplitude."""
    dec = _decoder()
    for T in (50, 125, 500):
        g = torch.Generator().manual_seed(1000 + T)
        codes = torch.randint(0, 1024, (T, 32), generator=g)
        wav = dec.decode(codes)  # chunked (reference pipeline semantics)
        assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
        assert wav.shape == (T * 1920,), f"T={T}: {wav.shape} != {(T * 1920,)}"
        assert np.isfinite(wav).all(), f"T={T}: non-finite samples"
        peak = float(np.abs(wav).max())
        rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
        assert 1e-5 < peak < 1e3, f"T={T}: peak {peak}"
        print(f"test_random_codes T={T}: len={len(wav)} peak={peak:.4f} rms={rms:.6f}")

    # full-sequence path must also run and produce identical length
    g = torch.Generator().manual_seed(2000)
    codes = torch.randint(0, 1024, (60, 32), generator=g)
    wav = dec.decode(codes, chunk_duration=None)
    assert wav.shape == (60 * 1920,) and np.isfinite(wav).all()
    print("test_random_codes full path: OK")


def test_chunked_vs_full_small():
    """For T <= 100 (one chunk), chunked and full paths are the exact same math."""
    dec = _decoder()
    g = torch.Generator().manual_seed(3000)
    codes = torch.randint(0, 1024, (80, 32), generator=g)
    a = dec.decode(codes, chunk_duration=8.0)
    b = dec.decode(codes, chunk_duration=None)
    assert a.shape == b.shape
    diff = np.abs(a - b).max()
    assert diff == 0.0, f"expected bitwise equality for T<=100, got {diff}"
    print("test_chunked_vs_full_small: OK (max|d| = 0)")


def test_parity_golden():
    """Parity vs reference HF decode (.tmp/golden). Skipped if golden missing."""
    if not (os.path.exists(GOLDEN_WAV) and os.path.exists(GOLDEN_CODES)):
        print("test_parity_golden: SKIPPED (golden assets not present yet)")
        return
    import soundfile as sf

    ref, sr = sf.read(GOLDEN_WAV, dtype="float32")
    assert sr == 24000 and ref.ndim == 1

    codes = torch.load(GOLDEN_CODES, map_location="cpu", weights_only=True)
    if isinstance(codes, dict):
        codes = codes["codes_NQ_T"]          # asset format: [32, T]
    codes = torch.as_tensor(codes).long().cpu()
    if codes.dim() == 3:
        codes = codes[:, 0]
    if codes.shape[0] == 32 and codes.shape[-1] != 32:
        codes = codes.t()                     # -> [T, 32]
    assert codes.shape[1] == 32, codes.shape

    dec = _decoder()
    wav = dec.decode(codes)  # default chunk_duration=8, matching the reference call
    n = min(len(ref), len(wav))
    tail = (len(ref), len(wav))
    d = ref[:n].astype(np.float64) - wav[:n].astype(np.float64)
    snr = 10 * np.log10((ref[:n].astype(np.float64) ** 2).sum() / max((d ** 2).sum(), 1e-30))
    print(f"test_parity_golden: n={n} lens(ref,out)={tail} SNR={snr:.2f} dB max|d|={np.abs(d).max():.3e}")
    assert snr >= 30.0, f"SNR {snr:.2f} dB < 30 dB"


def test_streamer_matches_decode():
    """Incremental streamer must reproduce decode(codes, chunk_duration=8) bitwise."""
    dec = _decoder()
    g = torch.Generator().manual_seed(4000)
    codes = torch.randint(0, 1024, (455, 32), generator=g)  # 4.55 blocks of 100 frames
    ref = dec.decode(codes, chunk_duration=8.0)

    from moss_tts_lite.codec import MossCodecStreamer
    s = MossCodecStreamer(dec, chunk_duration=8.0)
    outs = []
    for i in range(0, 455, 7):                       # irregular push sizes
        w = s.push(codes[i : i + 7])
        if w is not None:
            outs.append(w)
    outs.append(s.flush())
    wav = np.concatenate(outs)
    assert wav.shape == ref.shape, (wav.shape, ref.shape)
    assert np.array_equal(wav, ref), np.abs(wav - ref).max()
    print("test_streamer_matches_decode: OK (bitwise identical, pushed in chunks of 7)")


def test_delayed_rows_to_segments_golden():
    """gen_golden zh case: [137,32] delay rows -> 3 pad separators dropped -> [103,32]."""
    from moss_tts_lite.codec import delayed_rows_to_segments

    gold = torch.load(GOLDEN_CODES, map_location="cpu", weights_only=True)
    delayed = gold["codes_delayed_T32_raw"]                      # [137, 32] raw delay rows
    assert delayed.shape == (137, 32)
    segs = delayed_rows_to_segments(delayed)
    assert len(segs) == gold["n_segments"] == 1
    assert segs[0].shape == (103, 32), segs[0].shape
    assert torch.equal(segs[0], gold["codes_first_segment_T32"])
    # exactly the codes whose decode is the golden wav (transposed [32, 103])
    assert torch.equal(segs[0].t().contiguous(), gold["codes_NQ_T"])

    # cross-check against gen_golden.pt source of truth
    gen = torch.load(GEN_GOLDEN, map_location="cpu", weights_only=True)
    assert torch.equal(delayed, gen["cases"][0]["audio_codes"])
    print("test_delayed_rows_to_segments_golden: OK (137 rows -> 1 segment [103,32])")


def test_delayed_rows_to_segments_roundtrip():
    """Multi-segment roundtrip through the reference delay pattern."""
    from moss_tts_lite.codec import delayed_rows_to_segments

    def apply_delay_pattern(codes, pad_code):     # reference replica (test-only)
        t, n = codes.shape
        out = torch.full((t + n - 1, n), pad_code, dtype=codes.dtype)
        for i in range(n):
            out[i : i + t, i] = codes[:, i]
        return out

    pad = torch.full((1, 4), 1024, dtype=torch.long)
    s1 = (torch.arange(20).reshape(5, 4) * 13 + 1) % 1024
    s2 = (torch.arange(12).reshape(3, 4) * 29 + 7) % 1024
    delayed = apply_delay_pattern(torch.cat([s1, pad, s2], dim=0), 1024)
    segs = delayed_rows_to_segments(delayed)
    assert len(segs) == 2
    assert torch.equal(segs[0], s1) and torch.equal(segs[1], s2)
    print("test_delayed_rows_to_segments_roundtrip: OK (2 segments recovered exactly)")


def test_delayed_rows_to_segments_edges():
    from moss_tts_lite.codec import delayed_rows_to_segments

    def apply_delay_pattern(codes, pad_code):     # reference replica (test-only)
        t, n = codes.shape
        out = torch.full((t + n - 1, n), pad_code, dtype=codes.dtype)
        for i in range(n):
            out[i : i + t, i] = codes[:, i]
        return out

    # no separator -> exactly one segment (delay roundtrip)
    codes = (torch.arange(40).reshape(10, 4) * 11 + 3) % 1024
    (segs,) = delayed_rows_to_segments(apply_delay_pattern(codes, 1024))
    assert torch.equal(segs, codes)

    # all rows pad -> empty list
    assert delayed_rows_to_segments(torch.full((36, 32), 1024, dtype=torch.long)) == []

    # three segments with separators -> three segments back (multi-break sizes)
    s1 = (torch.arange(16).reshape(4, 4) * 13 + 1) % 1024
    s2 = (torch.arange(12).reshape(3, 4) * 29 + 7) % 1024
    s3 = (torch.arange(20).reshape(5, 4) * 47 + 11) % 1024
    pad = torch.full((1, 4), 1024, dtype=torch.long)
    delayed = apply_delay_pattern(torch.cat([s1, pad, s2, pad, s3], dim=0), 1024)
    segs = delayed_rows_to_segments(delayed)
    assert len(segs) == 3
    assert torch.equal(segs[0], s1) and torch.equal(segs[1], s2) and torch.equal(segs[2], s3)

    # T < n_vq -> explicit error
    try:
        delayed_rows_to_segments(torch.full((10, 32), 5, dtype=torch.long))
        raise AssertionError("expected ValueError for T < n_vq")
    except ValueError:
        pass

    # en case from gen_golden: segments are contiguous, none empty, no pad rows inside
    gen = torch.load(GEN_GOLDEN, map_location="cpu", weights_only=True)
    segs = delayed_rows_to_segments(gen["cases"][1]["audio_codes"])  # [174, 32]
    assert len(segs) >= 1 and all(s.shape[0] > 0 for s in segs)
    assert all(not bool((s == 1024).all(dim=1).any()) for s in segs)
    print(f"test_delayed_rows_to_segments_edges: OK (en case -> {len(segs)} segment(s), "
          f"rows={[s.shape[0] for s in segs]})")


def main():
    test_weight_norm_matches_torch()
    test_random_codes_shapes_and_sanity()
    test_chunked_vs_full_small()
    test_parity_golden()
    test_streamer_matches_decode()
    test_delayed_rows_to_segments_golden()
    test_delayed_rows_to_segments_roundtrip()
    test_delayed_rows_to_segments_edges()
    print("ALL CODEC TESTS PASSED")


if __name__ == "__main__":
    main()
