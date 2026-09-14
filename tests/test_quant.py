"""Quantization tests:"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from moss_tts_lite.codec import MossCodecDecoder, delayed_rows_to_segments
from moss_tts_lite.fast import FastMossTTS, _LIN_NAMES
from moss_tts_lite.fast import FastMossTTS, generate_fast
from moss_tts_lite.gptq import (SLOT_LINEARS, SLOT_NAMES, CapturingMossTTS, GptqMossTTS,
                    damped_cholesky_inverse, effective_weight, gptq_quantize,
                    pack_fast, rtn_quantize, slot_errors)
from moss_tts_lite.model import (AUDIO_PAD_CODE, N_VQ, MossTTSModel, _rms_norm,
                     _rotate_half)

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors

ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
CODEC_DIR = os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
OUT = os.path.join(ROOT, ".tmp", "perf_agent")

DEV = torch.device("cuda")
G = 128

def _rand_weight(n, k, scale=0.05, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=gen) * scale

    w = w * (1.0 + 0.5 * torch.sin(torch.arange(k, dtype=torch.float32) / 97.0))
    return w.to(torch.bfloat16).to(DEV)

def _check_int4pack_shapes(m) -> None:
    """`_convert_weight_to_int4pack(inner_k_tiles=8)` requires K/2 % 128 == 0 (i."""
    for li, lyr in enumerate(m.layers):
        for name in _LIN_NAMES:
            k, n = int(lyr[name].shape[1]), int(lyr[name].shape[0])
            assert k % 256 == 0, f"L{li} {name}: K={k} invalid for int4pack"
            assert n % 8 == 0, f"L{li} {name}: N={n} invalid for int4pack"

def _fake_backbone(n_layers=2, hidden=256, n_heads=8, head_dim=32,
                   text_vocab=64, seed=0):
    """Synthetic `MossTTSModel` skeleton:"""
    shapes = {"q": (n_heads * head_dim, hidden), "k": (2 * head_dim, hidden),
              "v": (2 * head_dim, hidden), "o": (hidden, n_heads * head_dim),
              "gate": (hidden * 2, hidden), "up": (hidden * 2, hidden),
              "down": (hidden, hidden * 2)}
    layers = []
    for li in range(n_layers):
        layers.append({name: _rand_weight(*shapes[name], seed=100 * li + i)
                       for i, name in enumerate(_LIN_NAMES)})
    m = SimpleNamespace(device=DEV, dtype=torch.bfloat16, n_layers=n_layers,
                        hidden_size=hidden, n_heads=n_heads, head_dim=head_dim,
                        text_vocab=text_vocab, layers=layers)
    _check_int4pack_shapes(m)
    return m

def phase_g1_format_parity():
    print("=== G1: bit-level format parity with FastMossTTS._quantize ===")
    ok = True
    for li, seed in enumerate((11, 12, 13)):
        m = _fake_backbone(n_layers=1, seed=seed)

        fast = FastMossTTS(m, quant="w4", w4_group_size=G)
        m2 = _fake_backbone(n_layers=1, seed=seed)
        for name in _LIN_NAMES:
            packed_ref, qsz_ref = fast.qlayers[0][name]
            w = m2.layers[0][name]
            q, s, mn = rtn_quantize(w, G)
            packed, qsz = pack_fast(q, s, mn, fast.inner_k_tiles)
            same_p = torch.equal(packed_ref, packed)
            same_s = torch.equal(qsz_ref, qsz)
            print(f"  case{li} {name:5s} packed_equal={same_p} qsz_equal={same_s}")
            ok = ok and same_p and same_s
            assert same_p and same_s, f"format mismatch on {name} (case {li})"
        del fast
        torch.cuda.empty_cache()
    print(f"  G1: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g2_kernel_semantics():
    print("=== G2: kernel output vs reference dequantization ===")
    torch.manual_seed(0)
    ok = True
    for (n, k) in ((512, 512), (1024, 4096)):
        w = _rand_weight(n, k, seed=n)
        q, s, mn = rtn_quantize(w, G)
        packed, qsz = pack_fast(q, s, mn, 8)
        w_eff = effective_weight(q, qsz, G)
        km, x = 8, torch.randn(8, k, dtype=torch.bfloat16, device=DEV)
        y = torch._weight_int4pack_mm(x, packed, G, qsz)
        y_ref = (x.float() @ w_eff.t())
        rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()

        scale = qsz[:, :, 0].t().float().reshape(n, k // G, 1)
        zero = qsz[:, :, 1].t().float().reshape(n, k // G, 1)
        d2 = ((q.float().reshape(n, k // G, G) - 8.0) * scale + zero).reshape(n, k)
        dmax = (d2 - w_eff).abs().max().item()
        wmax = w_eff.abs().max().item()
        print(f"  N={n} K={k}: kernel-vs-reference rel_err={rel:.2e}  "
              f"dequant identity max|d|={dmax:.3e} (|w|max={wmax:.3f})")
        ok = ok and rel < 5e-3 and dmax == 0.0
    print(f"  G2: {'PASS' if ok else 'FAIL'}")
    return ok

def _correlated_activations(n_tok, k, seed=0, device=DEV):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(n_tok, k, generator=gen)

    mix = torch.randn(k, 64, generator=gen) / 8.0
    x = base + base @ mix @ mix.t()
    x = x * torch.exp(torch.linspace(0, 1.2, k)).unsqueeze(0)
    return x.float().to(device)

def _objective(w_orig, w_hat, H):
    d = (w_hat - w_orig.float())
    return float((d @ H @ d.t()).diagonal().sum().item())

def phase_g3_core():
    print("=== G3: GPTQ core (objective, block-laziness, code sanity) ===")
    ok = True
    n, k, n_tok = 256, 512, 900
    x = _correlated_activations(n_tok, k, seed=3)
    H = x.t() @ x
    w = _rand_weight(n, k, scale=0.08, seed=7)

    q_rtn, s_rtn, mn_rtn = rtn_quantize(w, G)
    _, qsz_rtn = pack_fast(q_rtn, s_rtn, mn_rtn, 8)
    w_rtn = effective_weight(q_rtn, qsz_rtn, G)

    res = gptq_quantize(w, H, group_size=G, block_size=G, damp_percent=0.01)
    w_q = res.dequant()
    obj_rtn = _objective(w, w_rtn, H)
    obj_q = _objective(w, w_q, H)
    print(f"  objective: RTN={obj_rtn:.4e}  GPTQ={obj_q:.4e}  "
          f"ratio={obj_q / obj_rtn:.3f}")
    ok = ok and obj_q < obj_rtn * 0.9

    errs = slot_errors({"cal": x}, w, w_q)
    direct = float(((x @ (w_q - w.float()).t()) ** 2).mean().item())
    print(f"  slot_errors mse={errs['cal']['mse']:.6e}  direct={direct:.6e}  "
          f"rel={errs['cal']['rel']:.3e}")
    ok = ok and abs(errs["cal"]["mse"] - direct) <= 1e-9 + 1e-6 * abs(direct)

    assert res.q.min().item() >= 0 and res.q.max().item() <= 15
    print(f"  codes in [{int(res.q.min())},{int(res.q.max())}] "
          f"dtype={res.q.dtype} packed_dtype={res.packed.dtype} "
          f"qsz{tuple(res.qsz.shape)}/{res.qsz.dtype}")

    res_flat = gptq_quantize(w, H, group_size=G, block_size=k, damp_percent=0.01,
                             scale_from="original")
    res_block = gptq_quantize(w, H, group_size=G, block_size=G, damp_percent=0.01,
                              scale_from="original")
    dif = int((res_flat.q != res_block.q).sum().item())
    frac = dif / res_flat.q.numel()
    print(f"  lazy-block(128) vs single-block codes differing: {dif}/{res_flat.q.numel()}"
          f" ({frac:.2e})  [fp32 summation-order only]")
    ok = ok and frac <= 1e-3

    res_o = gptq_quantize(w, H, group_size=G, block_size=G, damp_percent=0.01,
                          scale_from="original")
    for tag, r in (("compensated", res), ("original", res_o)):
        e = slot_errors({"cal": x}, w, r.dequant())["cal"]
        print(f"  scale_from={tag:11s} mse={e['mse']:.6e} rel={e['rel']:.4e}")
        ok = ok and e["mse"] < errs["cal"]["mse"] * 5.0
    print(f"  G3: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g4_injection():
    print("=== G4: GptqMossTTS injection (incl. mixed-precision branch) ===")
    m = _fake_backbone(n_layers=2, seed=21)

    state = {}
    keep = {1}
    for li in range(m.n_layers):
        state[li] = {}
        for name in _LIN_NAMES:
            w = m.layers[li][name]
            q, s, mn = rtn_quantize(w, G)
            packed, qsz = pack_fast(q, s, mn, 8)
            state[li][name] = {"packed": packed.cpu(), "qsz": qsz.cpu()}
    m_ref = _fake_backbone(n_layers=2, seed=21)
    w_keep = m.layers[1]["q"].clone()
    fast_rtn = FastMossTTS(m_ref, quant="w4", w4_group_size=G)

    fast = GptqMossTTS(m, state, w4_group_size=G, bf16_keep=keep)
    ok = True
    for li in range(2):
        for name in _LIN_NAMES:
            if li in keep:
                same = fast.qlayers[li] is None
                print(f"  L{li} {name:5s} bf16_keep -> qlayers None: {same}")
                ok = ok and same
                continue
            p_ref, q_ref = fast_rtn.qlayers[li][name]
            p_my, q_my = fast.qlayers[li][name]
            same = torch.equal(p_ref, p_my) and torch.equal(q_ref, q_my)
            ok = ok and same
            if not same:
                print(f"  L{li} {name}: MISMATCH vs FastMossTTS(quant='w4')")
    print(f"  qlayers identical to FastMossTTS(quant='w4'): {ok}")

    x = torch.randn(3, m.hidden_size, dtype=torch.bfloat16, device=DEV)
    y_keep = fast._linear(x, 1, "q")
    y_ref = F.linear(x, w_keep)
    eq_bf16 = torch.equal(y_keep, y_ref)
    print(f"  bf16_keep layer path == F.linear: {eq_bf16}")
    ok = ok and eq_bf16

    p_my, q_my = fast.qlayers[0]["q"]
    y_kernel = torch._weight_int4pack_mm(x, p_my, G, q_my)
    y_lin = fast._linear(x, 0, "q")
    eq_k = torch.equal(y_kernel, y_lin)
    print(f"  quantized path == _weight_int4pack_mm: {eq_k}")
    ok = ok and eq_k
    print(f"  G4: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g5_purity():
    print("=== G5: gptq.py dependency purity ===")
    import ast
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "moss_tts_lite" / "gptq.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add((node.module or "").split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"torch", "numpy", "soundfile", "yaml"}
    bad = sorted(m for m in mods if m not in allowed)
    print(f"  imported modules: {sorted(mods)}")
    ok = not bad
    if bad:
        print(f"  VIOLATION: {bad}")
    print(f"  G5: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g7_shared_hinv():
    """The shared (slot-level) inverse and the CPU-factorisation path must give bitwise-identical codes to the per-linear GP..."""
    print("=== G7: shared Hinv + inverse_device equivalence ===")
    n, k, n_tok = 192, 512, 700
    x = _correlated_activations(n_tok, k, seed=17)
    H = x.t() @ x
    w = _rand_weight(n, k, scale=0.08, seed=23)
    ok = True

    h_gpu, d_gpu = damped_cholesky_inverse(H, 0.01)
    h_cpu, d_cpu = damped_cholesky_inverse(H.to("cpu"), 0.01,
                                           work_device="cpu")
    dmax = (h_gpu.cpu() - h_cpu).abs().max().item()
    scale = h_gpu.abs().max().item()
    rel = dmax / scale
    print(f"  CPU vs GPU inverse: rel|d|={rel:.3e} (max|d|={dmax:.3e}, "
          f"damp {d_gpu:.4e}/{d_cpu:.4e}) — LAPACK vs cuSOLVER roundoff")
    ok = ok and rel < 1e-5

    r_ref = gptq_quantize(w, H, group_size=G, block_size=G)
    r_shr = gptq_quantize(w, group_size=G, block_size=G, Hinv=h_gpu.clone())
    q_same = torch.equal(r_shr.q, r_ref.q)
    sz_same = torch.equal(r_shr.qsz, r_ref.qsz)
    print(f"  shared-Hinv (GPU) == per-linear inverse: codes {q_same}, "
          f"qsz {sz_same}")
    ok = ok and q_same and sz_same

    r_cpu = gptq_quantize(w, group_size=G, block_size=G, Hinv=h_cpu)
    flip = int((r_cpu.q != r_ref.q).sum().item()) / r_ref.q.numel()
    o_ref = _objective(w, r_ref.dequant(), H)
    o_cpu = _objective(w, r_cpu.dequant(), H)
    print(f"  shared-Hinv (CPU): code flips {flip * 100:.3f}%  "
          f"objective {o_cpu:.6e} vs {o_ref:.6e} "
          f"(rel {abs(o_cpu - o_ref) / o_ref:.2e})")
    ok = ok and flip < 1e-3 and abs(o_cpu - o_ref) / o_ref < 1e-2

    eye = torch.eye(k, device=H.device)
    Hd = H + d_gpu * eye
    resid = (h_gpu.t() @ h_gpu @ Hd - eye).abs().max().item()
    print(f"  ||Hinv^T Hinv H_damped - I||_max = {resid:.3e}")
    ok = ok and resid < 1e-2
    print(f"  G7: {'PASS' if ok else 'FAIL'}")
    return ok

def _mini_weights(hidden=64, head_dim=16, n_heads=4, n_kv=2, inter=96,
                  vocab=151700, n_layers=2, seed=5):
    """Synthetic checkpoint dict shaped like MOSS-TTS-v1."""
    gen = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    w = {"language_model.embed_tokens.weight": r(vocab, hidden),
         "language_model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16)}
    for li in range(n_layers):
        p = f"language_model.layers.{li}"
        w[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        w[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        w[f"{p}.self_attn.q_proj.weight"] = r(n_heads * head_dim, hidden)
        w[f"{p}.self_attn.k_proj.weight"] = r(n_kv * head_dim, hidden)
        w[f"{p}.self_attn.v_proj.weight"] = r(n_kv * head_dim, hidden)
        w[f"{p}.self_attn.o_proj.weight"] = r(hidden, n_heads * head_dim)
        w[f"{p}.self_attn.q_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
        w[f"{p}.self_attn.k_norm.weight"] = torch.ones(head_dim, dtype=torch.bfloat16)
        w[f"{p}.mlp.gate_proj.weight"] = r(inter, hidden)
        w[f"{p}.mlp.up_proj.weight"] = r(inter, hidden)
        w[f"{p}.mlp.down_proj.weight"] = r(hidden, inter)
    for i in range(N_VQ):
        w[f"emb_ext.{i}.weight"] = r(AUDIO_PAD_CODE + 1, hidden)
    w["lm_heads.0.weight"] = r(vocab, hidden)
    for i in range(N_VQ):
        w[f"lm_heads.{i + 1}.weight"] = r(AUDIO_PAD_CODE + 1, hidden)
    return w

def phase_g6_capture_tap():
    """CapturingMossTTS must be numerics-identical to MossTTSModel, and the captured slots must be the exact tensors the lin..."""
    print("=== G6: CapturingMossTTS tap parity (CPU mini model) ===")
    w = _mini_weights()
    base = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=256)
    cap = CapturingMossTTS(w, device="cpu", dtype=torch.bfloat16, max_seq_len=256)
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(0, 40, (1, 7, N_VQ + 1), generator=g)
    ids[..., 1:] = torch.randint(0, AUDIO_PAD_CODE + 1, (1, 7, N_VQ), generator=g)
    ok = True
    with torch.inference_mode():
        hb = base.prefill(ids).last_hidden
        hc = cap.prefill(ids).last_hidden
        eq = torch.equal(hb, hc)
        print(f"  prefill bitwise equal: {eq}  layers tapped: {len(cap.cap_slots)}")
        ok = ok and eq

        h = cap._embed(ids)
        for li in range(cap.n_layers):
            prev = h
            h = cap._layer(li, h, 0)
            sl = cap.cap_slots[li]
            lyr = cap.layers[li]
            xin_ref = _rms_norm(prev, lyr["input_layernorm"])
            mid = prev + F.linear(sl["attn_out"], lyr["o"])
            min_ref = _rms_norm(mid, lyr["post_attention_layernorm"])
            down_ref = F.silu(F.linear(min_ref, lyr["gate"])) \
                * F.linear(min_ref, lyr["up"])
            li_ok = (torch.equal(xin_ref, sl["attn_in"])
                     and torch.equal(min_ref, sl["mlp_in"])
                     and torch.equal(down_ref, sl["down_in"]))
            print(f"  L{li}: attn_in/mlp_in/down_in slots == consumers' inputs: {li_ok}")
            ok = ok and li_ok
        row = ids[0, -1].view(1, 1, N_VQ + 1)
        for _t in range(3):
            hb = base.step(row).last_hidden
            hc = cap.step(row).last_hidden
            ok = ok and torch.equal(hb, hc)
        print(f"  3 decode steps bitwise equal: {ok}")
        shapes = {s: tuple(cap.cap_slots[li][s].shape[-1:]) for s in SLOT_NAMES}
        print(f"  slots present: {sorted(cap.cap_slots[1])} {shapes}")
    print(f"  G6: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g8_mixed_dispatch():
    """Per-linear bf16 keep and per-linear group sizes must dispatch to exactly the intended kernel/F."""
    print("=== G8: mixed-precision / mixed-group dispatch ===")
    torch.manual_seed(0)
    m = _fake_backbone(n_layers=2, seed=31)
    state = {}
    gmap = {}
    for li in range(2):
        state[li] = {}
        for name in _LIN_NAMES:
            for g in (128, 32):
                q, sc, mn = rtn_quantize(m.layers[li][name], g)
                packed, qsz = pack_fast(q, sc, mn, 8)
                state[li].setdefault(name, {})[g] = (packed, qsz)

    keep = {(1, "down"), (1, "o")}
    gmap_pairs = {(0, "down"): 32, (1, "q"): 32}
    sel = {li: {name: gmap_pairs.get((li, name), 128) for name in _LIN_NAMES}
           for li in range(2)}
    base_state = {li: {n: {"packed": state[li][n][sel[li][n]][0],
                           "qsz": state[li][n][sel[li][n]][1]}
                       for n in _LIN_NAMES} for li in range(2)}

    w_bf16 = {li: {n: m.layers[li][n].clone() for n in _LIN_NAMES}
              for li in range(2)}
    fast = GptqMossTTS(m, base_state, w4_group_size=128, bf16_keep=keep,
                       group_size_map=gmap_pairs)
    ok = True
    print(f"  bf16_keep_linears={fast.bf16_keep_linears} "
          f"mixed_groups={ {f'{a}:{b}': g for (a, b), g in fast.mixed_groups().items()} }")
    for li in range(2):
        for name in _LIN_NAMES:

            x = torch.randn(5, int(w_bf16[li][name].shape[1]),
                            dtype=torch.bfloat16, device=DEV)
            y = fast._linear(x, li, name)
            if (li, name) in keep:
                ref = F.linear(x, w_bf16[li][name])
                kind = "F.linear(bf16)"
            else:
                g = sel[li][name]
                ref = torch._weight_int4pack_mm(
                    x, state[li][name][g][0], g, state[li][name][g][1])
                kind = f"int4pack(g={g})"
                if ref.dtype != m.dtype:
                    ref = ref.to(m.dtype)
            same = torch.equal(y, ref)
            wpk = fast.qlayers[li] is not None and name in fast.qlayers[li]
            print(f"  L{li} {name:5s} -> {kind:18s} bitwise={same} "
                  f"quantized={wpk}")
            ok = ok and same
    print(f"  G8: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_g9_group_sizes():
    """Which int4 group sizes does this build's kernel accept? The deployment kernel fixes group_size at runtime (`w4_group_..."""
    print("=== G9: kernel-supported int4 group sizes ===")
    if not torch.cuda.is_available():
        print("  G9: SKIP (no CUDA)")
        return True
    k = 4096
    n = 256
    w = (torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02)
    x = torch.randn(4, k, dtype=torch.bfloat16, device=DEV)
    ok_sizes = []
    for g in (16, 32, 64, 128, 256):
        if k % g:
            print(f"  g={g:4d}: skipped (does not divide K)")
            continue
        try:
            q, s, mn = rtn_quantize(w, g)
            packed, qsz = pack_fast(q, s, mn, 8)
            y = torch._weight_int4pack_mm(x, packed, g, qsz)

            wq = effective_weight(q, qsz, g).to(torch.bfloat16)
            ref = F.linear(x, wq)
            rel = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
            ok_sizes.append(g)
            print(f"  g={g:4d}: OK  qsz{tuple(qsz.shape)} "
                  f"kernel-vs-dequant rel={rel.item():.2e}")
        except Exception as e:                               # noqa: BLE001
            print(f"  g={g:4d}: FAIL {type(e).__name__}: {str(e)[:70]}")
    print(f"  supported: {ok_sizes}")
    print("  G9: PASS" if ok_sizes else "  G9: FAIL")
    return bool(ok_sizes)

def main_gptq() -> int:
    if not torch.cuda.is_available():
        print("CUDA required (int4pack kernel is CUDA-only)")
        return 1
    results = {
        "G1_format": phase_g1_format_parity(),
        "G2_kernel": phase_g2_kernel_semantics(),
        "G3_core": phase_g3_core(),
        "G4_injection": phase_g4_injection(),
        "G5_purity": phase_g5_purity(),
        "G6_capture": phase_g6_capture_tap(),
        "G7_shared_inv": phase_g7_shared_hinv(),
        "G8_mixed": phase_g8_mixed_dispatch(),
        "G9_groups": phase_g9_group_sizes(),
    }
    ok = all(results.values())
    summary = " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items())
    print(f"test_gptq: {summary} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors

SEED = 1234
SR = 24000
AUDIO_FPS = 12.5

def _steps_ps(step_ms, skip=0):
    ms = step_ms[skip:]
    return 1000.0 / (sum(ms) / len(ms)) if ms else float("nan")

def phase_a(fast, zh_ids):
    """Teacher-forced 40 steps:"""
    print("=== Phase A: teacher-forced 40-step (quantized forward, M4 gates) ===")
    z = np.load(os.path.join(GOLDEN, "logits_golden.npz"))
    gt_text = torch.from_numpy(z["logits_text"]).cuda()
    gt_audio = torch.from_numpy(z["logits_audio"]).cuda()
    gt_rows = torch.from_numpy(z["selected_rows"]).reshape(-1, 33).cuda()
    model = fast.m
    mine_text, mine_audio = [], []
    with torch.inference_mode():
        hs = fast.prefill(zh_ids)
        L0 = int(model.seq_len)
        fast.capture()
        h = hs.last_hidden[:, -1]
        mine_text.append(model.text_logits(h).float())
        mine_audio.append(model.audio_logits(h).float())
        for t in range(1, gt_rows.shape[0]):
            lt, la, h, _two = fast.step(gt_rows[t - 1].view(1, 33), L0 + t - 1)
            mine_text.append(lt.float())
            mine_audio.append(la.float())
    mine_text = torch.stack(mine_text)
    mine_audio = torch.stack(mine_audio)[..., :1024]
    gt_a = gt_audio[..., :1024]

    ok_t = (mine_text.argmax(-1) == gt_text.argmax(-1)).float().mean().item() * 100
    d_t = (mine_text - gt_text).abs().max().item()

    top25 = mine_audio.topk(25, dim=-1).indices
    gold1 = gt_a.argmax(-1).unsqueeze(-1)
    mem = (top25 == gold1).any(-1).float().mean().item() * 100

    ok_a_all = (mine_audio.argmax(-1) == gold1.squeeze(-1)).float().mean().item() * 100
    top2g = gt_a.topk(2, dim=-1).values
    gap = top2g[..., 0] - top2g[..., 1]
    tie_share = (gap == 0).float().mean().item() * 100
    nontie = gap > 0
    ok_a_nt = (((mine_audio.argmax(-1) == gold1.squeeze(-1)) & nontie).sum().item()
               / max(int(nontie.sum()), 1) * 100)
    d_a = (mine_audio - gt_a).abs().max().item()

    print(f"  logits_text : max|d|={d_t:.4f}  argmax agree={ok_t:.2f}%   "
          f"gate>=99: {'PASS' if ok_t >= 99 else 'FAIL'}")
    print(f"  logits_audio: max|d|={d_a:.4f}")
    print(f"  audio top-25 membership (golden top1 in mine top25) = {mem:.2f}%   "
          f"gate>=95: {'PASS' if mem >= 95 else 'FAIL'}")
    print(f"  audio argmax (report-only): all={ok_a_all:.2f}%  "
          f"non-tie={ok_a_nt:.2f}%  tie-share={tie_share:.2f}%")
    print(f"  gate (text argmax >=99% hard; top25 report-only per "
          f"supervisor M4 decision): {'PASS' if ok_t >= 99.0 else 'FAIL'}")
    ok = ok_t >= 99.0
    return ok, ok_t, mem, ok_a_all, ok_a_nt, tie_share

def _gen_wav(fast, ids, mask, out_path, label):
    model = fast.m
    res = generate_fast(fast, {"input_ids": ids.cuda(), "attention_mask": mask.cuda()},
                        max_new_tokens=4096, seed=SEED)
    codec = MossCodecDecoder(CODEC_DIR, device=torch.device("cuda"))
    segments = delayed_rows_to_segments(res.audio_frames)
    wavs = [codec.decode(seg, chunk_duration=8.0) for seg in segments]
    wav = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
    sf.write(out_path, wav, SR, subtype="FLOAT")
    fin = bool(np.isfinite(wav).all())
    dur = len(wav) / SR
    nonpad = (res.audio_frames != 1024).float().mean().item() * 100
    print(f"  [{label}] steps={res.n_steps} finished={res.finished} "
          f"segments={len(segments)} frames T={res.audio_frames.shape[0]} "
          f"non-pad codes={nonpad:.1f}%  wav={dur:.2f}s finite={fin}")
    del codec
    torch.cuda.empty_cache()
    ok = res.finished and res.audio_frames.shape[0] > 0 and fin and dur > 1.0
    return ok, dur, res

def main_fast_m4() -> int:
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)

    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    fast = FastMossTTS(model, quant="w4")
    torch.cuda.empty_cache()
    g = fast.w4_group_size
    print(f"after W4(g{g}) quantize: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB "
          f"(bf16 backbone linears freed, int4-packed + qsz resident)")

    zh_i = [c["name"] for c in pg["cases"]].index("zh_plain")
    en_i = [c["name"] for c in pg["cases"]].index("en_language")

    ok_a, ok_t, mem, ok_a_all, ok_a_nt, tie_share = phase_a(
        fast, pg["input_ids"][zh_i].cuda())

    print("=== Phase B: E2E zh/en (seed=1234, default sampling) ===")
    ok_zh, dur_zh, res_zh = _gen_wav(fast, pg["input_ids"][zh_i],
                                     pg["attention_mask"][zh_i],
                                     os.path.join(OUT, f"m4_zh_g{g}.wav"), "zh")
    ok_en, dur_en, res_en = _gen_wav(fast, pg["input_ids"][en_i],
                                     pg["attention_mask"][en_i],
                                     os.path.join(OUT, f"m4_en_g{g}.wav"), "en")

    print("=== Phase C: speed (zh) ===")
    stats: dict = {}
    torch.cuda.reset_peak_memory_stats()
    res_s = generate_fast(fast, {"input_ids": pg["input_ids"][zh_i].cuda(),
                                 "attention_mask": pg["attention_mask"][zh_i].cuda()},
                          max_new_tokens=4096, seed=SEED, stats=stats)
    step_ms = stats.get("step_ms") or []
    sps = _steps_ps(step_ms, 10)
    print(f"decode avg: {sum(step_ms) / len(step_ms):.2f} ms/step over "
          f"{len(step_ms)} steps -> {_steps_ps(step_ms):.1f} steps/s")
    print(f"decode steady (10+): {sps:.1f} steps/s "
          f"({sum(step_ms[10:]) / len(step_ms[10:]):.2f} ms/step)")
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"VRAM peak (decode only): {vram:.2f} GiB")

    rtf = (sum(step_ms[10:]) / 1000.0) / (len(step_ms[10:]) / AUDIO_FPS)
    print(f"decode RTF: {rtf:.3f} (audio seconds per decode second, steady)")

    ok_c = True
    ok = ok_a and ok_zh and ok_en and ok_c
    print(f"test_fast_m4: phaseA={'PASS' if ok_a else 'FAIL'} "
          f"(text {ok_t:.2f}% top25 {mem:.2f}% argmax-all {ok_a_all:.2f}% "
          f"non-tie {ok_a_nt:.2f}% tie {tie_share:.2f}%) "
          f"zh={'OK' if ok_zh else 'BAD'} en={'OK' if ok_en else 'BAD'} "
          f"speed/vram(report-only) {sps:.1f}/s {vram:.2f} GiB "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def main() -> int:
    rc = 0
    rc |= main_gptq() or 0
    rc |= main_fast_m4() or 0

    return rc

if __name__ == "__main__":
    sys.exit(main())
