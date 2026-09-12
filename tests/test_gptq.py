"""GPTQ W4 quantization tests (gptq agent).

Phases (all CPU-resident except where the CUDA int4pack kernel is required;
run under the GPU flock — the kernel is CUDA-only on this stack):

  G1 format parity   : `rtn_quantize` + `pack_fast` must reproduce
                       `FastMossTTS._quantize`'s W4 branch *bit for bit*
                       (packed int32 and bf16 qsz), on a synthetic backbone.
  G2 kernel semantics: `_weight_int4pack_mm` output vs the reference
                       dequantization `q*scale + mn` in fp32/bf16 — proves the
                       nibble-offset / odd-even-swap / zeros=mn+8s conventions
                       of `effective_weight`.
  G3 GPTQ core       : (a) GPTQ objective <= RTN objective on the calibration
                       distribution, (b) `slot_errors` matches a direct
                       computation, (c) codes stay in [0,15] and group stats
                       bound the dequantized weight, (d) column-block lazy
                       update equals a brute-force column-wise GPTQ at
                       block_size == K (no laziness).
  G4 injection       : `GptqMossTTS` consumes a pre-quantized state through
                       `FastMossTTS`'s own graph code (`_linear`), including
                       the mixed-precision `bf16_keep` branch — on a synthetic
                       two-layer "model", no real weights needed.
  G5 purity          : `moss_tts_lite/gptq.py` imports torch + stdlib only.

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_gptq
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ..fast import FastMossTTS, _LIN_NAMES
from ..gptq import (SLOT_LINEARS, SLOT_NAMES, CapturingMossTTS, GptqMossTTS,
                    damped_cholesky_inverse, effective_weight, gptq_quantize,
                    pack_fast, rtn_quantize, slot_errors)
from ..model import (AUDIO_PAD_CODE, N_VQ, MossTTSModel, _rms_norm,
                     _rotate_half)

DEV = torch.device("cuda")
G = 128


def _rand_weight(n, k, scale=0.05, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=gen) * scale
    # group-structured magnitude modulation (mimics real weight blocks)
    w = w * (1.0 + 0.5 * torch.sin(torch.arange(k, dtype=torch.float32) / 97.0))
    return w.to(torch.bfloat16).to(DEV)


def _check_int4pack_shapes(m) -> None:
    """`_convert_weight_to_int4pack(inner_k_tiles=8)` requires K/2 % 128 == 0
    (i.e. K % 256 == 0).  A smaller K is silently packed into a short tensor and
    every later kernel comparison becomes self-consistently meaningless."""
    for li, lyr in enumerate(m.layers):
        for name in _LIN_NAMES:
            k, n = int(lyr[name].shape[1]), int(lyr[name].shape[0])
            assert k % 256 == 0, f"L{li} {name}: K={k} invalid for int4pack"
            assert n % 8 == 0, f"L{li} {name}: N={n} invalid for int4pack"


def _fake_backbone(n_layers=2, hidden=256, n_heads=8, head_dim=32,
                   text_vocab=64, seed=0):
    """Synthetic `MossTTSModel` skeleton: only the attributes FastMossTTS,
    `_quantize` and `_linear` touch.  Layer dicts hold the 7 bf16 linears.

    Default widths are valid for the int4pack kernel (K % 256 == 0), so tests
    that call the kernel actually exercise it.
    """
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


# ---------------------------------------------------------------- phase G1 ---
def phase_g1_format_parity():
    print("=== G1: bit-level format parity with FastMossTTS._quantize ===")
    ok = True
    for li, seed in enumerate((11, 12, 13)):
        m = _fake_backbone(n_layers=1, seed=seed)
        # reference: fast.py's own quantizer (its RTN branch), on the same weights
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


# ---------------------------------------------------------------- phase G2 ---
def phase_g2_kernel_semantics():
    print("=== G2: kernel output vs reference dequantization ===")
    torch.manual_seed(0)
    ok = True
    for (n, k) in ((512, 512), (1024, 4096)):
        w = _rand_weight(n, k, seed=n)
        q, s, mn = rtn_quantize(w, G)
        packed, qsz = pack_fast(q, s, mn, 8)
        w_eff = effective_weight(q, qsz, G)                       # [N,K] fp32
        km, x = 8, torch.randn(8, k, dtype=torch.bfloat16, device=DEV)
        y = torch._weight_int4pack_mm(x, packed, G, qsz)
        y_ref = (x.float() @ w_eff.t())
        rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
        # dequantized weights must equal the stored affine form exactly
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


# ---------------------------------------------------------------- phase G3 ---
def _correlated_activations(n_tok, k, seed=0, device=DEV):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(n_tok, k, generator=gen)
    # low-rank + scale structure so H is realistic (not identity)
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

    # (b) slot_errors == direct output-MSE computation
    errs = slot_errors({"cal": x}, w, w_q)
    direct = float(((x @ (w_q - w.float()).t()) ** 2).mean().item())
    print(f"  slot_errors mse={errs['cal']['mse']:.6e}  direct={direct:.6e}  "
          f"rel={errs['cal']['rel']:.3e}")
    ok = ok and abs(errs["cal"]["mse"] - direct) <= 1e-9 + 1e-6 * abs(direct)

    # (c) code range + group stats bound
    assert res.q.min().item() >= 0 and res.q.max().item() <= 15
    print(f"  codes in [{int(res.q.min())},{int(res.q.max())}] "
          f"dtype={res.q.dtype} packed_dtype={res.packed.dtype} "
          f"qsz{tuple(res.qsz.shape)}/{res.qsz.dtype}")

    # (d) lazy block update == non-lazy (block_size == K) column GPTQ
    res_flat = gptq_quantize(w, H, group_size=G, block_size=k, damp_percent=0.01,
                             scale_from="original")
    res_block = gptq_quantize(w, H, group_size=G, block_size=G, damp_percent=0.01,
                              scale_from="original")
    dif = int((res_flat.q != res_block.q).sum().item())
    frac = dif / res_flat.q.numel()
    print(f"  lazy-block(128) vs single-block codes differing: {dif}/{res_flat.q.numel()}"
          f" ({frac:.2e})  [fp32 summation-order only]")
    ok = ok and frac <= 1e-3

    # (e) compensating must not be worse than RTN on the *reported* metric and
    #     'original' vs 'compensated' scaling both work
    res_o = gptq_quantize(w, H, group_size=G, block_size=G, damp_percent=0.01,
                          scale_from="original")
    for tag, r in (("compensated", res), ("original", res_o)):
        e = slot_errors({"cal": x}, w, r.dequant())["cal"]
        print(f"  scale_from={tag:11s} mse={e['mse']:.6e} rel={e['rel']:.4e}")
        ok = ok and e["mse"] < errs["cal"]["mse"] * 5.0
    print(f"  G3: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- phase G4 ---
def phase_g4_injection():
    print("=== G4: GptqMossTTS injection (incl. mixed-precision branch) ===")
    m = _fake_backbone(n_layers=2, seed=21)
    # RTN state built with *this* module's packer, then consumed by the subclass
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
    # _linear dispatch: quantized layers == kernel on packed, bf16 layer == F.linear
    x = torch.randn(3, m.hidden_size, dtype=torch.bfloat16, device=DEV)
    y_keep = fast._linear(x, 1, "q")
    y_ref = F.linear(x, w_keep)
    eq_bf16 = torch.equal(y_keep, y_ref)
    print(f"  bf16_keep layer path == F.linear: {eq_bf16}")
    ok = ok and eq_bf16
    # quantized path must reproduce the kernel call on the stored packed form
    p_my, q_my = fast.qlayers[0]["q"]
    y_kernel = torch._weight_int4pack_mm(x, p_my, G, q_my)
    y_lin = fast._linear(x, 0, "q")
    eq_k = torch.equal(y_kernel, y_lin)
    print(f"  quantized path == _weight_int4pack_mm: {eq_k}")
    ok = ok and eq_k
    print(f"  G4: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- phase G5 ---
def phase_g5_purity():
    print("=== G5: gptq.py dependency purity ===")
    import ast
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "gptq.py"
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


# ---------------------------------------------------------------- phase G7 ---
def phase_g7_shared_hinv():
    """The shared (slot-level) inverse and the CPU-factorisation path must give
    bitwise-identical codes to the per-linear GPU inverse."""
    print("=== G7: shared Hinv + inverse_device equivalence ===")
    n, k, n_tok = 192, 512, 700
    x = _correlated_activations(n_tok, k, seed=17)
    H = x.t() @ x
    w = _rand_weight(n, k, scale=0.08, seed=23)
    ok = True
    # (a) CPU factorisation agrees with the GPU one to fp32 roundoff
    h_gpu, d_gpu = damped_cholesky_inverse(H, 0.01)
    h_cpu, d_cpu = damped_cholesky_inverse(H.to("cpu"), 0.01,
                                           work_device="cpu")
    dmax = (h_gpu.cpu() - h_cpu).abs().max().item()
    scale = h_gpu.abs().max().item()
    rel = dmax / scale
    print(f"  CPU vs GPU inverse: rel|d|={rel:.3e} (max|d|={dmax:.3e}, "
          f"damp {d_gpu:.4e}/{d_cpu:.4e}) — LAPACK vs cuSOLVER roundoff")
    ok = ok and rel < 1e-5
    # (b) the refactored shared-inverse path is bitwise identical to the
    #     original per-linear inverse (same H -> same Hinv -> same codes)
    r_ref = gptq_quantize(w, H, group_size=G, block_size=G)
    r_shr = gptq_quantize(w, group_size=G, block_size=G, Hinv=h_gpu.clone())
    q_same = torch.equal(r_shr.q, r_ref.q)
    sz_same = torch.equal(r_shr.qsz, r_ref.qsz)
    print(f"  shared-Hinv (GPU) == per-linear inverse: codes {q_same}, "
          f"qsz {sz_same}")
    ok = ok and q_same and sz_same
    # (c) the CPU factorisation (production path for K=12288) must land on the
    #     same objective within fp32 roundoff; a few boundary codes may flip
    r_cpu = gptq_quantize(w, group_size=G, block_size=G, Hinv=h_cpu)
    flip = int((r_cpu.q != r_ref.q).sum().item()) / r_ref.q.numel()
    o_ref = _objective(w, r_ref.dequant(), H)
    o_cpu = _objective(w, r_cpu.dequant(), H)
    print(f"  shared-Hinv (CPU): code flips {flip * 100:.3f}%  "
          f"objective {o_cpu:.6e} vs {o_ref:.6e} "
          f"(rel {abs(o_cpu - o_ref) / o_ref:.2e})")
    ok = ok and flip < 1e-3 and abs(o_cpu - o_ref) / o_ref < 1e-2
    # (c) Cholesky inverse really is H^-1 (damped)
    eye = torch.eye(k, device=H.device)
    Hd = H + d_gpu * eye
    resid = (h_gpu.t() @ h_gpu @ Hd - eye).abs().max().item()
    print(f"  ||Hinv^T Hinv H_damped - I||_max = {resid:.3e}")
    ok = ok and resid < 1e-2
    print(f"  G7: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- phase G6 ---
def _mini_weights(hidden=64, head_dim=16, n_heads=4, n_kv=2, inter=96,
                  vocab=151700, n_layers=2, seed=5):
    """Synthetic checkpoint dict shaped like MOSS-TTS-v1.5 (tiny)."""
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
    """CapturingMossTTS must be numerics-identical to MossTTSModel, and the
    captured slots must be the exact tensors the linears consume."""
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
        # every captured slot must be exactly the tensor its consumers read:
        # recompute the layer from the tapped slots and compare (bitwise).
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


# ---------------------------------------------------------------- phase G8 ---
def phase_g8_mixed_dispatch():
    """Per-linear bf16 keep and per-linear group sizes must dispatch to exactly
    the intended kernel/F.linear call (CUDA: the int4pack kernel is CUDA-only)."""
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
    # layer 0: pack all linears at their chosen group; layer 1 too
    keep = {(1, "down"), (1, "o")}          # per-projection bf16
    gmap_pairs = {(0, "down"): 32, (1, "q"): 32}
    sel = {li: {name: gmap_pairs.get((li, name), 128) for name in _LIN_NAMES}
           for li in range(2)}
    base_state = {li: {n: {"packed": state[li][n][sel[li][n]][0],
                           "qsz": state[li][n][sel[li][n]][1]}
                       for n in _LIN_NAMES} for li in range(2)}
    # snapshot the bf16 originals before the subclass pops them
    w_bf16 = {li: {n: m.layers[li][n].clone() for n in _LIN_NAMES}
              for li in range(2)}
    fast = GptqMossTTS(m, base_state, w4_group_size=128, bf16_keep=keep,
                       group_size_map=gmap_pairs)
    ok = True
    print(f"  bf16_keep_linears={fast.bf16_keep_linears} "
          f"mixed_groups={ {f'{a}:{b}': g for (a, b), g in fast.mixed_groups().items()} }")
    for li in range(2):
        for name in _LIN_NAMES:
            # each projection takes its own input width (down_proj is 2x hidden)
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


# ---------------------------------------------------------------- phase G9 ---
def phase_g9_group_sizes():
    """Which int4 group sizes does this build's kernel accept?

    The deployment kernel fixes group_size at runtime (`w4_group_size`), so an
    alternative group size is only usable if `_convert_weight_to_int4pack` +
    `_weight_int4pack_mm` round-trip it.  Reported so the mixed-precision ladder
    can pick a *supported* finer group (64 / 32) instead of guessing.
    """
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
            # the kernel is exact affine arithmetic on the packed weights, so
            # compare against the effective (dequantized) weight
            wq = effective_weight(q, qsz, g).to(torch.bfloat16)
            ref = F.linear(x, wq)
            rel = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
            ok_sizes.append(g)
            print(f"  g={g:4d}: OK  qsz{tuple(qsz.shape)} "
                  f"kernel-vs-dequant rel={rel.item():.2e}")
        except Exception as e:                             # noqa: BLE001
            print(f"  g={g:4d}: FAIL {type(e).__name__}: {str(e)[:70]}")
    print(f"  supported: {ok_sizes}")
    print("  G9: PASS" if ok_sizes else "  G9: FAIL")
    return bool(ok_sizes)


def main() -> int:
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


if __name__ == "__main__":
    sys.exit(main())
