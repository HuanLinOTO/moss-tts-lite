"""单元级验证: liger kernel vs nn.py 原式, fwd + autograd 梯度, fp32/bf16."""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, "/home/lxm/Projects/moss-workspace/moss-tts-lite")
DEV = "cuda"
B, T, HID, NH, KVH, HD = 3, 129, 2560, 32, 8, 80


def max_rel(a, b):
    return ((a.float() - b.float()).abs() / b.float().abs().clamp(min=1e-3)).max().item()


def norm_rel(a, b):
    """L2-level relative error -- honest for gradients with near-zero entries."""
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-6)).item()


def ref_rms(x, w, eps):
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return w * xf.to(input_dtype)


def ref_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def ref_rope(q, k, cos, sin):
    c = cos[None]         # cos is [1,T,D] -> [1,1,T,D] broadcast like the model
    s = sin[None]
    return (q * c + ref_rotate_half(q) * s,
            k * c + ref_rotate_half(k) * s)


results = []
for tag, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    torch.manual_seed(7)
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction

    # --- 1. plain RMSNorm (hidden 2560) ---
    x = torch.randn(2, T, HID, device=DEV, dtype=dt)
    w = torch.randn(HID, device=DEV, dtype=dt)
    x1 = x.clone().requires_grad_(True); w1 = w.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
    y1 = ref_rms(x1, w1, 1e-6)
    y2 = LigerRMSNormFunction.apply(x2, w2, 1e-6, 0.0, "llama", True)
    g = torch.randn_like(y1)
    y1.backward(g); y2.backward(g)
    results.append((tag, "rms fwd", max_rel(y1, y2)))
    results.append((tag, "rms dx", norm_rel(x1.grad, x2.grad)))
    results.append((tag, "rms dw", norm_rel(w1.grad, w2.grad)))

    # --- 2. per-head q-norm (head_dim 80) ---
    qn = torch.randn(B, T, NH, HD, device=DEV, dtype=dt)
    wn = torch.randn(HD, device=DEV, dtype=dt)
    qn1 = qn.clone().requires_grad_(True); qn2 = qn.clone().requires_grad_(True)
    wn1 = wn.clone().requires_grad_(True); wn2 = wn.clone().requires_grad_(True)
    z1 = ref_rms(qn1, wn1, 1e-6)
    z2 = LigerRMSNormFunction.apply(qn2, wn2, 1e-6, 0.0, "llama", True)
    gz = torch.randn_like(z1)
    z1.backward(gz); z2.backward(gz)
    results.append((tag, "qnorm fwd", max_rel(z1, z2)))
    results.append((tag, "qnorm dx", norm_rel(qn1.grad, qn2.grad)))

    # --- 3. RoPE q,k together ---
    q = torch.randn(B, T, NH, HD, device=DEV, dtype=dt)
    k = torch.randn(B, T, KVH, HD, device=DEV, dtype=dt)
    inv = 1.0 / (1e6 ** (torch.arange(0, HD, 2, device=DEV, dtype=torch.float32) / HD))
    pos = torch.arange(T, device=DEV, dtype=torch.float32)
    fr = torch.outer(pos, inv)
    emb = torch.cat((fr, fr), dim=-1)
    cos = emb.cos().to(dt)[None]   # [1, T, D]
    sin = emb.sin().to(dt)[None]
    qt1 = q.transpose(1, 2).clone().requires_grad_(True)
    qt2 = q.transpose(1, 2).detach().clone().requires_grad_(True)
    kt1 = k.transpose(1, 2).clone().requires_grad_(True)
    kt2 = k.transpose(1, 2).detach().clone().requires_grad_(True)
    q1, k1 = ref_rope(qt1, kt1, cos, sin)
    q2, k2 = LigerRopeFunction.apply(qt2, kt2, cos, sin)
    gq = torch.randn_like(q1); gk = torch.randn_like(k1)
    q1.backward(gq, retain_graph=True); k1.backward(gk)
    torch.autograd.backward([q2, k2], [gq, gk])
    results.append((tag, "rope q fwd", max_rel(q1, q2)))
    results.append((tag, "rope k fwd", max_rel(k1, k2)))
    results.append((tag, "rope dq", norm_rel(qt1.grad, qt2.grad)))
    results.append((tag, "rope dk", norm_rel(kt1.grad, kt2.grad)))

    # --- 4. SiLUMul ---
    g_ = torch.randn(B, T, 9728, device=DEV, dtype=dt)
    u_ = torch.randn(B, T, 9728, device=DEV, dtype=dt)
    g1 = g_.clone().requires_grad_(True); g2 = g_.clone().requires_grad_(True)
    u1 = u_.clone().requires_grad_(True); u2 = u_.clone().requires_grad_(True)
    s1 = F.silu(g1) * u1
    s2 = LigerSiLUMulFunction.apply(g2, u2)
    gs = torch.randn_like(s1)
    s1.backward(gs); s2.backward(gs)
    results.append((tag, "silumul fwd", max_rel(s1, s2)))
    results.append((tag, "silumul dg", norm_rel(g1.grad, g2.grad)))
    results.append((tag, "silumul du", norm_rel(u1.grad, u2.grad)))

    # --- 5. FusedAddRMSNorm ---
    xr = torch.randn(B, T, HID, device=DEV, dtype=dt)
    rr = torch.randn(B, T, HID, device=DEV, dtype=dt)
    wf = torch.randn(HID, device=DEV, dtype=dt)
    xr1 = xr.clone().requires_grad_(True); xr2 = xr.clone().requires_grad_(True)
    rr1 = rr.clone().requires_grad_(True); rr2 = rr.clone().requires_grad_(True)
    wf1 = wf.clone().requires_grad_(True); wf2 = wf.clone().requires_grad_(True)
    s_ref = rr1 + xr1
    y1 = ref_rms(s_ref, wf1, 1e-6)
    y2, s2 = LigerFusedAddRMSNormFunction.apply(xr2, rr2, wf2, 1e-6)
    gy = torch.randn_like(y1); gsr = torch.randn_like(s_ref)
    (y1 * 0 + y1.backward(gy, retain_graph=True) if False else None)
    y1.backward(gy, retain_graph=True); s_ref.backward(gsr)
    torch.autograd.backward([y2, s2], [gy, gsr])
    results.append((tag, "fused y fwd", max_rel(y1, y2)))
    results.append((tag, "fused s fwd", max_rel(s_ref, s2)))
    results.append((tag, "fused dx", norm_rel(xr1.grad, xr2.grad)))
    results.append((tag, "fused dr", norm_rel(rr1.grad, rr2.grad)))

thr = 1e-4
bad = 0
for tag, name, v in results:
    lim = thr if tag == "fp32" else 2e-2
    flag = "OK " if v < lim else "!! "
    if v >= lim:
        bad += 1
    print(f"{flag}[{tag}] {name:14s} max_rel={v:.3e}")
print("VERDICT", "PASS" if bad == 0 else f"FAIL({bad})")
