"""GPTQ (error-compensating) W4 quantization for the FastMossTTS int4pack path."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .fast import FastMossTTS, _LIN_NAMES
from .model import MossTTSModel, _rms_norm, _rotate_half

__all__ = [
    "SLOT_LINEARS", "SLOT_NAMES", "GPTQResult",
    "group_scales_from_weight", "rtn_quantize", "pack_fast", "pack_qsz_only",
    "effective_scale_zero", "effective_weight",
    "damped_cholesky_inverse", "group_ls_scale_zero", "group_error",
    "gptq_quantize", "slot_errors", "rel_mse", "rel_mse_per_stratum",
    "build_hessian", "quantize_backbone_gptq", "load_gptq_state",
    "load_gptq_fast", "GptqMossTTS", "CapturingMossTTS",
]

SLOT_LINEARS: dict[str, tuple[str, ...]] = {
    "attn_in": ("q", "k", "v"),
    "attn_out": ("o",),
    "mlp_in": ("gate", "up"),
    "down_in": ("down",),
}
SLOT_NAMES: tuple[str, ...] = tuple(SLOT_LINEARS)

_SLOT_OF_LINEAR: dict[str, str] = {
    lin: slot for slot, lins in SLOT_LINEARS.items() for lin in lins
}

def group_scales_from_weight(w2d: torch.Tensor, group_size: int = 128
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric group min/max statistics, exactly as `fast."""
    n, k = w2d.shape
    if k % group_size:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    wg = w2d.reshape(n, k // group_size, group_size)
    mx = wg.amax(-1)
    mn = wg.amin(-1)
    s = ((mx - mn) / 15.0).clamp(min=1e-6)
    return s, mn

def bf16_effective_scale_zero(s: torch.Tensor, mn: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the *storage* rounding of `qsz` back into fp32 scale/zero."""
    s_eff = s.to(torch.bfloat16).to(torch.float32)
    zero = (mn + 8.0 * s).to(torch.bfloat16).to(torch.float32)
    return s_eff, zero - 8.0 * s_eff

def rtn_quantize(w2d: torch.Tensor, group_size: int = 128
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact replica of the `FastMossTTS."""
    w = w2d.float()
    n, k = w.shape
    g = group_size
    s, mn = group_scales_from_weight(w, g)
    wg = w.reshape(n, k // g, g)
    q = ((wg - mn.unsqueeze(-1)) / s.unsqueeze(-1)).round().clamp(0, 15) \
        .to(torch.uint8).reshape(n, k)
    return q, s, mn

def pack_fast(q: torch.Tensor, s: torch.Tensor, mn: torch.Tensor,
              inner_k_tiles: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack int4 codes + group stats into the `(packed, qsz)` pair fast."""
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    if not q.is_contiguous():
        q = q.contiguous()
    qsz = torch.stack([s, (mn + 8.0 * s)], -1).bfloat16() \
        .transpose(0, 1).contiguous()
    packed = torch.ops.aten._convert_weight_to_int4pack(
        (q[:, 1::2] | (q[:, 0::2] << 4)).contiguous(), inner_k_tiles)
    return packed, qsz

def pack_qsz_only(s: torch.Tensor, mn: torch.Tensor,
                  group_size: int = 128) -> torch.Tensor:
    """`qsz` half of `pack_fast` without the kernel packer (CPU-safe)."""
    n, gcount = s.shape
    del n
    return torch.stack([s, (mn + 8.0 * s)], -1).bfloat16() \
        .transpose(0, 1).contiguous()

def effective_weight(q: torch.Tensor, qsz: torch.Tensor,
                     group_size: int = 128, dtype: torch.dtype = torch.float32
                     ) -> torch.Tensor:
    """Dequantize exactly as `_weight_int4pack_mm` does (kernel semantics)."""
    n, k = q.shape
    g = group_size
    scale = qsz[:, :, 0].t().to(dtype)
    zero = qsz[:, :, 1].t().to(dtype)
    qf = q.to(dtype).reshape(n, k // g, g)
    w = (qf - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return w.reshape(n, k)

def damped_cholesky_inverse(H: torch.Tensor, damp_percent: float = 0.01,
                            max_tries: int = 8, verbose: bool = False,
                            work_device=None) -> tuple[torch.Tensor, float]:
    """Upper-triangular inverse of a damped Hessian (GPTQ reference recipe)."""
    dev = H.device if work_device is None else torch.device(work_device)
    Hd = H.to(dev, dtype=torch.float32, copy=True)
    diag = torch.diagonal(Hd)
    dead = diag == 0
    if bool(dead.any()):
        Hd[dead, dead] = 1.0
    base_damp = float(damp_percent * diag.mean().item())
    damp = base_damp
    with torch.no_grad():
        Hd.diagonal().add_(damp)
        for attempt in range(max_tries):
            try:
                L = torch.linalg.cholesky(Hd)
                Hinv = torch.cholesky_inverse(L)
                del L

                sym = torch.empty_like(Hinv)
                torch.add(Hinv, Hinv.t(), out=sym)
                sym.mul_(0.5)
                del Hinv
                Hiu = torch.linalg.cholesky(sym, upper=True)
                if bool(torch.isfinite(Hiu).all()):
                    return Hiu, damp
            except Exception:                                 # noqa: BLE001
                pass
            prev, damp = damp, max(damp * 4.0, 1e-12)
            Hd.diagonal().add_(damp - prev)
            if verbose:
                print(f"    [damped_cholesky_inverse] retry damp={damp:.3e}")
    raise RuntimeError("damped_cholesky_inverse failed (Hessian not invertible)")

@dataclass
class GPTQResult:
    """One quantized linear, in fast."""

    q: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    mn: torch.Tensor
    packed: torch.Tensor
    qsz: torch.Tensor
    group_size: int = 128
    info: dict = field(default_factory=dict)

    def dequant(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return effective_weight(self.q, self.qsz, self.group_size, dtype)

def group_ls_scale_zero(q: torch.Tensor, H: torch.Tensor, w_orig: torch.Tensor,
                        group_size: int, s_init: torch.Tensor,
                        mn_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-(output row, group) affine re-fit of (scale, zero)."""
    dev = q.device
    n, k = q.shape
    g = group_size
    G = k // g
    Hd = H.detach().to(dev, dtype=torch.float32)
    Hb = Hd.reshape(G, g, G, g)
    idx = torch.arange(G, device=dev)
    Hg = Hb[idx, :, idx, :].contiguous()
    Qf = q.to(torch.float32)
    Qg = Qf.reshape(n, G, g).permute(1, 0, 2).contiguous()
    Wg = w_orig.detach().to(dev, dtype=torch.float32).reshape(n, G, g) \
        .permute(1, 0, 2).contiguous()

    def _quad(Hg: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """sum_j A[."""
        HB = torch.bmm(Hg, B.transpose(1, 2))
        return (A * HB.transpose(1, 2)).sum(-1)

    h1 = Hg.sum(-1)
    a11 = _quad(Hg, Qg, Qg)
    a12 = torch.bmm(h1.unsqueeze(1), Qg.transpose(1, 2)).squeeze(1)
    a22 = (h1 * h1.new_ones(g)).sum(-1)
    b1 = _quad(Hg, Qg, Wg)
    b2 = (h1.unsqueeze(1) * Wg).sum(-1)

    det = a11 * a22.unsqueeze(-1) - a12 * a12
    safe = (det.abs() > 1e-20) & torch.isfinite(det)
    s_new = torch.where(safe, (b1 * a22.unsqueeze(-1) - a12 * b2) / det,
                        s_init.t().expand(G, n))
    z_new = torch.where(safe, (a11 * b2 - a12 * b1) / det,
                        mn_init.t().expand(G, n))
    finite = safe & (s_new > 0) & torch.isfinite(s_new) & torch.isfinite(z_new)
    s_new = torch.where(finite, s_new, s_init.t().expand(G, n))
    z_new = torch.where(finite, z_new, mn_init.t().expand(G, n))

    s_new_n = s_new.t().contiguous()
    z_new_n = z_new.t().contiguous()
    s_bf, mn_bf = bf16_effective_scale_zero(s_new_n, z_new_n)
    s_ib, mn_ib = bf16_effective_scale_zero(s_init, mn_init)
    e_new = group_error(Qf, Hg, s_bf.t(), mn_bf.t(), Wg)
    e_old = group_error(Qf, Hg, s_ib.t(), mn_ib.t(), Wg)
    take = (e_new < e_old).t().contiguous()
    return torch.where(take, s_bf, s_ib), torch.where(take, mn_bf, mn_ib)

def group_error(q: torch.Tensor, Hg: torch.Tensor, s: torch.Tensor,
                mn: torch.Tensor, Wg: torch.Tensor) -> torch.Tensor:
    """Per-(group,row) squared error (q*s + mn - w)^T H_g (q*s + mn - w)."""
    Qs = q.reshape(Wg.shape[1], Hg.shape[0], Hg.shape[1]).permute(1, 0, 2)
    d = (Qs * s.unsqueeze(-1) + mn.unsqueeze(-1)) - Wg
    Hd = torch.bmm(Hg, d.transpose(1, 2))
    return (d * Hd.transpose(1, 2)).sum(-1)

def gptq_quantize(w2d: torch.Tensor, H: torch.Tensor | None = None,
                  group_size: int = 128, block_size: int = 128,
                  damp_percent: float = 0.01,
                  scale_from: str = "compensated",
                  inner_k_tiles: int = 8,
                  Hinv: torch.Tensor | None = None,
                  refine_scales: bool = False,
                  static_groups: bool = False) -> GPTQResult:
    """GPTQ-quantize one linear weight against activation covariance H."""
    w = w2d.reshape(-1, w2d.shape[-1]).float()
    n, k = w.shape
    g = group_size
    if k % g or k % block_size:
        raise ValueError(f"K={k} not divisible by group_size={g}/block={block_size}")
    if Hinv is None:
        if H is None or H.shape != (k, k):
            raise ValueError(f"H {tuple(H.shape) if H is not None else None} "
                             f"does not match K={k}")
    elif Hinv.shape != (k, k):
        raise ValueError(f"Hinv {tuple(Hinv.shape)} does not match K={k}")
    if scale_from not in ("original", "compensated"):
        raise ValueError(f"bad scale_from {scale_from!r}")

    s0, mn0 = group_scales_from_weight(w, g)
    s_eff, mn_eff = bf16_effective_scale_zero(s0, mn0)

    if Hinv is None:
        Hinv, damp = damped_cholesky_inverse(H, damp_percent)
    else:
        Hinv, damp = Hinv, float("nan")
    Hinv = Hinv.to(w.device, dtype=torch.float32, non_blocking=False).contiguous()

    hdiag = torch.diagonal(Hinv).clone()
    hdiag = torch.where((hdiag == 0) | ~torch.isfinite(hdiag),
                        torch.ones_like(hdiag), hdiag)

    Wcur = w.clone()
    Q = torch.zeros(n, k, dtype=torch.uint8, device=w.device)
    if static_groups:

        for gi in range(k // g):
            c0, c1 = gi * g, (gi + 1) * g
            s_c, mn_c = s_eff[:, gi], mn_eff[:, gi]
            Wg = Wcur[:, c0:c1]
            qj = torch.round((Wg - mn_c[:, None]) / s_c[:, None]) \
                .clamp_(0, 15).to(torch.uint8)
            Q[:, c0:c1] = qj
            err = Wg - (qj.float() * s_c[:, None] + mn_c[:, None])
            if c1 < k:
                Wcur[:, c1:] -= err @ Hinv[c0:c1, c1:]
        if refine_scales and H is not None:
            s0, mn0 = group_ls_scale_zero(Q, H, w, g, s0, mn0)
        zero_bf, s_bf = (mn0 + 8.0 * s0).bfloat16().float(), s0.bfloat16().float()
        mn_bf = zero_bf - 8.0 * s_bf
        if Q.is_cuda:
            packed, qsz = pack_fast(Q, s0, mn0, inner_k_tiles)
        else:
            packed, qsz = None, pack_qsz_only(s0, mn0, g)
        return GPTQResult(q=Q, scale=s_bf, zero=zero_bf, mn=mn_bf,
                          packed=packed, qsz=qsz, group_size=g,
                          info={"damping": damp, "damp_percent": damp_percent,
                                "scale_from": scale_from, "group_size": g,
                                "block_size": k, "static_groups": True,
                                "refine_scales": bool(refine_scales),
                                "K": k, "N": n})
    for i1 in range(0, k, block_size):
        i2 = min(i1 + block_size, k)
        count = i2 - i1
        W1 = Wcur[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            j = i1 + i
            gi = j // g
            if scale_from == "compensated" and j % g == 0:

                j2 = min(j + g, k)
                if j2 <= i2:
                    grp = W1[:, i:i + (j2 - j)]
                else:
                    grp = torch.cat([W1[:, i:count], Wcur[:, i2:j2]], dim=1)
                s_g, mn_g = group_scales_from_weight(grp, g)
                s0[:, gi:gi + 1] = s_g
                mn0[:, gi:gi + 1] = mn_g
                s_eff[:, gi:gi + 1], mn_eff[:, gi:gi + 1] = \
                    bf16_effective_scale_zero(s_g, mn_g)
            s_c = s_eff[:, gi]
            mn_c = mn_eff[:, gi]
            wt = W1[:, i]
            d = hdiag[i1 + i]
            qj = torch.round((wt - mn_c) / s_c).clamp_(0, 15).to(torch.uint8)
            Q1[:, i] = qj
            w_hat = qj.float() * s_c + mn_c
            err = (wt - w_hat) / d
            if i + 1 < count:
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < k:
            Wcur[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    refined = False
    if refine_scales and H is not None:
        s0, mn0 = group_ls_scale_zero(Q, H, w, g, s0, mn0)
        refined = True

    zero_bf, s_bf = (mn0 + 8.0 * s0).bfloat16().float(), s0.bfloat16().float()
    mn_bf = zero_bf - 8.0 * s_bf

    if Q.is_cuda:
        packed, qsz = pack_fast(Q, s0, mn0, inner_k_tiles)
    else:
        packed, qsz = None, pack_qsz_only(s0, mn0, g)
    info = {
        "damping": damp,
        "damp_percent": damp_percent,
        "scale_from": scale_from,
        "group_size": g,
        "block_size": block_size,
        "refine_scales": refined,
        "K": k, "N": n,
    }
    return GPTQResult(q=Q, scale=s_bf, zero=zero_bf, mn=mn_bf,
                      packed=packed, qsz=qsz, group_size=g, info=info)

def slot_errors(x_by_stratum: dict, w_orig: torch.Tensor,
                w_quant: torch.Tensor, chunk: int = 1024, device=None,
                slot: str | None = None) -> dict[str, dict]:
    """Output MSE of a quantized linear, per calibration stratum."""
    dev = device if device is not None else w_quant.device
    dW = (w_quant - w_orig).float().to(dev)
    W = w_orig.float().to(dev)
    n_out = int(w_quant.shape[0])
    out: dict[str, dict] = {}

    flat: dict[str, torch.Tensor] = {}

    def _walk(prefix: str, node) -> None:
        if isinstance(node, dict):
            for k2, v2 in node.items():
                _walk(f"{prefix}/{k2}" if prefix else str(k2), v2)
        else:
            flat[prefix] = node

    _walk("", x_by_stratum)
    if slot is not None:
        flat = {k: v for k, v in flat.items() if k.rsplit("/", 1)[-1] == slot}
    for name, x in flat.items():
        if x.dim() != 2 or int(x.shape[1]) != int(w_quant.shape[1]):
            raise ValueError(
                f"activation '{name}' has shape {tuple(x.shape)} but the linear "
                f"expects K={int(w_quant.shape[1])} (capture/download mismatch?)")
        se = 0.0
        sr = 0.0
        se_max = 0.0
        n_tok = 0
        n = int(x.shape[0])
        for a in range(0, n, chunk):
            xb = x[a:a + chunk].to(dev, dtype=torch.float32, non_blocking=False)
            e = xb @ dW.t()
            r = xb @ W.t()
            se += float((e * e).sum().item())
            sr += float((r * r).sum().item())
            se_max = max(se_max, float((e * e).amax().item()))
            n_tok += int(xb.shape[0])
            del xb, e, r
        denom = max(n_tok * n_out, 1)
        mse = se / denom
        ref = sr / denom
        out[name] = {"n_tokens": n_tok, "mse": mse, "ref_mse": ref,
                     "rel": (mse / ref) if ref > 0 else float("nan"),
                     "max_sq_err": se_max}
    del dW, W
    return out

def rel_mse(x_by_stratum: dict, w_orig: torch.Tensor, w_quant: torch.Tensor,
            device=None, slot: str | None = None) -> float:
    """Weighted relative output MSE:"""
    e = slot_errors(x_by_stratum, w_orig, w_quant, device=device, slot=slot)
    num = sum(v["mse"] * v["n_tokens"] for v in e.values())
    den = sum(v["ref_mse"] * v["n_tokens"] for v in e.values())
    return num / den if den else float("nan")

def rel_mse_per_stratum(x_by_stratum: dict, w_orig: torch.Tensor,
                        w_quant: torch.Tensor, device=None) -> dict[str, float]:
    """Same as `rel_mse` but keeping the strata separate."""
    e = slot_errors(x_by_stratum, w_orig, w_quant, device=device)
    return {k: (v["rel"] if v["ref_mse"] > 0 else float("nan"))
            for k, v in e.items()}

def build_hessian(x_by_stratum: dict, li: int, slot: str, dev,
                  chunk: int = 512) -> tuple[torch.Tensor, dict]:
    """Sum of per-stratum X^T X for one (layer, slot), built in fp32 chunks."""
    H = None
    stats: dict = {"n_positions": 0, "per_stratum": {}}
    for sname, by_layer in x_by_stratum.items():
        x = by_layer.get(li, {}).get(slot)
        if x is None or x.shape[0] == 0:
            continue
        n = int(x.shape[0])
        ss = float((x.float() ** 2).sum().item())
        for a in range(0, n, chunk):
            xb = x[a:a + chunk].to(dev, dtype=torch.float32, non_blocking=False)
            hh = xb.t() @ xb
            H = hh if H is None else H.add_(hh)
            del xb, hh
        stats["n_positions"] += n
        stats["per_stratum"][sname] = {"n_positions": n, "sum_sq": ss,
                                       "norm_per_pos": ss / max(n, 1)}
    if H is None:
        raise KeyError(f"no calibration activations for layer {li}/{slot}")
    tot = sum(v["sum_sq"] for v in stats["per_stratum"].values()) or 1.0
    for v in stats["per_stratum"].values():
        v["ss_share"] = v["sum_sq"] / tot
    return H, stats

def quantize_backbone_gptq(model, x_by_stratum: dict[str, dict[int, dict[str, torch.Tensor]]],
                           group_size: int = 128, block_size: int = 128,
                           damp_percent: float = 0.01,
                           scale_from: str = "compensated",
                           inner_k_tiles: int = 8, device=None,
                           ridge_frac: float = 0.0, verbose: bool = True,
                           layers: list[int] | None = None,
                           inverse_device="cpu", chunk: int = 512,
                           refine_scales: bool = False,
                           static_groups: bool = False,
                           ) -> tuple[dict, dict]:
    """Quantize all 7 backbone linears of the requested layers with GPTQ."""
    dev = device or model.device
    layers = list(range(model.n_layers)) if layers is None else sorted(layers)
    state: dict[int, dict[str, dict]] = {}
    report: dict = {"layers": {}, "config": {"group_size": group_size,
                                            "block_size": block_size,
                                            "damp_percent": damp_percent,
                                            "scale_from": scale_from,
                                            "ridge_frac": ridge_frac,
                                            "refine_scales": bool(refine_scales),
                                            "static_groups": bool(static_groups),
                                            "inverse_device": str(inverse_device)}}
    for li in layers:
        lyr = model.layers[li]
        state[li] = {}
        lrep: dict = {"slots": {}, "linears": {}}
        for slot in SLOT_NAMES:
            H, hstats = build_hessian(x_by_stratum, li, slot, dev, chunk=chunk)
            if ridge_frac:
                lam = ridge_frac * float(torch.diagonal(H).mean().item())
                H.diagonal().add_(lam)
            hdiag = torch.diagonal(H)
            hstats["trace"] = float(hdiag.sum().item())
            hstats["mean_diag"] = float(hdiag.mean().item())
            hdiag_cpu = hdiag.detach().to("cpu").clone()
            lrep["slots"][slot] = hstats

            inv_dev = None if inverse_device in (None, dev, str(dev), "auto") \
                else inverse_device
            if inv_dev is None:
                Hinv, damp = damped_cholesky_inverse(H, damp_percent,
                                                     verbose=verbose)
            else:
                Hc = H.to("cpu", copy=True)
                Hinv, damp = damped_cholesky_inverse(Hc, damp_percent,
                                                     work_device=inv_dev,
                                                     verbose=verbose)
                del Hc
            torch.cuda.empty_cache()
            for name in SLOT_LINEARS[slot]:
                w_bf16 = lyr[name]
                res = gptq_quantize(
                    w_bf16, H=H, group_size=group_size, block_size=block_size,
                    damp_percent=damp_percent, scale_from=scale_from,
                    inner_k_tiles=inner_k_tiles,
                    Hinv=None if static_groups else Hinv,
                    refine_scales=refine_scales, static_groups=static_groups)
                w_hat = res.dequant()
                x_gpu = {s: x_by_stratum[s][li][slot]
                         for s in x_by_stratum if slot in x_by_stratum[s].get(li, {})}
                errs = slot_errors(x_gpu, w_bf16, w_hat, chunk=chunk, device=dev)
                q_r, s_r, mn_r = rtn_quantize(w_bf16, group_size)
                qsz_r = pack_qsz_only(s_r, mn_r, group_size)
                w_rtn = effective_weight(q_r, qsz_r, group_size)
                errs_rtn = slot_errors(x_gpu, w_bf16, w_rtn, chunk=chunk, device=dev)
                n_tok = sum(v["n_tokens"] for v in errs.values())
                gptq_obj = sum(v["mse"] * v["n_tokens"] for v in errs.values()) / max(n_tok, 1)
                rtn_obj = sum(v["mse"] * v["n_tokens"] for v in errs_rtn.values()) / max(n_tok, 1)
                lrep["linears"][name] = {
                    "shape": [int(w_bf16.shape[0]), int(w_bf16.shape[1])],
                    "gptq_obj_mse": gptq_obj, "rtn_obj_mse": rtn_obj,
                    "gptq_over_rtn": gptq_obj / rtn_obj if rtn_obj > 0 else float("nan"),
                    "errors": errs, "errors_rtn": errs_rtn,
                    "damping": damp, **res.info,
                }
                if verbose:
                    print(f"  L{li:02d} {name:5s} gptq_mse={gptq_obj:.3e} "
                          f"rtn_mse={rtn_obj:.3e} ratio="
                          f"{gptq_obj / rtn_obj if rtn_obj else float('nan'):.3f}",
                          flush=True)
                state[li][name] = {
                    "packed": res.packed, "qsz": res.qsz,
                    "q": res.q.cpu(), "scale": res.scale.cpu(),
                    "zero": res.zero.cpu(),
                    "shape": (int(w_bf16.shape[0]), int(w_bf16.shape[1])),
                }
                del res, w_hat, errs, errs_rtn, w_rtn, x_gpu
                torch.cuda.empty_cache()
            del Hinv, H
            torch.cuda.empty_cache()
        report["layers"][li] = lrep
        torch.cuda.empty_cache()
    return state, report

class CapturingMossTTS(MossTTSModel):
    """`MossTTSModel` that also records the four linear inputs of every layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_enabled = True
        self.cap_slots: dict[int, dict[str, torch.Tensor]] = {}

    def prefill(self, input_ids):
        self.cap_slots = {}
        return super().prefill(input_ids)

    def step(self, row):
        self.cap_slots = {}
        return super().step(row)

    def _lin(self, x: torch.Tensor, li: int, name: str) -> torch.Tensor:
        """Active-weight linear dispatch (bf16 here;"""
        return F.linear(x, self.layers[li][name])

    def _layer(self, li: int, h: torch.Tensor, s0: int) -> torch.Tensor:
        lyr = self.layers[li]
        t = h.shape[1]
        s1 = s0 + t
        residual = h
        x = _rms_norm(h, lyr["input_layernorm"])

        q = self._lin(x, li, "q").view(1, t, self.n_heads, self.head_dim)
        k = self._lin(x, li, "k").view(1, t, self.n_kv_heads, self.head_dim)
        v = self._lin(x, li, "v").view(1, t, self.n_kv_heads, self.head_dim)
        q = _rms_norm(q, lyr["q_norm"]).transpose(1, 2)
        k = _rms_norm(k, lyr["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[s0:s1].unsqueeze(0)
        sin = self.rope_sin[s0:s1].unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        self.k_cache[li, :, s0:s1] = k[0]
        self.v_cache[li, :, s0:s1] = v[0]

        keys = self.k_cache[li, :, :s1].repeat_interleave(self.n_rep, dim=0).unsqueeze(0)
        vals = self.v_cache[li, :, :s1].repeat_interleave(self.n_rep, dim=0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, keys, vals,
                                             is_causal=(s1 == t and t > 1))
        attn_out = out.transpose(1, 2).reshape(1, t, self.hidden_size)
        h = residual + self._lin(attn_out, li, "o")

        residual = h
        x2 = _rms_norm(h, lyr["post_attention_layernorm"])
        down_in = F.silu(self._lin(x2, li, "gate")) * self._lin(x2, li, "up")
        h = residual + self._lin(down_in, li, "down")

        if self.capture_enabled:
            self.cap_slots[li] = {"attn_in": x, "attn_out": attn_out,
                                  "mlp_in": x2, "down_in": down_in}
        return h

def kernel_linear(x: torch.Tensor, packed, g: int, qsz,
                  out_dtype: torch.dtype) -> torch.Tensor:
    """int4pack GEMM shaped like `F."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    out = torch._weight_int4pack_mm(x2, packed, g, qsz)
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out.reshape(*shp[:-1], qsz.shape[1])

class W4CapturingMossTTS(CapturingMossTTS):
    """Eager decoder that runs *already quantized* projections through the int4pack kernel and leaves the rest in bf16."""

    def __init__(self, *args, w4_state=None, group_size_map=None,
                 default_group_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.w4_state = w4_state if w4_state is not None else {}
        self.group_size_map = dict(group_size_map or {})
        self.default_group_size = int(default_group_size)

    def group_size_for(self, li: int, name: str) -> int:
        return int(self.group_size_map.get((li, name), self.default_group_size))

    def install(self, li: int, name: str, packed, qsz) -> None:
        self.w4_state.setdefault(li, {})[name] = {"packed": packed, "qsz": qsz}

    def _lin(self, x: torch.Tensor, li: int, name: str) -> torch.Tensor:
        rec = self.w4_state.get(li, {}).get(name)
        if rec is None:
            return F.linear(x, self.layers[li][name])
        return kernel_linear(x, rec["packed"], self.group_size_for(li, name),
                             rec["qsz"], x.dtype)

def load_gptq_state(path: str) -> tuple[dict, dict, list]:
    """Load an offline GPTQ state produced by the calibration pipeline."""
    import json
    import os
    state = torch.load(path, map_location="cpu", weights_only=False)
    meta_path = path + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found: per-linear group sizes and bf16 keeps "
            f"cannot be inferred from the weights alone")
    meta = json.load(open(meta_path))
    gmap = {}
    for key, val in meta.get("group_size_map", {}).items():
        layer, name = key.split(":")
        gmap[(int(layer), name)] = int(val)
    keep: list = [(int(x.split(":")[0]), x.split(":")[1])
                  for x in meta.get("bf16_linears", [])]
    keep += [int(x) for x in meta.get("bf16_layers", [])]
    return state, gmap, keep

def load_gptq_fast(model, path: str):
    """Convenience:"""
    state, gmap, keep = load_gptq_state(path)
    fast = GptqMossTTS(model, state, bf16_keep=keep, group_size_map=gmap)
    del state
    return fast

class GptqMossTTS(FastMossTTS):
    """`FastMossTTS` (identical graph/kernel/format) fed GPTQ-packed weights."""

    def __init__(self, model, state: dict, inner_k_tiles: int = 8,
                 w4_group_size: int = 128, bf16_keep=(),
                 group_size_map: dict | None = None):
        """Args:"""
        self._gptq_state = state
        self._bf16_keep = set(bf16_keep)
        self._keep_layers = {int(x) for x in self._bf16_keep
                             if isinstance(x, int)}
        self._keep_linears = {(int(a), b) for a, b in
                              (x for x in self._bf16_keep
                               if not isinstance(x, int))}
        self._group_size_map = {(int(a), b): int(g) for (a, b), g
                                in (group_size_map or {}).items()}
        super().__init__(model, quant="w4", inner_k_tiles=inner_k_tiles,
                         w4_group_size=w4_group_size)

        self.group_size_map: dict[int, dict[str, int]] = {}
        for li in range(model.n_layers):
            gmap = {}
            for name in _LIN_NAMES:
                gmap[name] = self._group_size_map.get((li, name),
                                                      self.w4_group_size)
            self.group_size_map[li] = gmap

    def _quantize(self) -> None:
        m = self.m
        dev = m.device
        self.qlayers = []
        for li in range(m.n_layers):
            lyr = m.layers[li]
            if li in self._keep_layers:
                self.qlayers.append(None)
                continue
            qd = {}
            for name in _LIN_NAMES:
                if (li, name) in self._keep_linears:
                    continue
                rec = self._gptq_state[li][name]
                packed = rec["packed"].to(device=dev, non_blocking=False)
                qsz = rec["qsz"].to(device=dev, non_blocking=False)
                if packed.dtype != torch.int32:
                    packed = packed.to(torch.int32)
                if qsz.dtype != torch.bfloat16:
                    qsz = qsz.to(torch.bfloat16)
                qd[name] = (packed.contiguous(), qsz.contiguous())
                lyr.pop(name)
            self.qlayers.append(qd if qd else None)
        torch.cuda.empty_cache()

    def _linear(self, x: torch.Tensor, li: int, name: str) -> torch.Tensor:
        qd = self.qlayers[li]
        if qd is None or name not in qd:
            return F.linear(x, self.m.layers[li][name])
        g = self.group_size_map[li][name]
        if g != self.w4_group_size:
            packed, qsz = qd[name]
            return kernel_linear(x, packed, g, qsz, self.m.dtype)
        return super()._linear(x, li, name)

    @property
    def bf16_keep_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._keep_layers))

    @property
    def bf16_keep_linears(self) -> tuple:
        return tuple(sorted(self._keep_linears))

    def mixed_groups(self) -> dict:
        """{(layer, name):"""
        out = {}
        for li, gmap in self.group_size_map.items():
            for name, g in gmap.items():
                if g != self.w4_group_size:
                    out[(li, name)] = g
        return out
