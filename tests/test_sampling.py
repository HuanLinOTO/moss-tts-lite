"""CPU unit tests for sampling.py: op-for-op equivalence with
inference_utils.py reference implementations (exact-comparison version).

Run: python3 -m moss_tts_lite.tests.test_sampling   (CPU only, fast)
"""

import torch
import torch.nn.functional as F

from moss_tts_lite.sampling import (
    apply_repetition_penalty_delay_pattern,
    apply_top_k,
    apply_top_p,
    apply_top_p_optimized,
    find_last_equal_C,
    sample_token,
)

# ---- exact copies of models/MOSS-TTS-v1.5/inference_utils.py ----


def ref_apply_top_k(logits, top_k):
    batch_size, vocab_size = logits.shape
    top_k = min(top_k, vocab_size)
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    batch_indices = torch.arange(batch_size).unsqueeze(-1)
    filtered_logits[batch_indices, top_k_indices] = top_k_values
    return filtered_logits


def ref_apply_top_p(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    batch_size = logits.shape[0]
    filtered_logits = logits.clone()
    for i in range(batch_size):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        filtered_logits[i, indices_to_remove] = float("-inf")
    return filtered_logits


def ref_apply_repetition_penalty_delay_pattern(logits, prev_tokens, penalty):
    if penalty == 1.0 or prev_tokens is None:
        return logits
    if logits.dim() == 2:
        prev_tokens_flat = prev_tokens.reshape(-1)
        unique_tokens = torch.unique(prev_tokens_flat)
        token_logits = logits[:, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, unique_tokens] = token_logits
        return logits
    assert logits.dim() == 3
    B, H, V = logits.shape
    for h in range(H):
        prev_tokens_h = prev_tokens[..., h].reshape(-1)
        unique_tokens = torch.unique(prev_tokens_h)
        if unique_tokens.numel() == 0:
            continue
        token_logits = logits[:, h, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, h, unique_tokens] = token_logits
    return logits


def ref_sample_token(logits, prev_tokens=None, repetition_penalty=1.0,
                     top_p=None, top_k=None, do_sample=True):
    vocab_size = logits.size(-1)
    if prev_tokens is not None and repetition_penalty != 1.0:
        logits = ref_apply_repetition_penalty_delay_pattern(logits, prev_tokens, repetition_penalty)
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    original_shape = logits.shape
    reshaped_logits = logits.view(-1, vocab_size)
    if top_k is not None and top_k > 0:
        reshaped_logits = ref_apply_top_k(reshaped_logits, top_k)
    if top_p is not None and top_p < 1.0:
        reshaped_logits = ref_apply_top_p(reshaped_logits, top_p)
    probs = F.softmax(reshaped_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens.view(original_shape[:-1])


def main():
    torch.manual_seed(0)

    # ---- top_k ----
    x = torch.randn(4, 100)
    assert torch.equal(apply_top_k(x, 7), ref_apply_top_k(x, 7))
    assert torch.equal(apply_top_k(x, 500), ref_apply_top_k(x, 500))
    print("apply_top_k: exact match")

    # ---- top_p (loop version == optimized version == port) ----
    x = torch.randn(4, 100)
    assert torch.equal(apply_top_p(x.clone(), 0.9), ref_apply_top_p(x.clone(), 0.9))
    x2 = torch.randn(4, 100)
    assert torch.equal(apply_top_p_optimized(x2.clone(), 0.9), ref_apply_top_p(x2.clone(), 0.9))
    x2 = torch.randn(1, 1025)
    assert torch.equal(apply_top_p(x2, 0.5), ref_apply_top_p(x2, 0.5))
    print("apply_top_p: exact match (loop == optimized == port)")

    # ---- repetition penalty 3D ----
    lg = torch.randn(1, 8, 50)
    pv = torch.randint(0, 50, (1, 20, 8))
    out = apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.2)
    ref = ref_apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.2)
    assert torch.equal(out, ref)
    print("repetition_penalty [B,H,V]: exact match")

    # ---- repetition penalty 2D (text) ----
    lg = torch.randn(1, 200)
    pv = torch.randint(0, 200, (1, 30))
    out = apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.3)
    ref = ref_apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.3)
    assert torch.equal(out, ref)
    print("repetition_penalty [N,V]: exact match")

    # ---- sample_token, greedy ----
    lg = torch.randn(1, 50)
    assert torch.equal(sample_token(lg, do_sample=False),
                       ref_sample_token(lg, do_sample=False))
    # ---- sample_token, sampled, all arg combos ----
    for (tk, tp) in [(None, None), (25, 0.8), (50, 1.0), (None, 0.95), (5, None)]:
        for rep in (1.0, 1.2):
            torch.manual_seed(42)
            a = sample_token(lg.clone(), prev_tokens=torch.randint(0, 50, (1, 9)),
                             repetition_penalty=rep, top_p=tp, top_k=tk, do_sample=True)
            torch.manual_seed(42)
            b = ref_sample_token(lg.clone(), prev_tokens=torch.randint(0, 50, (1, 9)),
                                 repetition_penalty=rep, top_p=tp, top_k=tk, do_sample=True)
            assert torch.equal(a, b), (tk, tp, rep, a, b)
    # ---- audio-shaped [N, V] batch of heads, same RNG stream ----
    lg3 = torch.randn(31, 1025)
    pv3 = torch.randint(0, 1025, (1, 77, 32))
    torch.manual_seed(7)
    a = sample_token(lg3.clone(), prev_tokens=pv3, repetition_penalty=1.1,
                     top_p=0.8, top_k=25, do_sample=True)
    torch.manual_seed(7)
    b = ref_sample_token(lg3.clone(), prev_tokens=pv3, repetition_penalty=1.1,
                         top_p=0.8, top_k=25, do_sample=True)
    assert torch.equal(a, b)
    print("sample_token: exact RNG-stream match across arg combos")

    # ---- find_last_equal_C ----
    t = torch.tensor([[5, 3, 7, 3, 9]])
    assert find_last_equal_C(t, 3).item() == 3
    assert find_last_equal_C(t, 9).item() == 4
    assert find_last_equal_C(t, 42).item() == -1
    print("find_last_equal_C: OK")

    print("test_sampling PASS")


if __name__ == "__main__":
    main()
