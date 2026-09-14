"""Delay-pattern generation loop, ported line-by-line from models/MOSS-TTS-v1."""

from dataclasses import dataclass

import torch

from .model import (
    AUDIO_END_TOKEN_ID,
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    AUDIO_DELAY_SLOT_TOKEN_ID,
    IM_END_TOKEN_ID,
    N_VQ,
    PAD_TOKEN_ID,
)
from .sampling import find_last_equal_C, sample_token

_INT64_MAX = 9223372036854775807

@dataclass
class GenResult:
    text_ids: torch.Tensor
    audio_frames: torch.Tensor
    finished: bool
    n_steps: int

@torch.inference_mode()
def generate(
    model,
    prompt: dict,
    max_new_tokens=4096,
    seed=1234,
    greedy=False,
    text_temperature=1.5,
    text_top_p=1.0,
    text_top_k=50,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
    stats: dict | None = None,
) -> GenResult:
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")
    device = model.device
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.dim() != 3 or input_ids.shape[0] != 1 or input_ids.shape[-1] != N_VQ + 1:
        raise ValueError(f"expected prompt input_ids [1, L, {N_VQ + 1}], got {tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones(1, int(input_ids.shape[1]),
                                    dtype=torch.bool, device=device)
    else:
        am = attention_mask.to(device)
        if am.dim() == 1:
            am = am.unsqueeze(0)
        if tuple(am.shape) != (1, input_ids.shape[1]):
            raise ValueError(f"bad attention_mask shape {tuple(am.shape)}")
        if not bool(am.all()):
            raise NotImplementedError("batch=1 generate() requires an all-True prompt mask (no left padding)")
        attention_mask = am

    seq_len = int(input_ids.shape[1])

    _stats = stats if (stats is not None and device.type == "cuda") else None

    def _t0():
        if _stats is None:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _t1(e0, slot):
        if _stats is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        if slot == "prefill":
            _stats["prefill"] = (e0, e)
        else:
            _stats.setdefault("step_ms", []).append((e0, e))
    if seq_len + max_new_tokens > model.max_seq_len:
        raise ValueError(f"prompt {seq_len} + max_new_tokens {max_new_tokens} "
                         f"exceeds KV cache {model.max_seq_len}")

    if greedy:
        text_temperature = 0
        audio_temperature = 0
    if text_temperature > 0:
        text_do_sample = True
    else:
        text_temperature = 1
        text_do_sample = False
    if audio_temperature > 0:
        audio_do_sample = True
    else:
        audio_temperature = 1
        audio_do_sample = False

    torch.manual_seed(seed)

    batch_size = 1
    n_vq = N_VQ
    torch_int64_max = _INT64_MAX

    generation_ids = input_ids.clone()
    is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)

    audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
    delayed_lengths = torch.full((batch_size,), torch_int64_max, dtype=torch.int64, device=device)

    is_continuation = (input_ids[:, -1, 0] == AUDIO_START_TOKEN_ID) | (
        input_ids[:, -1, 0] == AUDIO_GEN_SLOT_TOKEN_ID)
    audio_start_indices = find_last_equal_C(input_ids[..., 0], AUDIO_START_TOKEN_ID)
    audio_start_mask = is_continuation & (audio_start_indices != -1)
    audio_lengths[audio_start_mask] = seq_len - audio_start_indices[audio_start_mask]

    is_audio = audio_start_mask.clone()

    pre_exclude_mask0 = torch.tensor(
        [PAD_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID],
        device=device)
    pre_exclude_mask1 = torch.ones(model.text_vocab, device=device, dtype=torch.bool)
    pre_exclude_mask1[[AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID]] = False

    text_steps: list = []
    audio_steps: list = []
    n_steps = 0
    current_input_ids = None

    for time_step in range(max_new_tokens):

        e0 = _t0()
        hs = model.prefill(input_ids) if time_step == 0 else model.step(current_input_ids)
        _t1(e0, "prefill" if time_step == 0 else "step")
        h = hs.last_hidden[:, -1]

        next_text_token = torch.full((batch_size,), PAD_TOKEN_ID, device=device)
        next_text_token[~is_stopping & (delayed_lengths < n_vq)] = AUDIO_DELAY_SLOT_TOKEN_ID
        is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
        next_text_token[is_audio_eos] = AUDIO_END_TOKEN_ID
        is_audio[is_audio_eos] = False
        sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)

        next_audio_tokens = torch.full((batch_size, n_vq), AUDIO_PAD_CODE, device=device)
        ar = torch.arange(n_vq, device=device)
        pre_audio_mask = audio_lengths.unsqueeze(1) > ar.unsqueeze(0).expand(batch_size, n_vq)
        post_audio_mask = ar.unsqueeze(0).expand(batch_size, n_vq) > delayed_lengths.unsqueeze(1) - 1
        post_audio_mask[delayed_lengths == torch_int64_max] = True
        sampling_audio_mask = pre_audio_mask & post_audio_mask
        next_audio_tokens[~sampling_audio_mask] = AUDIO_PAD_CODE

        if bool(sampling_text_mask[0]):
            if bool(is_audio[0]) and not text_do_sample:

                two = model.text_logits_2way(h) / text_temperature
                if time_step == 0:
                    two[1] = float("-inf")
                pick = int(torch.argmax(two))
                next_text_token[sampling_text_mask] = (
                    AUDIO_GEN_SLOT_TOKEN_ID if pick == 0 else AUDIO_DELAY_SLOT_TOKEN_ID)
            else:
                lt = model.text_logits(h) / text_temperature
                if bool(is_audio[0]):
                    lt = lt.masked_fill(pre_exclude_mask1, float("-inf"))
                else:
                    lt = lt.index_fill(0, pre_exclude_mask0, float("-inf"))
                if time_step == 0:
                    lt[AUDIO_DELAY_SLOT_TOKEN_ID] = float("-inf")
                if time_step <= n_vq:
                    lt[IM_END_TOKEN_ID] = float("-inf")
                tok = sample_token(lt.view(1, -1), top_p=text_top_p, top_k=text_top_k,
                                   do_sample=text_do_sample)
                next_text_token[sampling_text_mask] = int(tok[0])
        is_audio[next_text_token == AUDIO_START_TOKEN_ID] = True
        is_stopping[next_text_token == IM_END_TOKEN_ID] = True

        if bool(sampling_audio_mask[0].any()):
            audio_logit = model.audio_logits(h) / audio_temperature
            if bool(sampling_audio_mask[0, 0]):
                audio_ch0_logits = audio_logit[0].view(1, -1)
                audio_ch0_logits[..., AUDIO_PAD_CODE] = float("-inf")
                tok0 = sample_token(
                    logits=audio_ch0_logits,
                    prev_tokens=generation_ids[:, :, 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p, top_k=audio_top_k, do_sample=audio_do_sample)
                next_audio_tokens[:, 0][sampling_audio_mask[:, 0]] = int(tok0[0])
            rest_idx = [j for j in range(1, n_vq) if bool(sampling_audio_mask[0, j])]
            if rest_idx:
                audio_logits_rest = audio_logit[rest_idx]
                audio_logits_rest[..., AUDIO_PAD_CODE] = float("-inf")
                tok = sample_token(
                    logits=audio_logits_rest,
                    prev_tokens=generation_ids[:, :, 2:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p, top_k=audio_top_k, do_sample=audio_do_sample)
                for k, j in enumerate(rest_idx):
                    next_audio_tokens[0, j] = int(tok[k])

        audio_lengths[(next_text_token == AUDIO_START_TOKEN_ID)
                      | (next_text_token == AUDIO_GEN_SLOT_TOKEN_ID)
                      | (next_text_token == AUDIO_DELAY_SLOT_TOKEN_ID)] += 1
        audio_lengths[next_text_token == AUDIO_END_TOKEN_ID] = 0
        delayed_lengths[(delayed_lengths == torch_int64_max)
                        & (next_text_token == AUDIO_DELAY_SLOT_TOKEN_ID)] = 0
        delayed_lengths[delayed_lengths != torch_int64_max] += 1
        delayed_lengths[delayed_lengths > n_vq] = torch_int64_max

        current_input_ids = torch.cat(
            [next_text_token[:, None, None], next_audio_tokens[:, None, :]], dim=2)
        attention_mask = torch.cat([attention_mask, (~is_stopping).unsqueeze(-1)], dim=-1)
        generation_ids = torch.cat([generation_ids, current_input_ids], dim=1)

        text_steps.append(int(next_text_token[0]))
        audio_steps.append(next_audio_tokens[0].clone())
        n_steps += 1

        if bool(is_stopping.all()):
            break

    text_ids = (torch.tensor(text_steps, dtype=torch.long) if text_steps
                else torch.empty(0, dtype=torch.long))
    audio_frames = (torch.stack(audio_steps) if audio_steps
                    else torch.empty(0, n_vq, dtype=torch.long))
    if _stats is not None:
        torch.cuda.synchronize()
        p = _stats.get("prefill")
        _stats["prefill_ms"] = float(p[0].elapsed_time(p[1])) if p else None
        _stats["step_ms"] = [float(a.elapsed_time(b))
                             for a, b in _stats.get("step_ms", [])]
        _stats["steps"] = len(_stats["step_ms"])
    return GenResult(text_ids=text_ids.cpu(), audio_frames=audio_frames.cpu(),
                     finished=bool(is_stopping.all()), n_steps=n_steps)
