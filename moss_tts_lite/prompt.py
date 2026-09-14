"""Build MOSS-TTS direct-TTS model inputs from raw text."""

from __future__ import annotations

import os

import torch

from moss_tts_lite.bpe import QwenBPE
from moss_tts_lite.normalizer import normalize_tts_text

__all__ = ["build_tts_prompt", "build_continuation_prompt", "apply_delay_pattern",
           "default_tokenizer", "MODEL_DIR"]

MODEL_DIR = os.environ.get(
    "MOSS_MIN_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "models", "MOSS-TTS-v1.5"),
)

N_VQ = 32
AUDIO_PAD_CODE = 1024

_USER_INST_TEMPLATE = """<user_inst>
- Reference(s):
{reference}
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""

_DEFAULT_TOKENIZER: QwenBPE | None = None

def default_tokenizer() -> QwenBPE:
    """Lazy singleton QwenBPE for MODEL_DIR (fast-tokenizer added tokens)."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        vocab = os.path.join(MODEL_DIR, "vocab.json")
        merges = os.path.join(MODEL_DIR, "merges.txt")

        tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
        added = tok_json if os.path.exists(tok_json) else \
            os.path.join(MODEL_DIR, "added_tokens.json")
        _DEFAULT_TOKENIZER = QwenBPE(vocab, merges, added)
    return _DEFAULT_TOKENIZER

def _render_user_inst(text: str, reference=None, instruction=None, tokens=None,
                      quality=None, sound_event=None, ambient_sound=None,
                      language=None) -> str:
    """UserMessage."""
    if reference is None:
        reference = "None"
    content = (
        _USER_INST_TEMPLATE
        .replace("{reference}", str(reference))
        .replace("{instruction}", str(instruction))
        .replace("{tokens}", str(tokens))
        .replace("{quality}", str(quality))
        .replace("{sound_event}", str(sound_event))
        .replace("{ambient_sound}", str(ambient_sound))
        .replace("{language}", str(language))
        .replace("{text}", str(text))
    )
    return content

def _apply_chat_template_user(content: str) -> str:
    """chat_template."""
    return (f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def build_tts_prompt(text: str, language: str | None = None,
                     tokens: int | None = None,
                     normalizer=None, tokenizer=None) -> dict:
    """Text -> TTS model inputs (single sample, no padding)."""
    norm = normalizer if normalizer is not None else normalize_tts_text
    tok = tokenizer if tokenizer is not None else default_tokenizer()

    text = norm(text)
    content = _render_user_inst(text, reference=None, instruction=None,
                                tokens=tokens, quality=None, sound_event=None,
                                ambient_sound=None, language=language)
    prompt_str = _apply_chat_template_user(content)
    text_ids = tok.encode(prompt_str)

    n = len(text_ids)
    text_codes = torch.tensor(text_ids, dtype=torch.long).unsqueeze(1)
    audio_codes = torch.full((n, N_VQ), AUDIO_PAD_CODE, dtype=torch.long)
    unified = torch.cat([text_codes, audio_codes], dim=1)
    input_ids = unified.unsqueeze(0)
    attention_mask = torch.ones(1, n, dtype=torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
ASSISTANT_GEN_SLOT_TOKEN = "<|audio_assistant_gen_slot|>"
ASSISTANT_DELAY_SLOT_TOKEN = "<|audio_assistant_delay_slot|>"
_AUDIO_START_ID = 151652
_AUDIO_END_ID = 151653
_GEN_SLOT_ID = 151656
_DELAY_SLOT_ID = 151662

def apply_delay_pattern(codes: torch.Tensor,
                        pad_code: int = AUDIO_PAD_CODE) -> torch.Tensor:
    """Verbatim port of MossTTSDelayProcessor."""
    delayed = torch.full(
        (codes.shape[0] + codes.shape[1] - 1, codes.shape[1]),
        pad_code, dtype=codes.dtype, device=codes.device)
    for i in range(codes.shape[1]):
        delayed[i:i + codes.shape[0], i] = codes[:, i]
    return delayed

def _assistant_audio_block(n_frames: int) -> str:
    """_replace_audio_placeholders."""
    if n_frames == 0:
        return f"{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}"
    return (f"{AUDIO_START_TOKEN}"
            f"{ASSISTANT_GEN_SLOT_TOKEN * n_frames}"
            f"{ASSISTANT_DELAY_SLOT_TOKEN * (N_VQ - 1)}"
            f"{AUDIO_END_TOKEN}")

def build_continuation_prompt(text: str, language: str | None = None,
                              prefix_codes: torch.Tensor | None = None,
                              tokens: int | None = None,
                              normalizer=None, tokenizer=None) -> dict:
    """Text + reference-audio codes -> continuation model inputs."""
    if (not isinstance(prefix_codes, torch.Tensor) or prefix_codes.dim() != 2
            or prefix_codes.shape[1] != N_VQ):
        raise ValueError(
            f"prefix_codes must be a Tensor[T, {N_VQ}], got "
            f"{type(prefix_codes).__name__} "
            f"{tuple(prefix_codes.shape) if isinstance(prefix_codes, torch.Tensor) else ''}")
    n_frames = int(prefix_codes.shape[0])
    if n_frames < N_VQ:
        raise ValueError(f"prefix_codes needs >= {N_VQ} rows for the delay "
                         f"ramp, got {n_frames}")
    norm = normalizer if normalizer is not None else normalize_tts_text
    tok = tokenizer if tokenizer is not None else default_tokenizer()

    for tok_str, want in ((AUDIO_START_TOKEN, _AUDIO_START_ID),
                          (AUDIO_END_TOKEN, _AUDIO_END_ID),
                          (ASSISTANT_GEN_SLOT_TOKEN, _GEN_SLOT_ID),
                          (ASSISTANT_DELAY_SLOT_TOKEN, _DELAY_SLOT_ID)):
        got = tok.encode(tok_str)
        if got != [want]:
            raise ValueError(f"tokenizer maps {tok_str!r} to {got}, "
                             f"expected [{want}]")

    text = norm(text)
    content = _render_user_inst(text, reference=None, instruction=None,
                                tokens=tokens, quality=None, sound_event=None,
                                ambient_sound=None, language=language)

    user_str = f"<|im_start|>user\n{content}<|im_end|>\n"
    asst_str = (f"<|im_start|>assistant\n"
                f"{_assistant_audio_block(n_frames)}<|im_end|>\n")

    user_ids = tok.encode(user_str)
    n_u = len(user_ids)
    user_unified = torch.cat(
        [torch.tensor(user_ids, dtype=torch.long).unsqueeze(1),
         torch.full((n_u, N_VQ), AUDIO_PAD_CODE, dtype=torch.long)], dim=1)

    asst_ids = tok.encode(asst_str)
    starts = [i for i, t in enumerate(asst_ids) if t == _AUDIO_START_ID]
    ends = [i for i, t in enumerate(asst_ids) if t == _AUDIO_END_ID]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"expected exactly one audio_start/audio_end in the "
                         f"assistant block, got {len(starts)}/{len(ends)}")
    audio_start_idx = starts[0]

    prefix = prefix_codes.detach().to(torch.long).cpu()
    delay_audio = apply_delay_pattern(prefix, AUDIO_PAD_CODE)
    pad_head = torch.full((audio_start_idx + 1, N_VQ), AUDIO_PAD_CODE,
                          dtype=torch.long)
    segments = [pad_head, delay_audio]

    segments[-1] = segments[-1][:-(N_VQ - 1), :]
    audio_ch = torch.cat(segments, dim=0)

    text_ch = torch.tensor(asst_ids, dtype=torch.long).unsqueeze(1)
    if text_ch.shape[0] != audio_ch.shape[0]:
        text_ch = text_ch[:audio_ch.shape[0]]
    asst_unified = torch.cat([text_ch, audio_ch], dim=1)

    unified = torch.cat([user_unified, asst_unified], dim=0)
    input_ids = unified.unsqueeze(0)
    attention_mask = torch.ones(1, int(unified.shape[0]), dtype=torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
