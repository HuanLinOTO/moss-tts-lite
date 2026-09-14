"""Build MOSS-TTS direct-TTS model inputs from raw text.

Replicates the reference pipeline exactly:
    1. normalize_tts_text(text)                       (vendored normalizer)
    2. UserMessage.__post_init__ template             (processing_moss_tts.py)
       - reference=None -> "None"; every field rendered via str()
    3. chat template + generation prompt              (chat_template.jinja)
       "<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n"
    4. tokenizer.encode(content)                      (QwenBPE)
    5. unified codes [L, 1+n_vq]: channel 0 = text ids, channels 1..32 =
       audio_pad_code (1024)

Returns {"input_ids": LongTensor[1, L, 33], "attention_mask": BoolTensor[1, L]}
(all True for the single, unpadded sample).
"""

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

N_VQ = 32            # audio channels (MossTTSDelayConfig.n_vq)
AUDIO_PAD_CODE = 1024  # MossTTSDelayConfig.audio_pad_code

# UserMessage.__post_init__ template (verbatim from processing_moss_tts.py)
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
        # tokenizer.json wins over added_tokens.json: AutoTokenizer resolves to
        # the FAST tokenizer, whose added-token strings are authoritative
        # (they disagree on ids 151654/151656/151662).
        tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
        added = tok_json if os.path.exists(tok_json) else \
            os.path.join(MODEL_DIR, "added_tokens.json")
        _DEFAULT_TOKENIZER = QwenBPE(vocab, merges, added)
    return _DEFAULT_TOKENIZER


def _render_user_inst(text: str, reference=None, instruction=None, tokens=None,
                      quality=None, sound_event=None, ambient_sound=None,
                      language=None) -> str:
    """UserMessage.__post_init__ with reference=None (v1 direct TTS):
    every absent field is rendered through str() exactly like the reference."""
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
    """chat_template.jinja for one user message + generation prompt
    (verified byte-identical against jinja rendering)."""
    return (f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def build_tts_prompt(text: str, language: str | None = None,
                     tokens: int | None = None,
                     normalizer=None, tokenizer=None) -> dict:
    """Text -> TTS model inputs (single sample, no padding).

    Args:
        text: raw input text (normalized with normalize_tts_text first).
        language: optional language tag rendered into the user template.
        tokens: optional max-audio-tokens hint rendered into the template.
        normalizer: injectable normalize_tts_text (default: vendored one).
        tokenizer: injectable tokenizer exposing ``encode(str) -> list[int]``
            (default: QwenBPE on MODEL_DIR).

    Returns:
        dict with:
          "input_ids": LongTensor[1, L, 33]  (ch0=text ids, ch1..32=1024)
          "attention_mask": BoolTensor[1, L] (all True)
    """
    norm = normalizer if normalizer is not None else normalize_tts_text
    tok = tokenizer if tokenizer is not None else default_tokenizer()

    # reference pipeline: build_user_message normalizes text before the template
    text = norm(text)
    content = _render_user_inst(text, reference=None, instruction=None,
                                tokens=tokens, quality=None, sound_event=None,
                                ambient_sound=None, language=language)
    prompt_str = _apply_chat_template_user(content)
    text_ids = tok.encode(prompt_str)

    n = len(text_ids)
    text_codes = torch.tensor(text_ids, dtype=torch.long).unsqueeze(1)  # [L,1]
    audio_codes = torch.full((n, N_VQ), AUDIO_PAD_CODE, dtype=torch.long)
    unified = torch.cat([text_codes, audio_codes], dim=1)               # [L,33]
    input_ids = unified.unsqueeze(0)                                    # [1,L,33]
    attention_mask = torch.ones(1, n, dtype=torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# Continuation mode: two-turn conversation [user_text, assistant_audio_prefix].
# Mirrors MossTTSDelayProcessor(mode="continuation"); the delay/truncation
# semantics are documented in build_continuation_prompt's docstring.

# Assistant-side slot token STRINGS are the tokenizer.json added-token names
# (the fast tokenizer is authoritative; added_tokens.json names the same ids
# differently: 151656="<|video_pad|>", 151662="<|fim_pad|>" -- those strings
# are NOT atomic tokens and must not be used here).  Ids are the
# MossTTSDelayConfig defaults (configuration_moss_tts.py).
AUDIO_START_TOKEN = "<|audio_start|>"                          # 151652
AUDIO_END_TOKEN = "<|audio_end|>"                              # 151653
ASSISTANT_GEN_SLOT_TOKEN = "<|audio_assistant_gen_slot|>"      # 151656
ASSISTANT_DELAY_SLOT_TOKEN = "<|audio_assistant_delay_slot|>"  # 151662
_AUDIO_START_ID = 151652
_AUDIO_END_ID = 151653
_GEN_SLOT_ID = 151656
_DELAY_SLOT_ID = 151662


def apply_delay_pattern(codes: torch.Tensor,
                        pad_code: int = AUDIO_PAD_CODE) -> torch.Tensor:
    """Verbatim port of MossTTSDelayProcessor.apply_delay_pattern (CPU).

    codes [T, n_vq] -> delayed [T + n_vq - 1, n_vq] where delayed[i, ch] =
    codes[i - ch, ch] when 0 <= i - ch < T, else pad_code.
    """
    delayed = torch.full(
        (codes.shape[0] + codes.shape[1] - 1, codes.shape[1]),
        pad_code, dtype=codes.dtype, device=codes.device)
    for i in range(codes.shape[1]):
        delayed[i:i + codes.shape[0], i] = codes[:, i]
    return delayed


def _assistant_audio_block(n_frames: int) -> str:
    """_replace_audio_placeholders.build_audio_block for the assistant turn
    (gen_slot*n + delay_slot*(n_vq-1) between audio_start/audio_end)."""
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
    """Text + reference-audio codes -> continuation model inputs.

    Replicates, for the single-sample case::

        processor([build_user_message(text=..., language=...),
                   build_assistant_message(audio_codes_list=[prefix_codes])],
                  mode="continuation")

    NOTE (reference contract, docs/moss_tts_model_card.md): continuation-based
    cloning expects the PREFIX TRANSCRIPT to be included at the start of
    ``text`` -- pass ``prefix_transcript + new_text``.

    Args:
        text: full text (prefix transcript + content to continue with);
            normalized with normalize_tts_text first, like the reference.
        language: optional language tag rendered into the user template.
        prefix_codes: LongTensor[T, 32] codec codes of the reference audio
            (raw encoder output, one row per 80 ms frame, values [0, 1023]).
        tokens/normalizer/tokenizer: same injectables as build_tts_prompt.

    Returns:
        dict with "input_ids" LongTensor[1, L, 33] and "attention_mask"
        BoolTensor[1, L] (all True), same format as build_tts_prompt.
    """
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

    # added-token id guard: fail loudly if this tokenizer maps the slot
    # strings to anything but the config ids (naming drift guard).
    for tok_str, want in ((AUDIO_START_TOKEN, _AUDIO_START_ID),
                          (AUDIO_END_TOKEN, _AUDIO_END_ID),
                          (ASSISTANT_GEN_SLOT_TOKEN, _GEN_SLOT_ID),
                          (ASSISTANT_DELAY_SLOT_TOKEN, _DELAY_SLOT_ID)):
        got = tok.encode(tok_str)
        if got != [want]:
            raise ValueError(f"tokenizer maps {tok_str!r} to {got}, "
                             f"expected [{want}]")

    # reference pipeline: build_user_message normalizes text before the template
    text = norm(text)
    content = _render_user_inst(text, reference=None, instruction=None,
                                tokens=tokens, quality=None, sound_event=None,
                                ambient_sound=None, language=language)
    # chat_template.jinja, one message at a time, add_generation_prompt=False
    # for both turns (mode="continuation" ends on the assistant message)
    user_str = f"<|im_start|>user\n{content}<|im_end|>\n"
    asst_str = (f"<|im_start|>assistant\n"
                f"{_assistant_audio_block(n_frames)}<|im_end|>\n")

    # ---- user turn: _get_unified_codes(role="user") -> all-pad audio rows
    user_ids = tok.encode(user_str)
    n_u = len(user_ids)
    user_unified = torch.cat(
        [torch.tensor(user_ids, dtype=torch.long).unsqueeze(1),
         torch.full((n_u, N_VQ), AUDIO_PAD_CODE, dtype=torch.long)], dim=1)

    # ---- assistant turn: _get_unified_codes(role="assistant", truncation=True)
    asst_ids = tok.encode(asst_str)
    starts = [i for i, t in enumerate(asst_ids) if t == _AUDIO_START_ID]
    ends = [i for i, t in enumerate(asst_ids) if t == _AUDIO_END_ID]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"expected exactly one audio_start/audio_end in the "
                         f"assistant block, got {len(starts)}/{len(ends)}")
    audio_start_idx = starts[0]

    prefix = prefix_codes.detach().to(torch.long).cpu()
    delay_audio = apply_delay_pattern(prefix, AUDIO_PAD_CODE)   # [T+31, 32]
    pad_head = torch.full((audio_start_idx + 1, N_VQ), AUDIO_PAD_CODE,
                          dtype=torch.long)                     # rows .. audio_start
    segments = [pad_head, delay_audio]
    # truncation=True (continuation): drop the last n_vq-1 ramp-out rows
    segments[-1] = segments[-1][:-(N_VQ - 1), :]                # [T, 32]
    audio_ch = torch.cat(segments, dim=0)                       # [start+1+T, 32]

    text_ch = torch.tensor(asst_ids, dtype=torch.long).unsqueeze(1)
    if text_ch.shape[0] != audio_ch.shape[0]:
        text_ch = text_ch[:audio_ch.shape[0]]                   # drop ramp + tail
    asst_unified = torch.cat([text_ch, audio_ch], dim=1)        # [start+1+T, 33]

    unified = torch.cat([user_unified, asst_unified], dim=0)    # [L, 33]
    input_ids = unified.unsqueeze(0)                            # [1, L, 33]
    attention_mask = torch.ones(1, int(unified.shape[0]), dtype=torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
