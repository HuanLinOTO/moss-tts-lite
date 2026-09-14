"""Standalone (self-contained) export of a GPTQ-quantized MOSS-TTS checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

import numpy as np
import torch

from .fast import _LIN_NAMES
from .gptq import GptqMossTTS
from .model import MossTTSModel
from .st_loader import read_safetensors, safetensors_header

__all__ = [
    "FORMAT_TAG", "Q_FILE", "META_FILE", "base_weight_key", "export_standalone",
    "write_safetensors", "read_standalone", "load_standalone_model",
    "is_standalone_dir", "standalone_presets", "STANDALONE_SCHEMA",
]

FORMAT_TAG = "moss_tts_lite.standalone-gptq"
FORMAT_VERSION = 1
Q_FILE = "quantized.safetensors"
META_FILE = "meta.json"
README_FILE = "README.md"
DEFAULT_BASE_REPO = "OpenMOSS-Team/MOSS-TTS-v1.5"
BASE_COLLECTION_MS = "OpenMOSS-Team/MOSS-TTS"

SMALL_FILES = (

    "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
    "added_tokens.json", "special_tokens_map.json", "chat_template.jinja",

    "config.json", "configuration.json", "processor_config.json",
    "configuration_moss_tts.py", "modeling_moss_tts.py",
    "processing_moss_tts.py", "tts_robust_normalizer_single_script.py",
    "inference_utils.py",
    ".gitattributes",
)
LICENSE_FILE = "LICENSE"

_PROJ = {
    "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
    "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
_Q_RE = re.compile(r"^layers\.(\d+)\.(" + "|".join(_LIN_NAMES) + r")\.(q|qsz)$")

_GITATTRIBUTES_FALLBACK = "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"

PRESET_METRICS: dict[str, dict] = {
    "w1": {
        "label": "default tier", "group_size": 32, "bf16_keeps": "v_proj (36 layers)",
        "gate2_audio_top25_pct": 89.30, "gate2_text_argmax_pct": 100.0,
        "gate2_audio_mean_abs_d": 0.389, "steps_per_s_steady": 81.5,
        "vram_peak_gib": 8.28, "runaway": "0/12",
    },
    "w2": {
        "label": "quality tier", "group_size": 32,
        "bf16_keeps": "v_proj (36 layers) + up_proj (layers 6,7,9,35)",
        "gate2_audio_top25_pct": 90.55, "gate2_text_argmax_pct": 100.0,
        "gate2_audio_mean_abs_d": 0.384, "steps_per_s_steady": 78.6,
        "vram_peak_gib": 8.53, "runaway": "0/12",
    },
}
METRICS_SOURCE = ("gates 1/2/4; A10G-24G, "
                  "torch 2.9.1; speed floor 78 steps/s, VRAM ceiling 12 GiB)")

DEFAULT_CROSS_LANG_SECTION = r"""### 跨语言验证 / Cross-language validation

基座声明支持 31 种语言；本档已在其中 **10 种语言 × 4 条轨迹 = 40 条轨迹**上完成 teacher-forced
跨语言验收（zh / en / fr / ja / de / ko / yue / ru / es / ar，覆盖**拉丁、汉字、西里尔、阿拉伯、韩文**
五种文字系统），指标为 tie-robust `cover` 与 mean\|Δlogit\|（并列口径见发布仓库文档）：

| 项 / Item | 值 / Value |
|---|---|
| 轨迹一致改善 / trajectories improving | **40/40**（mean\|Δlogit\| 在 10 种语言上全部同向下降） |
| 合并增益 merged gain（w2 相对 w1，tie-robust `cover`，10 语言等权，轨迹自助 95% CI） | **+0.641 pt [+0.59, +0.69]** |
| 音频 logit 平均绝对偏差变化 / mean\|Δlogit\| change（w2 相对 w1，越低越好） | **−0.341 [−0.371, −0.303]** |
| runaway 观测 / runaway observations（bf16 直生成，看门狗开启） | **0/25，25/25 正常收尾**（含本项目首个 RTL 语言：阿拉伯语） |
| 本档水平 / this preset's level on the same corpus | 见发布仓库文档 / see the release-repo documentation |

*Cross-language validation: 10 languages × 4 trajectories = 40 trajectories (zh/en/fr/ja/de/ko/yue/ru/es/ar,
spanning Latin, Han, Cyrillic, Arabic and Hangul scripts). **40/40 trajectories improve** under the
tie-robust `cover` metric and mean |Δlogit| falls in all 10 languages; merged gain of `w2` over `w1`
(equal language weight, trajectory bootstrap 95 % CI) is `cover` **+0.641 pt [+0.59, +0.69]** and
mean |Δlogit| **−0.341 [−0.371, −0.303]**; runaway **0/25** (25/25 finished, watchdog on, including
this project's first RTL language, Arabic). Both metrics are tie-robust — the magnitudes are not
comparable to the 40-step gate number in §3. Method, per-language detail and residual attribution:
the quantization acceptance documentation (langcheck-1) in the release repository. **Honest scope:**
the other 21 of the 31 declared languages were not individually verified, and this result is not
extrapolated to them.*"""

DEFAULT_TIE_CONVENTION_SECTION = r"""### 3.1 关于 top-25 数字的口径（重要）

**音频 top-25 命中率高度依赖并列（tie）的打破方式，报数必须注明口径。**
参考分布上 63.9% 的声道 top1==top2、20.1% 的声道有 ≥25 个值并列在最大值；
在这种分布上"参考 argmax 是否落在候选的前 25 大之内"这句话本身是不唯一的：
同一份 logits，仅改并列打破方式，分数可以从 80.2% 变到 100%（一个与参考**逐位相同**的
模型也会被打成 86.8%）。本卡因此同时报两个口径：

* **tie-robust `cover`**（推荐）：取候选的前 25 大**集合**，看参考 argmax 的值是否 ≥ 第 25 大的值
  —— 并列一律算命中，与任何排序实现无关，可复现、可跨实现比较；
* **CUDA `topk` 口径**：历史上的门禁口径（`torch.topk` 在 CUDA 上的并列顺序），
  与旧版本报告可直接对照。

两个口径的方向一致；**幅度不可跨口径比较**（tie 主导时旧口径会把同等真实增益放大约 7 倍）。
两个口径方向一致；幅度不可跨口径比较。"""

DEFAULT_TOP25_LABEL = "teacher-forced 40 步 / 1280 位置"

DEFAULT_QUALITY_TRADEOFF_SENTENCE = "音频 top-25 命中率 {top25}%，"
DEFAULT_EN_SUMMARY_METRICS = (
    "\n**{top25}%**, text argmax {text_argmax}%, **{speed} steps/s**, "
    "**{vram} GiB**\nresident VRAM, runaway **{runaway}** "
    "(12 seed long-form battery, watchdog off),\n{size_gib} GiB on disk.")
STANDALONE_SCHEMA = {
    "format": FORMAT_TAG,
    "format_version": FORMAT_VERSION,
    "weight_file": Q_FILE,
    "quantized_key_template": "layers.{layer}.{proj}.q  /  .qsz",
    "bf16_keys": "original MOSS-TTS-v1.5 checkpoint key names",
    "loader_api": "moss_tts_lite.export.load_standalone_model(dir) -> (model, GptqMossTTS)",
}

def base_weight_key(li: int, name: str) -> str:
    """Original checkpoint key of backbone projection `name` in layer `li`."""
    if name not in _PROJ:
        raise KeyError(f"unknown projection {name!r} (expected one of {_LIN_NAMES})")
    return f"language_model.layers.{int(li)}.{_PROJ[name]}.weight"

_ST_DTYPE: dict[torch.dtype, str] = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
    torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
    torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
    torch.bool: "BOOL",
}
_WRITE_CHUNK = 1 << 26

def _u8_view(t: torch.Tensor) -> np.ndarray:
    """Flat uint8 numpy view of a contiguous tensor (byte-exact payload)."""
    if not t.is_contiguous():
        t = t.contiguous()
    return t.view(torch.uint8).reshape(-1).numpy()

def write_safetensors(path: str, entries, metadata: dict | None = None,
                      ) -> tuple[int, str]:
    """Write `entries` (iterable of (name, tensor)) as one safetensors file."""
    entries = [(n, t) for n, t in entries]
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for name, t in entries:
        if t.dtype not in _ST_DTYPE:
            raise TypeError(f"cannot serialize dtype {t.dtype} for {name!r}")
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": _ST_DTYPE[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad

    h = hashlib.sha256()
    total = 8 + len(blob)
    with open(path, "wb") as f:
        pre = struct.pack("<Q", len(blob)) + blob
        f.write(pre)
        h.update(pre)
        for name, t in entries:
            arr = _u8_view(t)
            for off in range(0, arr.size, _WRITE_CHUNK):
                chunk = arr[off:off + _WRITE_CHUNK]
                f.write(chunk)
                h.update(chunk)
        total += offset
        f.flush()
        os.fsync(f.fileno())
    return total, h.hexdigest()

def _sha256_file(path: str) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            size += len(b)
            h.update(b)
    return size, h.hexdigest()

def _hash_cache_path(repo_root: str) -> str:
    """Cache for base-checkpoint hashes (created on first use)."""
    return os.path.join(repo_root, ".cache", "base_hashes.json")

def base_provenance(model_dir: str, hash_base: bool = True) -> dict:
    """{repo_id, local_dir, files:"""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    shards: list[str] = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        n_keys = len(weight_map)
    else:
        shards = ["model.safetensors"]
        n_keys = None
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rel = os.path.relpath(os.path.abspath(model_dir), repo_root).replace(os.sep, "/")
    except ValueError:
        rel = os.path.abspath(model_dir)
    prov = {"repo_id": DEFAULT_BASE_REPO, "local_dir": rel, "n_keys": n_keys,
        "shards": [{"name": s} for s in shards], "hashed": bool(hash_base)}
    if not hash_base:
        return prov
    cache_path = _hash_cache_path(repo_root)
    cache: dict = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except (OSError, json.JSONDecodeError):
            cache = {}
    todo = []
    for name in list(shards) + (["model.safetensors.index.json"] if n_keys else []):
        p = os.path.join(model_dir, name)
        st = os.stat(p)
        key = os.path.abspath(p)
        rec = cache.get(key)
        if rec and rec.get("bytes") == st.st_size and rec.get("mtime") == int(st.st_mtime):
            continue
        todo.append((key, p, st))
    for key, p, st in todo:
        size, sha = _sha256_file(p)
        cache[key] = {"bytes": size, "mtime": int(st.st_mtime), "sha256": sha}
        print(f"[export] hashed base {os.path.basename(p)} "
              f"({size / 2**30:.2f} GiB) {sha[:16]}…", flush=True)
    if todo:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
    for rec in prov["shards"]:
        p = os.path.abspath(os.path.join(model_dir, rec["name"]))
        c = cache[p]
        rec.update({"bytes": c["bytes"], "sha256": c["sha256"]})
    if n_keys:
        c = cache[os.path.abspath(index_path)]
        prov["index_sha256"] = c["sha256"]
    return prov

def _state_meta(gptq_state_path: str) -> dict:
    meta_path = gptq_state_path + ".meta.json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — the GPTQ state is not self-describing "
            f"(per-linear group sizes / bf16 keeps); refusing to export.")
    return json.load(open(meta_path))

def _tensor_order(model_dir: str, n_layers: int):
    """Deterministic output order:"""
    yield ("language_model.embed_tokens.weight", None)
    yield ("language_model.norm.weight", None)
    for li in range(n_layers):
        yield (f"language_model.layers.{li}.input_layernorm.weight", None)
        for name in _LIN_NAMES:
            yield (base_weight_key(li, name), (li, name))
        for tail in ("self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "post_attention_layernorm.weight"):
            yield (f"language_model.layers.{li}.{tail}", None)
    for i in range(32):
        yield (f"emb_ext.{i}.weight", None)
    for i in range(33):
        yield (f"lm_heads.{i}.weight", None)

def export_standalone(model_dir: str, gptq_state_path: str, out_dir: str,
                      preset_name: str, *, metrics: dict | None = None,
                      hash_base: bool = True, template_path: str | None = None,
                      license_path: str | None = None, force: bool = False,
                      dry_run: bool = False, base_repo: str = DEFAULT_BASE_REPO,
                      quiet: bool = False) -> dict:
    """Export a self-contained model directory."""
    def log(msg: str) -> None:
        if not quiet:
            print(f"[export] {msg}", flush=True)

    t_start = time.time()
    smeta = _state_meta(gptq_state_path)
    gmap: dict[tuple[int, str], int] = {}
    for key, val in smeta.get("group_size_map", {}).items():
        a, b = key.split(":")
        gmap[(int(a), b)] = int(val)
    bf16_linears = [(int(x.split(":")[0]), x.split(":")[1])
                    for x in smeta.get("bf16_linears", [])]
    bf16_layers = [int(x) for x in smeta.get("bf16_layers", [])]
    keep_keys = {base_weight_key(li, n) for li, n in bf16_linears}
    keep_keys |= {base_weight_key(li, n) for li in bf16_layers for n in _LIN_NAMES}
    group_default = max(set(gmap.values()), key=list(gmap.values()).count) if gmap else 128

    state = torch.load(gptq_state_path, map_location="cpu", weights_only=False)
    n_layers = len(state)
    quantized: dict[str, tuple[int, str]] = {}
    records: dict[str, dict] = {}
    mode_counts: dict[int, int] = {}
    for li in sorted(state):
        for name in _LIN_NAMES:
            rec = state[li].get(name)
            if rec is None:
                if base_weight_key(li, name) not in keep_keys:
                    raise KeyError(
                        f"state has no entry for layer {li} {name}: neither "
                        f"quantized nor listed in bf16_linears — refusing to "
                        f"export an incomplete model")
                continue
            packed, qsz = rec["packed"], rec["qsz"]
            if not packed.is_contiguous():
                packed = packed.contiguous()
            if not qsz.is_contiguous():
                qsz = qsz.contiguous()
            if packed.dtype != torch.int32:
                packed = packed.to(torch.int32)
            if qsz.dtype != torch.bfloat16:
                qsz = qsz.to(torch.bfloat16)
            records[f"layers.{li}.{name}.q"] = packed
            records[f"layers.{li}.{name}.qsz"] = qsz
            quantized[base_weight_key(li, name)] = (li, name)
            g = int(gmap.get((li, name), group_default))
            mode_counts[g] = mode_counts.get(g, 0) + 1
    log(f"state: {n_layers} layers, {len(quantized)} quantized projections, "
        f"{len(keep_keys)} bf16 keeps (groups {mode_counts})")

    want = [name for name, _ in _tensor_order(model_dir, n_layers)
            if name not in quantized]
    log(f"reading {len(want)} bf16 tensors from {model_dir} (mmap) …")
    base = read_safetensors(model_dir, names=want)

    entries: list[tuple[str, torch.Tensor]] = []
    shapes: dict[str, list[int]] = {}
    for name, qkey in _tensor_order(model_dir, n_layers):
        if qkey is not None:
            li, pname = qkey
            if name in quantized:
                entries.append((f"layers.{li}.{pname}.q", records[f"layers.{li}.{pname}.q"]))
                entries.append((f"layers.{li}.{pname}.qsz", records[f"layers.{li}.{pname}.qsz"]))
                shapes[f"layers.{li}.{pname}.q"] = list(records[f"layers.{li}.{pname}.q"].shape)
                shapes[f"layers.{li}.{pname}.qsz"] = list(records[f"layers.{li}.{pname}.qsz"].shape)
                continue
        t = base.get(name)
        if t is None:
            raise KeyError(f"{name!r} missing from {model_dir}")
        entries.append((name, t))
        shapes[name] = list(t.shape)

    def _shape_from_record(li: int, name: str) -> list[int]:
        rec = records[f"layers.{li}.{name}.qsz"]
        g = int(gmap.get((li, name), group_default))
        return [int(rec.shape[1]), int(rec.shape[0]) * g]

    emb_shape = shapes["language_model.embed_tokens.weight"]
    hidden = int(emb_shape[1])
    l0q = _shape_from_record(0, "q")
    l0k = _shape_from_record(0, "k")
    chk = read_safetensors(model_dir, names=[base_weight_key(0, "q"),
                                            base_weight_key(0, "k")])
    for nm, want in (("q", l0q), ("k", l0k)):
        got = list(chk[base_weight_key(0, nm)].shape)
        if got != want:
            raise ValueError(f"state/base mismatch for layer-0 {nm}_proj: state "
                             f"implies {want}, base checkpoint has {got}")
    del chk
    hd = int(shapes["language_model.layers.0.self_attn.q_norm.weight"][0])
    model_config = {"hidden_size": hidden, "head_dim": hd,
                    "n_heads": int(l0q[0]) // hd, "n_kv_heads": int(l0k[0]) // hd,
                    "text_vocab": int(emb_shape[0]), "n_layers": n_layers,
                    "max_seq_len": 8192,
                    "layer0_q_proj_shape": l0q, "layer0_k_proj_shape": l0k}
    del base, records, state

    quant_bytes = sum(t.numel() * t.element_size() for n, t in entries
                      if _Q_RE.match(n))
    total_bytes = sum(t.numel() * t.element_size() for _, t in entries)
    meta = {
        "format": FORMAT_TAG,
        "format_version": FORMAT_VERSION,
        "schema": STANDALONE_SCHEMA,
        "preset_name": preset_name,
        "presets": {preset_name: {
            "label": (metrics or PRESET_METRICS.get(preset_name, {})).get("label"),
            "group_size_default": int(group_default),
            "group_size_map": {f"{li}:{n}": int(gmap.get((li, n), group_default))
                               for (li, n) in sorted(quantized.values())},
            "bf16_linears": [f"{li}:{n}" for li, n in sorted(bf16_linears)],
            "bf16_layers": sorted(bf16_layers),
            "n_quantized_linears": len(quantized),
            "n_bf16_entries": len(keep_keys),
            "metrics": metrics if metrics is not None
                       else PRESET_METRICS.get(preset_name, {}),
        }},
        "metrics_source": METRICS_SOURCE,
        "quantization": {
            "method": "gptq",
            "weight_bits": 4,
            "inner_k_tiles": 8,
            "kernel": "torch._weight_int4pack_mm",
            "q_dtype": "I32", "qsz_dtype": "BF16",
            "note": "int4 payload as produced by "
                    "aten._convert_weight_to_int4pack (odd|even<<4 nibble order)",
        },
        "model_config": model_config,
        "tensor_shapes": shapes,
        "n_tensors": len(entries),
        "quantized_bytes": quant_bytes,
        "weight_bytes": total_bytes,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_epoch": int(time.time()),
        "tool": {"name": "moss_tts_lite.export.export_standalone",
                 "repo": "OpenMOSS/MOSS-TTS + moss_tts_lite minimal runtime",
                 "torch": torch.__version__, "numpy": np.__version__},
        "state": {"source_file": os.path.basename(gptq_state_path),
                  "source_bytes": os.path.getsize(gptq_state_path),
                  "sha256": _sha256_file(gptq_state_path)[1]},
        "base": base_provenance(model_dir, hash_base=hash_base),
        "runtime": {"python": ">=3.10", "packages": ["torch", "numpy",
                                                     "soundfile", "pyyaml"],
                    "cli": f"python -m moss_tts_lite \"text\" -o out.wav "
                           f"--model-dir <this dir>",
                    "codec_repo": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
                    "note": "the audio codec is a separate checkpoint "
                            "(--codec-dir); CUDA required (int4 kernel)"},
        "license": "Apache-2.0 (inherited from the base model)",
        "base_repo": base_repo,
    }
    log(f"inventory: {len(entries)} tensors, {total_bytes / 2**30:.2f} GiB "
        f"(int4 payload {quant_bytes / 2**30:.2f} GiB)")
    if dry_run:
        log("dry-run: nothing written")
        return meta

    os.makedirs(out_dir, exist_ok=True)
    qpath = os.path.join(out_dir, Q_FILE)
    if os.path.exists(qpath) and not force:
        raise FileExistsError(f"{qpath} exists (use force=True to overwrite)")
    tmp = qpath + ".part"

    size, digest = write_safetensors(
        tmp, entries, metadata={"format": FORMAT_TAG, "preset": preset_name})
    os.replace(tmp, qpath)
    meta["weight_file"] = {"name": Q_FILE, "bytes": size, "sha256": digest,
                           "tensors": len(entries)}
    with open(qpath + ".sha256", "w") as f:
        f.write(f"{digest}  {Q_FILE}\n")
    log(f"wrote {Q_FILE}: {size / 2**30:.2f} GiB sha256 {digest[:16]}…")

    copied = []
    for name in SMALL_FILES:
        src = os.path.join(model_dir, name)
        if not os.path.isfile(src):
            log(f"  skip missing {name}")
            continue
        shutil.copy2(src, os.path.join(out_dir, name))
        copied.append(name)
    if ".gitattributes" not in copied:
        with open(os.path.join(out_dir, ".gitattributes"), "w") as f:
            f.write(_GITATTRIBUTES_FALLBACK)
        copied.append(".gitattributes")
    lic = license_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), LICENSE_FILE)
    if os.path.isfile(lic):
        shutil.copy2(lic, os.path.join(out_dir, LICENSE_FILE))
        copied.append(LICENSE_FILE)
    meta["files"] = {name: {"bytes": os.path.getsize(os.path.join(out_dir, name)),
                            "sha256": _sha256_file(os.path.join(out_dir, name))[1]}
                     for name in sorted(copied)}

    tpl = template_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "README_hf.md")
    if os.path.isfile(tpl):
        card = render_model_card(open(tpl, encoding="utf-8").read(), meta, preset_name)
        with open(os.path.join(out_dir, README_FILE), "w", encoding="utf-8") as f:
            f.write(card)
        meta["files"][README_FILE] = {
            "bytes": os.path.getsize(os.path.join(out_dir, README_FILE)),
            "sha256": _sha256_file(os.path.join(out_dir, README_FILE))[1]}
    else:
        log(f"WARNING: model-card template {tpl} not found; no README.md written")

    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    hdr = safetensors_header(qpath)
    got = len([k for k in hdr if k != "__metadata__"])
    if got != len(entries):
        raise RuntimeError(f"read-back mismatch: {got} tensors vs {len(entries)}")
    probe = read_safetensors(qpath, names=[entries[0][0],
                                           next(n for n, _ in entries
                                                if _Q_RE.match(n))])
    log(f"self-check OK: header {got} tensors, probe "
        f"{[(k, tuple(v.shape), str(v.dtype)) for k, v in probe.items()]}")
    log(f"done in {time.time() - t_start:.1f}s -> {out_dir}")
    return meta

def _fmt_metric(val, fmt: str) -> str:
    """Format an optional metric;"""
    if val is None:
        return "n/a"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "n/a"

def _as_section(text, lead: str = "\n") -> str:
    """Normalize a conditional section to ``lead + block`` ("" when absent)."""
    body = str(text or "").strip("\n")
    return lead + body if body else ""

def render_model_card(template: str, meta: dict, preset_name: str) -> str:
    """Substitute {{PLACEHOLDER}}s;"""
    preset = meta["presets"][preset_name]
    m = preset.get("metrics") or {}
    q = meta["quantization"]
    base = meta["base"]
    shards = base.get("shards", [])
    base_sha = "\n".join(
        f"| `{s['name']}` | {s.get('bytes', 0) / 2**30:.2f} GiB | "
        f"`{s.get('sha256', 'not hashed')[:16]}…` |" for s in shards)
    keeps = (f"{len(preset['bf16_linears'])} projections kept bf16"
             + (f" — {m.get('bf16_keeps')}" if m.get("bf16_keeps") else ""))

    has_tie = "tie_robust_cover_pct" in m
    top25 = f"{float(m.get('gate2_audio_top25_pct', float('nan'))):.2f}"
    if has_tie:
        tie_rows = "\n".join([
            "| **音频 top-25 命中率（tie-robust `cover`）**：主指标，40 步 / 1280 位置 "
            f"| **{_fmt_metric(m.get('tie_robust_cover_pct'), '.2f')}%** | 100% |",
            "| 音频 top-25 命中率（CUDA topk 口径，原门禁口径） "
            f"| {top25}% | 100% |",
            "| **语言分层 tie-robust `cover` 增益**（4 语言 × 4 段，相对 W1：g32+v_proj 保护） "
            f"| **+{_fmt_metric(m.get('tie_robust_cover_gain_lang_pt'), '.3f')} pt** | — |",
            "| **情感文本 tie-robust `cover` 增益**（6 项，同样相对 W1） "
            f"| **+{_fmt_metric(m.get('tie_robust_cover_gain_emo_pt'), '.3f')} pt** | — |",
            "| 音频 logit 平均绝对偏差 mean\\|Δ\\|（越低越好） "
            f"| **{_fmt_metric(m.get('audio_mean_abs_logit_delta'), '.4f')}** | 0.0006 |"])
        tie_note = DEFAULT_TIE_CONVENTION_SECTION
    else:
        label = str(m.get("top25_label", DEFAULT_TOP25_LABEL))
        tie_rows = f"| 音频 top-25 命中率（{label}） | **{top25}%** | 100% |"
        tie_note = ""
    tie_section = _as_section(tie_note)

    common = dict(top25=top25,
                  text_argmax=_fmt_metric(m.get("gate2_text_argmax_pct"), ".1f"),
                  speed=_fmt_metric(m.get("steps_per_s_steady"), ".1f"),
                  vram=_fmt_metric(m.get("vram_peak_gib"), ".2f"),
                  runaway=str(m.get("runaway", "n/a")),
                  size_gib=f"{meta['weight_file']['bytes'] / 2**30:.2f}")
    if has_tie:
        tradeoff = ("音频 top-25 命中率\n  "
                    f"{_fmt_metric(m.get('tie_robust_cover_pct'), '.2f')}%（tie-robust 口径）／"
                    f"{top25}%（CUDA topk 口径，见 §3.1），")
        en_metrics = ("\n**{}%** (tie-robust `cover`) / **{}%** (CUDA topk convention,\n"
                      "see §3.1 — the two are not comparable in magnitude), "
                      "text argmax {}%,\n**{} steps/s**, **{} GiB** resident VRAM, "
                      "runaway **{}**\n(12 seed long-form battery, watchdog off), "
                      "{} GiB on disk.").format(
                          _fmt_metric(m.get("tie_robust_cover_pct"), ".2f"), top25,
                          common["text_argmax"], common["speed"], common["vram"],
                          common["runaway"], common["size_gib"])
    else:
        tradeoff = str(m.get("quality_tradeoff_sentence",
                              DEFAULT_QUALITY_TRADEOFF_SENTENCE)).format(**common)
        en_metrics = str(m.get("en_summary_metrics",
                               DEFAULT_EN_SUMMARY_METRICS)).format(**common)

    cross_section = _as_section(m.get("cross_lang_section",
                                      DEFAULT_CROSS_LANG_SECTION),
                                "\n\n" if tie_note else "\n")
    repl = {
        "MODEL_NAME": f"MOSS-TTS-v1.5-W4GPTQ-{preset_name}",
        "PRESET": preset_name,
        "PRESET_LABEL": str(preset.get("label")),
        "EXPORT_DATE": meta["created_utc"][:10],
        "GROUP_SIZE": str(preset["group_size_default"]),
        "N_QUANTIZED": str(preset["n_quantized_linears"]),
        "N_BF16_KEEPS": keeps,
        "BF16_KEEP_DESC": keeps,
        "N_TENSORS": str(meta["n_tensors"]),
        "SIZE_GIB": f"{meta['weight_file']['bytes'] / 2**30:.2f}",
        "WEIGHT_SHA256": meta["weight_file"]["sha256"],
        "TOP25": f"{float(m.get('gate2_audio_top25_pct', float('nan'))):.2f}",
        "TEXT_ARGMAX": f"{float(m.get('gate2_text_argmax_pct', float('nan'))):.1f}",
        "SPEED": f"{float(m.get('steps_per_s_steady', float('nan'))):.1f}",
        "VRAM": f"{float(m.get('vram_peak_gib', float('nan'))):.2f}",
        "RUNAWAY": str(m.get("runaway", "n/a")),
        "BASE_REPO": meta.get("base_repo", DEFAULT_BASE_REPO),
        "BASE_DIR": base.get("local_dir", "models/MOSS-TTS-v1.5"),

        "STATE_FILE": str(meta.get("state", {}).get("source_file", "n/a")),
        "STATE_SHA256": str(
            meta.get("state", {}).get("sha256")
            or (m.get("build") or {}).get("state_sha256") or "not recorded"),
        "BASE_SHARDS": base_sha or "| (not hashed) | | |",
        "MS_COLLECTION": BASE_COLLECTION_MS,
        "CREATED": meta["created_utc"],
        "TORCH": meta["tool"]["torch"],
        "Q_DTYPE": q["q_dtype"],
        "TIE_ROWS": tie_rows,
        "TIE_CONVENTION_SECTION": tie_section,
        "QUALITY_TRADEOFF_SENTENCE": tradeoff,
        "EN_SUMMARY_METRICS": en_metrics,
        "TIE_COVER": _fmt_metric(m.get("tie_robust_cover_pct"), ".2f"),
        "MEAND": _fmt_metric(m.get("audio_mean_abs_logit_delta"), ".4f"),
        "LANG_GAIN": _fmt_metric(m.get("tie_robust_cover_gain_lang_pt"), ".3f"),
        "EMO_GAIN": _fmt_metric(m.get("tie_robust_cover_gain_emo_pt"), ".3f"),
        "BF16_KEEP_TEXT": str(m.get("bf16_keep_text",
                                    "`v_proj` 全 36 层保留 bf16")),
        "CROSS_LANG_SECTION": cross_section,
        "KEEP_RATIONALE": str(m.get("keep_rationale",
                                    "`v_proj` 全 36 层（见档位说明）。")),
    }
    out = template
    for key, val in repl.items():
        out = out.replace("{{" + key + "}}", str(val))
    leftover = re.findall(r"\{\{([A-Z_]+)\}\}", out)
    if leftover:
        raise ValueError(f"unrendered placeholders in the model card: "
                         f"{sorted(set(leftover))}")
    return out

def is_standalone_dir(path: str) -> bool:
    """True if `path` is a standalone export (meta."""
    p = os.path.join(path, META_FILE)
    if not os.path.isfile(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("format") == FORMAT_TAG

def standalone_presets(path: str) -> list[str]:
    """Preset names offered by a standalone directory (e."""
    with open(os.path.join(path, META_FILE), encoding="utf-8") as f:
        meta = json.load(f)
    names = list(meta.get("presets") or {})
    if not names and meta.get("preset_name"):
        names = [meta["preset_name"]]
    return names

class _PlaceholderWeights(dict):
    """Weights dict that fakes the projections replaced by int4 payloads."""

    _SENTINEL = object()

    def __init__(self, real: dict, placeholders: dict[str, tuple | None],
                 device, dtype: torch.dtype):
        """placeholders:"""
        super().__init__(real)
        self._ph = dict(placeholders)
        self._device = device
        self._dtype = dtype
        self._empty: torch.Tensor | None = None
        for key in self._ph:
            dict.__setitem__(self, key, self._SENTINEL)

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if val is self._SENTINEL:
            shape = self._ph[key]
            if shape is not None:
                val = torch.zeros(tuple(shape), dtype=self._dtype,
                                  device=self._device)
            else:
                if self._empty is None:
                    self._empty = torch.zeros(0, dtype=self._dtype,
                                              device=self._device)
                val = self._empty
            dict.__setitem__(self, key, val)
        return val

def read_standalone(model_dir: str, expected_preset: str | None = None,
                    ) -> tuple[dict, dict, dict, dict, list]:
    """Read a standalone directory *without* touching the base checkpoint."""
    meta_path = os.path.join(model_dir, META_FILE)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != FORMAT_TAG:
        raise ValueError(f"{meta_path} is not a {FORMAT_TAG} directory")
    preset_name = meta.get("preset_name")
    if expected_preset and expected_preset != preset_name:
        raise SystemExit(
            f"[export] {model_dir} provides preset {preset_name!r} only, but "
            f"{expected_preset!r} was requested. Available: "
            f"{standalone_presets(model_dir)}")
    wf = meta.get("weight_file")
    qfile = os.path.join(model_dir, wf["name"] if isinstance(wf, dict) else Q_FILE)
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"{qfile} not found in {model_dir}")
    raw = read_safetensors(qfile)
    weights: dict[str, torch.Tensor] = {}

    _RK = {"q": "packed", "qsz": "qsz"}
    state: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for key, t in raw.items():
        m = _Q_RE.match(key)
        if m:
            li, proj, kind = int(m.group(1)), m.group(2), m.group(3)
            state.setdefault(li, {}).setdefault(proj, {})[_RK[kind]] = t
        else:
            weights[key] = t
    for li, projs in state.items():
        for proj, rec in projs.items():
            if set(rec) != {"packed", "qsz"}:
                raise ValueError(f"incomplete record for layer {li} {proj}: "
                                 f"{sorted(rec)}")
    pmeta = (meta.get("presets") or {}).get(preset_name, {})
    gmap = {}
    for key, val in (pmeta.get("group_size_map") or {}).items():
        a, b = key.split(":")
        gmap[(int(a), b)] = int(val)
    keep: list = [(int(x.split(":")[0]), x.split(":")[1])
                  for x in (pmeta.get("bf16_linears") or [])]
    keep += [int(x) for x in (pmeta.get("bf16_layers") or [])]
    return meta, weights, state, gmap, keep

def load_standalone_model(model_dir: str, device="cuda",
                          dtype: torch.dtype = torch.bfloat16,
                          max_seq_len: int = 8192,
                          expected_preset: str | None = None):
    """Assemble `(MossTTSModel, GptqMossTTS)` straight from a standalone dir."""
    dev = torch.device(device)
    meta, weights, state, gmap, keep = read_standalone(model_dir, expected_preset)
    q = meta.get("quantization", {})
    if q.get("method") != "gptq":
        raise ValueError(f"{model_dir}: unsupported quantization "
                         f"{q.get('method')!r} (only 'gptq')")

    cfg = meta["model_config"]
    placeholders: dict[str, tuple | None] = {}
    for li, projs in state.items():
        for proj in projs:
            key = base_weight_key(li, proj)
            if (li, proj) in keep or key in weights:
                continue
            if li == 0 and proj in ("q", "k"):
                placeholders[key] = tuple(cfg["layer0_q_proj_shape"] if proj == "q"
                                          else cfg["layer0_k_proj_shape"])
            else:
                placeholders[key] = None
    model = MossTTSModel(_PlaceholderWeights(weights, placeholders, dev, dtype),
                         device=dev, dtype=dtype, max_seq_len=max_seq_len)
    fast = GptqMossTTS(model, state, bf16_keep=keep, group_size_map=gmap,
                       inner_k_tiles=int(q.get("inner_k_tiles", 8)),
                       w4_group_size=int(
                           (meta.get("presets") or {}).get(
                               meta.get("preset_name"), {}).get(
                               "group_size_default", 128)))
    del weights, state
    return model, fast

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m moss_tts_lite.export",
        description="Export a self-contained (standalone) GPTQ model directory.")
    ap.add_argument("--model-dir", required=True,
                    help="bf16 base checkpoint directory")
    ap.add_argument("--gptq-state", required=True,
                    help="offline GPTQ state file (needs a sibling .meta.json)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--preset", required=True, help="preset name, e.g. w1")
    ap.add_argument("--metrics-json", default=None,
                    help="JSON file with measured metrics to embed")
    ap.add_argument("--readme-template", default=None,
                    help="model-card template (default: moss_tts_lite/README_hf.md)")
    ap.add_argument("--no-hash-base", action="store_true",
                    help="skip sha256 of the 16.6 GiB base shards")
    ap.add_argument("--force", action="store_true", help="overwrite the output")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory + meta only, write nothing")
    args = ap.parse_args(argv)
    metrics = json.load(open(args.metrics_json)) if args.metrics_json else None
    meta = export_standalone(args.model_dir, args.gptq_state, args.out, args.preset,
                             metrics=metrics, hash_base=not args.no_hash_base,
                             template_path=args.readme_template, force=args.force,
                             dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({k: meta[k] for k in
                          ("preset_name", "n_tensors", "quantized_bytes",
                           "model_config")}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
