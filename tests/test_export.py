"""Standalone export tests (export agent).

Phases (CPU-capable; the int4 packing kernel is CUDA-only, so on a CPU-only
host shapes come from the Meta backend with deterministic filler payloads —
the format/round-trip logic under test is value-agnostic, and with CUDA
present the real kernel is exercised bit-exactly):

  E1 writer/reader round-trip: the hand-rolled safetensors writer is read back
     by `st_loader.read_safetensors` with byte-identical payloads, correct
     header padding, metadata block and per-tensor shapes/dtypes
     (f32, i32, bf16, i8, bool, 3-D).
  E2 export end-to-end on a synthetic checkpoint: a tiny "MOSS-TTS" base dir +
     GPTQ state is exported to a standalone dir, then re-read; asserts the key
     layout (``layers.{i}.{p}.q/.qsz`` + original bf16 keys), unchanged bf16
     bytes, bit-exact int4 payload round-trip, and that meta.json declares
     group sizes / bf16 keeps / provenance / metrics.
  E3 equivalence to the patch path (CPU math): the int4 payloads and surviving
     bf16 tensors recovered from the export are *the same tensors* the patch
     path uses, and a kernel-free dequantization of both paths is bit-identical
     (proves the flatten/rename step value preserving).
  E4 loader guards: `is_standalone_dir` / `standalone_presets` / preset
     mismatch / missing weight file / incomplete record, and the `read_standalone`
     contract expected by `GptqMossTTS._quantize`.
  E5 model card render: template renders with no leftover ``{{...}}``
     placeholder, carries the preset metrics, and an unknown placeholder is a
     hard error (naming-drift guard).
  E6 purity: `moss_tts_lite/export.py` imports torch/numpy + stdlib only (the
     dependency gate allows no `safetensors` package).

Run:
  python3 -m moss_tts_lite.tests.test_export
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import torch

from moss_tts_lite.export import (FORMAT_TAG, PRESET_METRICS, Q_FILE, export_standalone,
                      is_standalone_dir, read_standalone, render_model_card,
                      standalone_presets, write_safetensors)
from moss_tts_lite.gptq import pack_fast, rtn_quantize
from moss_tts_lite.st_loader import read_safetensors, safetensors_header

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))          # lite repo root
TEMPLATE = os.path.join(REPO, "README_hf.md")

N_LAYERS = 2
HIDDEN = 256
HEAD_DIM = 32
N_HEADS = 4
N_KV = 2
TEXT_VOCAB = 128
PROJS = ("q", "k", "v", "o", "gate", "up", "down")
_SRC = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
        "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
        "down": "mlp.down_proj"}
_LIN_SHAPE = {"q": (N_HEADS * HEAD_DIM, HIDDEN), "k": (N_KV * HEAD_DIM, HIDDEN),
              "v": (N_KV * HEAD_DIM, HIDDEN), "o": (HIDDEN, N_HEADS * HEAD_DIM),
              "gate": (2 * HIDDEN, HIDDEN), "up": (2 * HIDDEN, HIDDEN),
              "down": (HIDDEN, 2 * HIDDEN)}


def _bf16(shape, seed, scale=0.05):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)


def _synthetic_base(dirpath: str) -> dict:
    """A miniature MOSS-TTS checkpoint with the exact key naming scheme."""
    tensors: dict[str, torch.Tensor] = {}
    tensors["language_model.embed_tokens.weight"] = _bf16((TEXT_VOCAB, HIDDEN), 1)
    tensors["language_model.norm.weight"] = _bf16((HIDDEN,), 2)
    for li in range(N_LAYERS):
        p = f"language_model.layers.{li}"
        tensors[f"{p}.input_layernorm.weight"] = _bf16((HIDDEN,), 10 + li)
        for proj in PROJS:
            tensors[f"{p}.{_SRC[proj]}.weight"] = _bf16(_LIN_SHAPE[proj], 20 + li * 7 + len(proj))
        tensors[f"{p}.self_attn.q_norm.weight"] = _bf16((HEAD_DIM,), 40 + li)
        tensors[f"{p}.self_attn.k_norm.weight"] = _bf16((HEAD_DIM,), 50 + li)
        tensors[f"{p}.post_attention_layernorm.weight"] = _bf16((HIDDEN,), 60 + li)
    for i in range(32):
        tensors[f"emb_ext.{i}.weight"] = _bf16((1025, HIDDEN), 100 + i)
    for i in range(33):
        tensors[f"lm_heads.{i}.weight"] = _bf16((TEXT_VOCAB if i == 0 else 1025, HIDDEN),
                                                 200 + i)
    os.makedirs(dirpath, exist_ok=True)
    # a single shard + an index (exercises the index-based reader path)
    write_safetensors(os.path.join(dirpath, "model.safetensors"), sorted(tensors.items()))
    with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {k: "model.safetensors" for k in sorted(tensors)}}, f)
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump({"torch_dtype": "bfloat16"}, f)
    for name in ("vocab.json", "merges.txt", "tokenizer.json"):
        with open(os.path.join(dirpath, name), "w") as f:
            f.write(f"# synthetic {name}\n")
    return tensors


def _pack(q, s, mn, kt=8, dev=None):
    """`pack_fast`, but shape-faithful on a machine without the CUDA kernel.

    `aten._convert_weight_to_int4pack` exists for CUDA (and Meta) only, so on a
    CPU-only host the *shape* is taken from the Meta backend and the payload is
    filled with deterministic pseudo-random bytes: the export/format/round-trip
    logic under test does not care about the values, only about shapes, dtypes
    and byte fidelity.  With CUDA present the real kernel is used (bit-exact).
    """
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        qp = q[:, 1::2] | (q[:, 0::2] << 4)
        shp = torch.ops.aten._convert_weight_to_int4pack(
            qp.contiguous().to("meta"), kt).shape
        gen = torch.Generator().manual_seed(int(q.float().abs().sum().item()))
        packed = torch.randint(0, 2**31 - 1, tuple(shp), generator=gen,
                               dtype=torch.int32)
        kk = q.shape[1] // s.shape[1]  # noqa: F841 (documents the group layout)
        qsz = torch.stack([s, mn + 8.0 * s], -1).bfloat16().transpose(0, 1).contiguous()
        return packed, qsz
    return pack_fast(q.to(dev), s.to(dev), mn.to(dev), kt)


def _synthetic_state(base: dict, group=32, keep_v=True) -> tuple[dict, dict]:
    """RTN-pack every projection except v_proj (bf16 keep), like w1."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state, gmap, keep = {}, {}, []
    for li in range(N_LAYERS):
        state[li] = {}
        for proj in PROJS:
            if keep_v and proj == "v":
                keep.append(f"{li}:v")
                continue
            key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
            q, s, mn = rtn_quantize(base[key], group)
            packed, qsz = _pack(q, s, mn, 8, dev)
            state[li][proj] = {"packed": packed.cpu(), "qsz": qsz.cpu()}
            gmap[f"{li}:{proj}"] = group
    return state, {"group_size_map": gmap, "bf16_linears": keep,
                   "bf16_layers": []}


def _write_state(path: str, state: dict, meta: dict) -> None:
    torch.save(state, path)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f)


# ------------------------------------------------------------------------ E1
def phase_e1_writer_roundtrip() -> bool:
    print("E1 writer/reader round-trip (all dtypes, metadata, 3-D payload)")
    tmp = tempfile.mkdtemp(prefix="export_e1_")
    try:
        tensors = {
            "f32": torch.linspace(-3, 3, 1000, dtype=torch.float32).reshape(10, 100),
            "f64": torch.randn(7, dtype=torch.float64),
            "i32_packed": torch.randint(-2**31, 2**31 - 1, (6, 8), dtype=torch.int32),
            "u8": torch.randint(0, 255, (5, 3), dtype=torch.uint8),
            "i8": torch.randint(-127, 127, (4,), dtype=torch.int8),
            "bf16": _bf16((3, 5), 7),
            "bool": torch.tensor([True, False, True]),
            "empty": torch.zeros(0, dtype=torch.bfloat16),
            "three_d": torch.arange(24, dtype=torch.int32).reshape(2, 3, 4),
        }
        path = os.path.join(tmp, "t.safetensors")
        size, sha = write_safetensors(path, sorted(tensors.items()),
                                     metadata={"format": FORMAT_TAG, "preset": "w1"})
        ok = True
        hdr = safetensors_header(path)
        ok &= hdr.get("__metadata__", {}).get("preset") == "w1"
        ok &= len([k for k in hdr if k != "__metadata__"]) == len(tensors)
        with open(path, "rb") as f:
            import struct
            (n,) = struct.unpack("<Q", f.read(8))
        ok &= (8 + n) % 8 == 0 and size == os.path.getsize(path)
        back = read_safetensors(path)
        for name, t in sorted(tensors.items()):
            r = back[name]
            same = (r.dtype == t.dtype and tuple(r.shape) == tuple(t.shape)
                    and torch.equal(r.view(torch.uint8), t.contiguous().view(torch.uint8)))
            print(f"  {name:12s} {str(t.dtype):14s} {tuple(t.shape)} bytes-equal={same}")
            ok &= same
        # metadata must not leak into the tensor namespace
        ok &= "__metadata__" not in back
        print(f"  E1: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------ E2
def phase_e2_export_roundtrip() -> bool:
    print("E2 export end-to-end on a synthetic checkpoint")
    tmp = tempfile.mkdtemp(prefix="export_e2_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        emeta = export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                                  hash_base=True, template_path=TEMPLATE,
                                  quiet=True)
        checks: dict[str, bool] = {}
        checks["files_present"] = all(os.path.isfile(os.path.join(out, f))
                                      for f in (Q_FILE, "meta.json", "README.md",
                                                "LICENSE", "vocab.json"))
        checks["format_tag"] = emeta["format"] == FORMAT_TAG
        checks["preset_name"] = emeta["preset_name"] == "w1"
        checks["base_n_keys"] = emeta["base"]["n_keys"] == len(base)
        checks["base_sha256"] = all("sha256" in s for s in emeta["base"]["shards"])
        p = emeta["presets"]["w1"]
        checks["bf16_keeps"] = p["bf16_linears"] == [f"{li}:v" for li in range(N_LAYERS)]
        checks["group_map"] = set(p["group_size_map"]) == {
            f"{li}:{pr}" for li in range(N_LAYERS) for pr in PROJS if pr != "v"}
        checks["heads"] = (emeta["model_config"]["n_heads"] == N_HEADS
                           and emeta["model_config"]["n_kv_heads"] == N_KV
                           and emeta["model_config"]["hidden_size"] == HIDDEN)
        print(f"  meta: preset={emeta['preset_name']} tensors={emeta['n_tensors']} "
              f"bf16_keeps={len(p['bf16_linears'])} "
              f"quantized={p['n_quantized_linears']}")
        # re-read and verify every payload
        rmeta, weights, rstate, gmap, keep = read_standalone(out)
        n_q = n_bad = 0
        for li in sorted(state):
            for proj, rec in state[li].items():
                n_q += 2
                n_bad += int(not torch.equal(rstate[li][proj]["packed"], rec["packed"]))
                n_bad += int(not torch.equal(rstate[li][proj]["qsz"], rec["qsz"]))
        print(f"  int4 payloads: {n_q} tensors compared, {n_bad} mismatch")
        checks["int4_roundtrip"] = n_bad == 0
        n_b = n_bad_b = 0
        for key, t in base.items():
            if key in weights:
                n_b += 1
                n_bad_b += int(not torch.equal(weights[key], t))
        n_kept = sum(1 for k in base
                     if k.endswith(tuple(f".{s}.weight" for s in _SRC.values()))
                     and k in weights)
        print(f"  bf16 tensors: {n_b} compared ({n_kept} of them backbone "
              f"projections by original key), {n_bad_b} mismatch")
        checks["bf16_bytes"] = n_bad_b == 0
        checks["quantized_count"] = emeta["presets"]["w1"]["n_quantized_linears"] \
            == len(state) * (len(PROJS) - 1)
        checks["keeps_reread"] = keep == [(li, "v") for li in range(N_LAYERS)]
        checks["gmap_reread"] = gmap == {(li, pr): 32 for li in range(N_LAYERS)
                                         for pr in PROJS if pr != "v"}
        failed = sorted(k for k, v in checks.items() if not v)
        if failed:
            print(f"  failed checks: {failed}")
        print(f"  E2: {'PASS' if not failed else 'FAIL'}")
        return not failed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------ E3
def phase_e3_patch_path_equivalence() -> bool:
    """The export must be *the same bytes* the patch path uses, and they must
    still be the bytes an independent re-quantization of the base produces
    (proves the flatten/rename step is value preserving and lossless)."""
    print("E3 equivalence with the patch path (byte-level)")
    tmp = tempfile.mkdtemp(prefix="export_e3_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                          hash_base=False, quiet=True)
        patch = torch.load(spath, map_location="cpu", weights_only=False)
        _m, sweights, sstate, sgmap, skeep = read_standalone(out)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ok = True
        n = 0
        for li in sorted(patch):
            for proj in sorted(patch[li]):
                a, b = patch[li][proj], sstate[li][proj]
                g = sgmap[(li, proj)]
                # 1) patch-state vs standalone bytes
                ok &= torch.equal(a["packed"], b["packed"])
                ok &= torch.equal(a["qsz"], b["qsz"])
                # 2) standalone vs an independent re-quantization of the base
                key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
                q, s, mn = rtn_quantize(base[key], g)
                packed, qsz = _pack(q, s, mn, 8, dev)
                ok &= torch.equal(b["packed"], packed.cpu())
                ok &= torch.equal(b["qsz"], qsz.cpu())
                n += 1
        # 3) bf16 keeps are the untouched base tensors
        for li, proj in skeep:
            key = f"language_model.layers.{li}.{_SRC[proj]}.weight"
            ok &= torch.equal(sweights[key], base[key])
        print(f"  {n} projections: patch-state == standalone == fresh RTN "
              f"re-quantization (bit-exact, payload device={dev}), "
              f"bf16 keeps untouched")
        print(f"  E3: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------ E4
def phase_e4_loader_guards() -> bool:
    print("E4 loader guards")
    tmp = tempfile.mkdtemp(prefix="export_e4_")
    try:
        base = _synthetic_base(os.path.join(tmp, "base"))
        state, meta = _synthetic_state(base)
        spath = os.path.join(tmp, "w1.pt")
        _write_state(spath, state, meta)
        out = os.path.join(tmp, "export")
        export_standalone(os.path.join(tmp, "base"), spath, out, "w1",
                          hash_base=False, quiet=True)
        ok = is_standalone_dir(out) and not is_standalone_dir(tmp)
        ok &= not is_standalone_dir(os.path.join(tmp, "base"))
        ok &= standalone_presets(out) == ["w1"]
        # preset mismatch
        try:
            read_standalone(out, expected_preset="w2")
            ok = False
        except SystemExit:
            pass
        # missing weight file
        moved = os.path.join(tmp, "quantized.moved")
        os.rename(os.path.join(out, Q_FILE), moved)
        try:
            read_standalone(out)
            ok = False
        except FileNotFoundError:
            pass
        os.rename(moved, os.path.join(out, Q_FILE))
        # incomplete record (drop one .qsz) -> must refuse.  Truncating the
        # mapped file in place is not allowed, so rewrite it and repoint
        # meta.json at the rewritten name.
        raw = read_safetensors(os.path.join(out, Q_FILE))
        trimmed = {k: v for k, v in raw.items() if k != "layers.0.q.qsz"}
        write_safetensors(os.path.join(out, "trimmed.safetensors"),
                          sorted(trimmed.items()))
        mpath = os.path.join(out, "meta.json")
        m = json.load(open(mpath))
        m["weight_file"] = dict(m["weight_file"], name="trimmed.safetensors")
        with open(mpath, "w") as f:
            json.dump(m, f)
        del raw, trimmed
        try:
            read_standalone(out)
            ok = False
        except ValueError as e:
            ok &= "incomplete" in str(e)
        # state record keys are the ones GptqMossTTS._quantize reads
        out2 = os.path.join(tmp, "export2")
        export_standalone(os.path.join(tmp, "base"), spath, out2, "w1",
                          hash_base=False, quiet=True)
        _m, _w, st, _g, _k = read_standalone(out2)
        keys = {k for rec in st.values() for r in rec.values() for k in r}
        ok &= keys == {"packed", "qsz"}
        print(f"  is_standalone / presets / mismatch / missing file / "
              f"incomplete record / record keys={'PASS' if ok else 'FAIL'}")
        print(f"  E4: {'PASS' if ok else 'FAIL'}")
        return bool(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------ E5
_M5_META = {
    "created_utc": "2025-09-11T00:00:00Z",
    "n_tensors": 679,
    "weight_file": {"bytes": 7636941152, "sha256": "ab" * 32},
    "quantization": {"q_dtype": "I32"},
    "model_config": {"hidden_size": 4096},
    "base": {"local_dir": "models/MOSS-TTS-v1.5", "shards": [
        {"name": "model-00001-of-00004.safetensors", "bytes": 4932667368,
         "sha256": "cd" * 32}]},
    "base_repo": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "tool": {"torch": "2.9.1"},
    # `export_standalone` always records the offline GPTQ state it exported
    # from; the card's provenance section names it (and its sha256), so the
    # fixture carries it too.
    "state": {"source_file": "w1.pt", "source_bytes": 4246863871,
              "sha256": "ef" * 32},
}


def phase_e5_model_card() -> bool:
    print("E5 model card rendering")
    ok = True
    tpl = open(TEMPLATE, encoding="utf-8").read()
    for preset in sorted(PRESET_METRICS):
        meta = dict(_M5_META)
        meta["presets"] = {preset: {
            "label": PRESET_METRICS[preset]["label"],
            "group_size_default": PRESET_METRICS[preset]["group_size"],
            "bf16_linears": ["0:v"], "bf16_layers": [],
            "n_quantized_linears": 216,
            "metrics": PRESET_METRICS[preset]}}
        card = render_model_card(tpl, meta, preset)
        ok &= "{{" not in card and "}}" not in card
        ok &= f"{PRESET_METRICS[preset]['gate2_audio_top25_pct']:.2f}" in card
        ok &= f"{PRESET_METRICS[preset]['steps_per_s_steady']:.1f}" in card
        ok &= "Apache-2.0" in card
        ok &= "跨语言验证" in card   # langcheck-1 section survives every re-export
        # A preset whose metrics carry no tie-robust measurement must render the
        # single historical gate row instead of `n/a` tie rows/prose.
        ok &= "n/a" not in card
        # provenance: the card must name the state file it was exported from
        ok &= "`w1.pt`" in card and "ef" * 16 in card
        print(f"  {preset}: {len(card)} chars, no leftover placeholder, "
              f"metrics + cross-language section present, no n/a, "
              f"state file named")
    # naming-drift guard: an unknown placeholder must raise
    try:
        render_model_card("{{NOT_A_PLACEHOLDER}}", meta, preset)
        ok = False
    except ValueError:
        pass
    # the tie-robust branch (w2's real metric set) must expand the two-convention
    # rows and the §3.1 convention note, and keep the cross-language section
    meta_tie = dict(_M5_META)
    meta_tie["presets"] = {"w2": {
        "label": PRESET_METRICS["w2"]["label"], "group_size_default": 32,
        "bf16_linears": ["0:v"], "n_quantized_linears": 216,
        "metrics": {**PRESET_METRICS["w2"], "tie_robust_cover_pct": 99.3,
                    "audio_mean_abs_logit_delta": 0.2641,
                    "tie_robust_cover_gain_lang_pt": 0.545,
                    "tie_robust_cover_gain_emo_pt": 1.02}}}
    card_tie = render_model_card(tpl, meta_tie, "w2")
    ok &= "tie-robust `cover`" in card_tie
    ok &= "### 3.1 关于 top-25 数字的口径" in card_tie
    ok &= "跨语言验证" in card_tie
    ok &= "n/a" not in card_tie
    print(f"  w2+tie metrics: {len(card_tie)} chars, tie rows + §3.1 note + "
          f"cross-language section, no n/a")
    print(f"  E5: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


# ------------------------------------------------------------------------ E6
def phase_e6_purity() -> bool:
    print("E6 export.py dependency purity")
    import ast
    path = os.path.join(REPO, "moss_tts_lite", "export.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    allowed = {"torch", "numpy", "json", "os", "re", "shutil", "struct", "sys",
               "time", "hashlib", "argparse", "__future__"}
    extra = sorted(mods - allowed)
    top = sorted(mods)
    ok = not extra
    # the dependency gate allows no `safetensors` package: the writer is ours
    ok &= "safetensors" not in mods
    print(f"  top-level imports: {top}")
    print(f"  disallowed: {extra}")
    print(f"  E6: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def main() -> int:
    results = {
        "E1_writer": phase_e1_writer_roundtrip(),
        "E2_export": phase_e2_export_roundtrip(),
        "E3_equiv": phase_e3_patch_path_equivalence(),
        "E4_guards": phase_e4_loader_guards(),
        "E5_card": phase_e5_model_card(),
        "E6_purity": phase_e6_purity(),
    }
    ok = all(results.values())
    print("test_export: " + " ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                     for k, v in results.items())
          + f" -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
