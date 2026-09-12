---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-TTS-v1.5
library_name: moss_tts_lite
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- quantized
- int4
- gptq
- w4
- moss-tts
language:
- zh
- en
- yue
- ja
- ko
- de
- fr
- es
- ru
- ar
---

# {{MODEL_NAME}} — 独立（自包含）W4-GPTQ 量化版

**MOSS-TTS-v1.5** 的 int4 GPTQ 量化权重，**自包含单目录**：下载本目录即可推理，**不再需要 16.6 GB 的
bf16 基座检查点**。

*Standalone int4 (GPTQ) weights for MOSS-TTS-v1.5. This directory is
self-contained — the 16.6 GB bf16 base checkpoint is **not** required.
Requires the minimal inference runtime `moss_tts_lite` (this repository) and a
separate audio-codec checkpoint.*

| 项 / Item | 值 / Value |
|---|---|
| 档位 / preset | `{{PRESET}}` ({{PRESET_LABEL}}) |
| 量化 / quantization | GPTQ, int4, group size **{{GROUP_SIZE}}**, {{Q_DTYPE}} payload + BF16 scales/zeros |
| 量化投影数 / quantized projections | {{N_QUANTIZED}} |
| 保 bf16 / kept in bf16 | {{BF16_KEEP_DESC}} |
| 权重文件 / weight file | `quantized.safetensors` — {{SIZE_GIB}} GiB, {{N_TENSORS}} tensors |
| sha256 | `{{WEIGHT_SHA256}}` |
| 导出时间 / exported | {{CREATED}} (torch {{TORCH}}) |
| 基座 / base | [{{BASE_REPO}}](https://huggingface.co/{{BASE_REPO}}) |

---

## 1. 这是什么

这是 MOSS-TTS-v1.5（Qwen3-8B 骨干 + 32 路音频码本 + 33 个输出头）的**逐层 GPTQ 校准量化**版本：
骨干 36 层的 `q/k/o/gate/up/down` 六个投影被压成 int4（group {{GROUP_SIZE}}），**最敏感的
{{BF16_KEEP_TEXT}}**；嵌入、输出头、各类归一化全部保持 bf16。

推理内核与未量化的 `--quant w4` 档**完全相同**（`torch._weight_int4pack_mm`，
`inner_k_tiles=8`，offset-binary nibble 打包），因此量化只改变权重数值、不引入任何运行时开销。

## 2. 量化方法

* **校准语料**：6 层分层设计（纯文本 / 语流 / 边界 / 长静音 / 收尾 / 深段），约 6200 个位置，
  由 bf16 模型自回归生成，覆盖 RTN 失效的病灶区。
* **GPTQ 核心**：经典逐列（block 128）惰性批更新，`H = XᵀX` 带 0.1% 阻尼的 Cholesky 逆
  （Hessian 数值秩亏，零阻尼不可解）；误差按列补偿。
* **逐层顺序量化**：第 L 层的 Hessian 由"已量化"的前 L−1 层产生的激活统计，
  使校准分布与部署分布一致（error-propagated）。
* **保护项选择**：由三路独立证据（校准相对误差排名、held-out 相对误差、llama.cpp Q4_K_M 配方）
  取交集 → `v_proj` 全 36 层（本档 {{PRESET}}）。
* **保护项的第二批**：{{KEEP_RATIONALE}}
* **细分组**：g32 是内核支持的最细粒度（g16/g8 内核报错）。

## 3. 指标（A10G-24G, torch {{TORCH}}；数据源 {{PRESET}} 档门禁实测）

| 指标 / Metric | 本档 / This preset | bf16 参考 / Reference |
|---|---|---|
{{TIE_ROWS}}
| 文本 argmax 一致率 | {{TEXT_ARGMAX}}% | 100% |
| 解码速度（稳态） | **{{SPEED}} 步/s** | ~36 步/s |
| 显存驻留峰值 | **{{VRAM}} GiB** | ~17.0 GiB |
| runaway（12 种子长文本，看门狗关闭） | **{{RUNAWAY}}** | 0/12（配看门狗） |
| 权重体积 | {{SIZE_GIB}} GiB | 16.6 GiB |

GPTQ 档在同 12 个种子上修掉了 RTN int4 的近静音 runaway（RTN：3/12），
长静音 / 收尾段的相对误差约为 RTN g128 的 1/50。
{{TIE_CONVENTION_SECTION}}{{CROSS_LANG_SECTION}}

## 4. 使用方法

```bash
# 1) 运行时依赖（仅 4 个第三方包）
pip install torch --index-url https://download.pytorch.org/whl/cu128   # torch>=2.9, CUDA 12.x
pip install numpy soundfile pyyaml

# 2) 本目录（TTS 主模型）
hf download <用户名>/{{MODEL_NAME}} --local-dir ./{{MODEL_NAME}}
# 或 ModelScope 集合 {{MS_COLLECTION}}；音频 codec 是**独立**检查点，需另行下载：
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir ./MOSS-Audio-Tokenizer

# 3) 推理（--model-dir 指向本目录；目录内含 meta.json 时会自动识别量化档）
python -m moss_tts_lite "你好，欢迎收听这段试音。" -o out.wav \
    --model-dir ./{{MODEL_NAME}} \
    --codec-dir ./MOSS-Audio-Tokenizer
# 等价显式写法：
python -m moss_tts_lite "Hello, this is a test." -o out_en.wav --language English \
    --model-dir ./{{MODEL_NAME}} --quant w4gptq
```

Python API：

```python
from moss_tts_lite.export import load_standalone_model       # 装配（自动读 meta.json）
from moss_tts_lite.fast import generate_fast
from moss_tts_lite.prompt import build_tts_prompt

model, fast = load_standalone_model("./{{MODEL_NAME}}")   # (MossTTSModel, GptqMossTTS)
prompt = build_tts_prompt("你好，欢迎收听。")
res = generate_fast(fast, prompt, seed=1234)              # 之后交给 moss_tts_lite.codec 解码
```

## 5. 目录内容 / Files

| 文件 / File | 说明 / Description |
|---|---|
| `quantized.safetensors` | int4 打包权重（`layers.{i}.{proj}.q` / `.qsz`）+ 全部保 bf16 张量（沿用原检查点 key 名） |
| `meta.json` | 格式版本、量化参数、逐项 group size、bf16 保留清单、基座溯源（含 sha256）、指标 |
| `README.md` | 本模型卡 |
| `config.json`, `configuration*.py`, `modeling_moss_tts.py`, `processing_moss_tts.py` | 原模型配置与参考实现（provenance / 互操作） |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`, `chat_template.jinja` | 分词器全套（`moss_tts_lite` 只需要 `vocab.json`/`merges.txt`/`tokenizer.json`） |
| `LICENSE` | Apache-2.0（继承自基座） |

## 6. 限制声明 / Limitations

* **量化质量权衡**：int4 不是 bf16 的位级等价物。{{QUALITY_TRADEOFF_SENTENCE}}
  主观质量接近但不能等同于 bf16；对音质极端敏感的场景请使用未量化权重。
* **速度余量很窄**：稳态 {{SPEED}} 步/s，距离本项目 78 步/s 的硬底线只有约 0.6%。
  再加任何 bf16 保护项（每 4 个投影约 −3.8%）会跌破底线，需先重新验收速度。
* **仅 CUDA**：int4 kernel（`_weight_int4pack_mm`）与 CUDA Graph 路径要求 NVIDIA GPU
  （Ampere/sm_80 及以上）；没有 CPU 回退。
* **codec 不在本目录**：音频解码器 MOSS-Audio-Tokenizer(1.6B) 是独立检查点（~3.6 GB），
  需另行下载；仅下载本目录无法出声。
* **看门狗默认开启**：连续低能量超过 ~5.12 s（或超长段落尾部已静音）会强制收尾，
  以避免失控生成；需要长静音时可调大 `--watchdog-silence-frames` 或 `--no-watchdog`。
* **Windows 未实测**：仅在 Linux + A10G-24G（torch 2.9.1, CUDA 12.x）上验证；
  Windows/WSL 与其它 GPU 架构未做验收测试。
* **语言覆盖**：基座声明支持 31 种语言，本量化版已在其中 **10 种**上完成跨语言验收（见上文“跨语言验证”小节），其余 21 种未逐一验证。
* **许可证继承**：本量化产物继承基座 Apache-2.0，署名 OpenMOSS-Team；商用请遵循原模型许可与使用条款。

## 7. 基座溯源 / Provenance

| 基座分片 / Base shard | 大小 | sha256 |
|---|---|---|
{{BASE_SHARDS}}

* 基座仓库 / base repo：[{{BASE_REPO}}](https://huggingface.co/{{BASE_REPO}})
* 本地导出源 / exported from：`{{BASE_DIR}}`
* 权重文件 sha256：`{{WEIGHT_SHA256}}`
* 导出时间 / exported at：{{CREATED}}（A10G-24G, torch {{TORCH}}）
* 量化状态文件 / GPTQ state：`{{STATE_FILE}}`（sha256 `{{STATE_SHA256}}`）
* 复现脚本 / regeneration：`python -m moss_tts_lite.export --model-dir {{BASE_DIR}} --gptq-state {{STATE_FILE}} --out <dir> --preset {{PRESET}}`

## 8. English summary

This is a **self-contained int4 GPTQ quantization** of
[MOSS-TTS-v1.5](https://huggingface.co/{{BASE_REPO}}) (Qwen3-8B backbone + 32 audio
codebooks + 33 heads, 31 languages). The six backbone projections per layer
(`q/k/o/gate/up/down`) are int4 (group {{GROUP_SIZE}}, GPTQ with six-stratum
calibration and error-propagated sequential quantization) while the most
sensitive projections ({{BF16_KEEP_DESC}}) stay bf16; embeddings, heads and norms
are bf16. Runtime kernel is identical to the non-GPTQ `w4` tier, so the
quantization costs nothing at inference.

Measured on an A10G-24G with torch {{TORCH}}: audio top-25 membership{{EN_SUMMARY_METRICS}}

Install `torch`/`numpy`/`soundfile`/`pyyaml`, download the separate
MOSS-Audio-Tokenizer codec, then:

```bash
python -m moss_tts_lite "Hello, this is a test." -o out.wav \
    --language English --model-dir <this directory>
```

Limitations: int4 quality trade-off (not bit-equivalent to bf16), CUDA-only
int4 kernel, codec checkpoint not included, run-away watchdog on by default
(~5.12 s of consecutive low-energy frames forces an audio end), validated on
Linux + A10G only (Windows untested), license inherited from the Apache-2.0
base model (attribution: OpenMOSS-Team).
