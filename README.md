# moss-tts-lite

MOSS-TTS-v1.5 的极简推理库：**4 个依赖**、**单卡可跑**（8GB 起）、输出与官方实现 **bitwise 一致**。

```bash
pip install moss-tts-lite
```

一句话合成语音：

```bash
moss-tts-lite "你好，欢迎使用。" -o out.wav --model-dir /path/to/W4GPTQ-w1
```

## 它是什么

| | |
|---|---|
| 模型 | MOSS-TTS-v1.5（OpenMOSS），31 语言，24kHz，支持 `[pause 2s]` 韵律标记 |
| 实现 | 纯 PyTorch（torch/numpy/soundfile/pyyaml，无 triton/compile/flash-attn），Windows 可用 |
| 正确性 | 与 HF 原版同 seed **逐位一致**（bitwise EXACT 档）；量化档经十语言轨迹验证 |
| 速度 | W4 GPTQ 82-97 步/s（A10G，≈6.5× 实时）；bf16 36 步/s |
| 显存 | **8GB 卡可跑质量档**（W4GPTQ，按需 KV）；24GB 卡可跑 bf16 |

## 安装

```bash
# 1) 先装 CUDA 版 torch（按你的 CUDA 版本选索引；>= 2.5）
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) 装本库
pip install moss-tts-lite
```

> torch 2.4 及以下仅支持 bf16 eager 路径（量化档需 2.5+，实测跨版本轨迹一致）；
> 图化 fast 路径需 CUDA，纯 CPU 下（`--device cpu`）CLI 会自动回退到 eager。

## 模型权重（三选一）

| 选项 | 命令 | 说明 |
|---|---|---|
| **量化档（推荐）** | HF 或 ModelScope 下载 `MOSS-TTS-v1.5-W4GPTQ-w1` | 7.2GB，自包含目录，8GB 显存可用 |
| 质量优先 | 同上，`-w2` | 略大，离线生产场景 |
| 原版权重 | `OpenMOSS-Team/MOSS-TTS-v1.5` + `MOSS-Audio-Tokenizer` | bf16 全精度，需 24GB |

```bash
# Hugging Face
huggingface-cli download baicai1145/MOSS-TTS-v1.5-W4GPTQ-w1 --local-dir ./W4GPTQ-w1
# 或 ModelScope
modelscope download --model baicai1145/MOSS-TTS-v1.5-W4GPTQ-w1 --local_dir ./W4GPTQ-w1
```

## 用法

```bash
# 量化档（默认 w1p：82 步/s，质量十语言验证 99.5%）
moss-tts-lite "要合成的文本" -o out.wav --model-dir ./W4GPTQ-w1

# 英文（带语言标签更稳）
moss-tts-lite "Hello world." -o en.wav --model-dir ./W4GPTQ-w1 --language English

# 停顿控制
moss-tts-lite "先别急——[pause 1s]我们再想想。" -o pause.wav --model-dir ./W4GPTQ-w1

# 极速档（97 步/s，非 bitwise，统计等价）
moss-tts-lite "文本" -o out.wav --model-dir ./W4GPTQ-w1 --fast-native

# 固定种子复现
moss-tts-lite "文本" -o out.wav --model-dir ./W4GPTQ-w1 --seed 1234
```

> **默认就是快路径**（v1.1.0 起）：裸命令行直接走 CUDA Graph 解码（`moss_tts_lite.fast`），
> 与 eager 参照**逐位一致**。旧脚本里的 `--fast` 仍被接受，但已是 no-op（打印一行提示）。
> 要跑慢的 eager 参照路径请显式加 `--eager`（仅调试用）。

Python API：

```python
from moss_tts_lite.cli import synthesize

wav, sr, res, strategy = synthesize(
    "你好。", output="out.wav",
    model_dir="./W4GPTQ-w1",   # standalone 目录
    quant="w4gptq",             # 或 None=bf16（需原版权重）
    seed=1234,
    # fast=True 是默认（与 CLI 一致）；只有显式 fast=False 才是 eager 参照路径
)
```

## 档位速查

| 命令 | 速度* | 显存 | 质量 | bitwise |
|---|---|---|---|---|
| **（默认，等价于 `--fast`）** | **36 步/s** | ~17GB | 参照 | ✅ |
| `--quant w4gptq` | 82 步/s | 8.5GB / **8GB 卡可用** | top-25 98.6%（tie-robust） | ❌ 量化档 |
| `--quant w4gptq` + `--fast-native` | 97 步/s | 8.2GB | argmax 100% / top-25 97.5% | ❌ |
| `--eager`（仅调试） | 30 步/s | ~17GB | 与默认**逐位相同** | ✅ |

*A10G 实测；4090 约按带宽等比提升。

8GB 卡注意：加 `--max-new-tokens 2048`（约 164 秒音频容量），详见 `--help`。

三路径的关系：默认 `fast`（逐位一致）、`--fast-native`（更快，非逐位）、`--eager`（慢参照）。
**eager 路径本体仍保留在 API 层**——直接调用 `moss_tts_lite.generate` / `MossTTSModel.step`
的代码不受本次默认翻转影响，只是 CLI 不再默认走它。

## 常见问题

**生成的音频不结束/一直静音？** 内置看门狗 v2 会自动截断（默认开启，`--no-watchdog` 关闭）。这是 RTN 量化档的已知风险，GPTQ 档 0/32 复现。

**CUDA 装不上？** torch 必须从 pytorch 官方索引装（见安装节），PyPI 上的 `torch` 默认包可能不带你的 CUDA 版本。

**Windows？** 支持但未在真机验证（Linux 验证 + API 全兼容）。遇到问题提 issue。

## 更多

- 完整技术文档（量化方法、验证方法论、性能分析）：[GitHub](https://github.com/baicai-1145/moss-tts-lite/blob/main/README_ADVANCED.md)
- 模型卡与量化验证报告：随模型仓库发布
- License：Apache-2.0（继承 MOSS-TTS 上游）
