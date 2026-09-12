# moss-tts-lite — MOSS-TTS-v1.5 最小推理栈（含 CUDA Graph 快速路径与 W4 量化）

`moss_tts_lite` 包是 MOSS-TTS-v1.5（Qwen3-8B 骨干 + MOSS-Audio-Tokenizer 编解码）的极简推理实现：
预填充 + 逐帧音频解码，输出 24 kHz wav。本包内置三条加速路径：

- **fast bf16**：38 个分段 CUDA 子图 + 静态缓冲（`--fast`），输出与原始 eager 路径**逐位一致**（EXACT）；
- **fast-native**（`--fast-native`）：整步单图 + 融合算子（`moss_tts_lite.fast_native`，
  臂 `n2`）。**非逐位一致**：它换掉内核（`F.rms_norm`、融合 int4 GEMM、融合音频头）
  以削减每步核数，因此解码出另一条同样合法的 utterance。热图下 **~97 步/s**（vs
  `--fast` 的 ~81），文本 argmax 100%、音频 top-25 97.5%（`--fast` 同口径 75.4%）。
  ⚠️ **一次性 CLI 调用下它更慢**（实测 105 vs 28 ms/步）：整步图把注意力长度烘进内核，
  每个新解码长度都要单独 capture（144 步 utterance = 143 次），而 `fast.py` 的子图与
  长度无关。只有在长驻进程反复合成（图已热）时才有收益。另有 `arm="n1"`：其余逐字复用
  `fast.py`，与它**逐位一致**（zh 164 / en 190 步文本音频逐行相等），但只快 ~2%；
- **fast W4**：骨干 36 层 × 7 个 linear 的 int4 group 量化。两套权重来源：
  - **GPTQ 校准量化**（`--quant w4gptq` / `w4gptq:w2`，**推荐**）：用蒸馏语料的分层激活做
    Hessian 校准，含误差补偿与细分组，并按其自身误差排名把最敏感投影/层保 bf16；
  - **RTN 直接量化**（`--quant w4` / `w4g32`，perf-m4 原语义）：无需离线产物，随时可用。

  GPTQ 档同时修掉了 RTN 的非终止近静音 runaway 缺陷（verdict-1 §五）：
  同 12 个种子下 GPTQ **0/12**，RTN **3/12**。

## 安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # torch >= 2.9, CUDA 12.x 构建
pip install numpy soundfile pyyaml
```

4 个第三方依赖：`torch`（CUDA 12.x 构建，Ampere/sm_80 及以上 GPU）、`numpy`、`soundfile`、`pyyaml`。
模型目录（默认相对仓库根）：

- `models/MOSS-TTS-v1.5/`（TTS 主模型，safetensors）
- `models/MOSS-Audio-Tokenizer/`（音频 codec）

可用 `--model-dir` / `--codec-dir` 或环境变量 `MOSS_TTS_MODEL_DIR` / `MOSS_AUDIO_MODEL_DIR` 指定。

## CLI 用法

入口是**包级模块**：`python -m moss_tts_lite`（不是 `python -m moss_tts_lite.cli`——cli.py 自身无 `__main__` 块）。

```bash
# 1) 原始 eager 路径（基准，无需 CUDA graph）
python -m moss_tts_lite "你好，欢迎收听。" -o out_eager.wav

# 2) fast bf16：与 eager 逐位一致（EXACT），36 步/s，需 ~17 GiB
python -m moss_tts_lite "你好，欢迎收听。" -o out_fast.wav --fast

# 3) GPTQ-W4 默认档（W1：g32 + v_proj 保 bf16）：~81 步/s，~8.3 GiB
python -m moss_tts_lite "你好，欢迎收听。" -o out_w4gptq.wav --quant w4gptq

# 4) GPTQ-W4 质量档（W2：W1 + 4 层 up_proj 保 bf16）：~78 步/s，~8.5 GiB
python -m moss_tts_lite "你好，欢迎收听。" -o out_w4gptq2.wav --quant w4gptq:w2

# 5) RTN W4（perf-m4 语义，无离线产物也能跑）：~91 步/s，~7.5 GiB
python -m moss_tts_lite "你好，欢迎收听。" -o out_w4.wav --quant w4

# 7) RTN W4 细分变体（group-32）：~84 步/s，~8.1 GiB
python -m moss_tts_lite "你好，欢迎收听。" -o out_w4g32.wav --quant w4g32

# 8) 独立（自包含）量化目录：单下载即用，无需 16.6 GB bf16 基座（见下节）
python -m moss_tts_lite "你好，欢迎收听。" -o out_standalone.wav \
    --model-dir models_export/MOSS-TTS-v1.5-W4GPTQ-w1 \
    --codec-dir models/MOSS-Audio-Tokenizer

# 常用参数
python -m moss_tts_lite "Text here." -o out.wav \
    --language English \      # 语言标签（如 'English'）
    --seed 1234 \             # 采样种子（缺省随机）
    --greedy \                # 贪心（argmax）解码，替代种子采样
    --max-new-tokens 2000 \   # 解码步数上限（缺省 4096）
    --device cuda             # 'cuda'（缺省）或 'cpu'（cpu 不支持 fast/quant）
```

注意：`--quant` 会**自动启用** `--fast`（量化只在 fast 路径上生效），终端会打印一行提示；
单独 `--quant` 不再会出现"被静默忽略、实际跑 eager"的组合。

## 路径对照表（A10G-24G / torch 2.9.1 实测；其他卡等比参考）

| 路径 | 速度 | VRAM(峰值) | RTF | 保真 |
|---|---|---|---|---|
| eager bf16 | 29.7 步/s | ~17 GiB | 0.42 | 数值基准 |
| `--fast`（bf16） | **36.0 步/s** | 16.98 GiB | 0.22 | **EXACT 逐位一致**（zh/en 全轨迹 0 翻转） |
| `--quant w4gptq`（W1p，**默认量化档**） | **81.6 步/s** | 8.28 GiB | 文本 argmax 100%；十语言 pooled cover **99.52%**（tie-robust，最低语言格 99.12）；golden top-25 98.64%；runaway **0/12** |
| `--quant w4gptq:w2`（W2，质量档） | 78.5 步/s | 8.53 GiB | 文本 argmax 100%；音频 top-25 **91.25%**（CUDA topk 口径）/ **99.30%**（tie-robust）；多语言 +0.545 pt、情感 +1.020 pt；runaway **0/12** |
| `--quant w4`（RTN g128） | **91.5 步/s** | 7.48 GiB | 文本 argmax 100%；音频 top-25 75.4%；runaway 3/12 |
| `--quant w4g32`（RTN g32） | 84.0 步/s | 8.08 GiB | 文本 argmax 100%；音频 top-25 81.4% |
| `--quant w8`（显存模式） | 10.7 步/s | 10.51 GiB | 文本 100%；音频 top-25 94.8%（**慢于实时，仅省显存**） |
| `--fast-native`（W4，**热图**） | **97.1 步/s** | 8.3 GiB | 文本 argmax 100%；音频 top-25 **97.5%**；**非**逐位一致（换内核）；一次性调用更慢（见上） |

速度硬线 ≥78 步/s：W1/W2 均过线（对标 llama.cpp Q8_0 对照档 73.2 步/s 仍更快）。
gate-2（top-25 成员率）的统计噪声：40 步 × 32 通道，解析 SE ≈ 0.85 个点，
bootstrap 95% CI ≈ ±3 个点 —— W1 与 W2 在统计上不可区分，**选档依据应是速度取舍**。

RTF = 合成 1 秒音频所需解码秒数（越小越好）。每步解码产出 1/12.5 s 音频。

## 质量权衡说明（读这一段再选档）

- **EXACT 的含义**：`--fast` 的 bf16 路径与原始 eager 路径**逐位相同**（CUDA graph 回放
  不引入任何数值偏差；注意力保持图外精确长度）。要求可复现/可审计选它。
- **GPTQ 档与 RTN 档的区别**：RTN 逐组四舍五入到 int4；GPTQ 用校准激活的 Hessian 做
  逐列误差补偿（列块 128 惰性更新 + 阻尼 Cholesky 逆），再对固定码本做闭合解的
  (scale, zero) 重拟合。校准语料 6200 个位置分六层：文本提示 / 语流稳态 / 静音边界 /
  长静音 / 收尾 / 深层长文（种子与文本清单落 `.tmp/gptq_agent/calib/manifest2.json`）。
  分层误差（用部署内核本身测）：**长静音段为 RTN 的 0.019×、收尾段 0.024×、语流段
  0.032×** —— 即病灶区（长静音/收尾）是保护最好的分层。详见 `.tmp/reports/gptq-2-final.md`。
- **W4 改变了什么**：骨干线性层权重从 bf16 量化为 int4（分组仿标量化）。文本头 argmax
  在 40 步 teacher-forced 测试下 100% 稳健；音频码本的 logits 分布非常密集（golden
  top1-top2 gap 中位数为 0，63.9% 的位置是精确并列），任何权重量化都会改变这些近平局
  的采样分布。**"音频 top-25 成员率"**（golden top1 是否仍落在量化模型 logits 的
  top-25 内，对齐实际采样的 top_k=25）：g128 为 75.4%，g32 为 81.4%。
- **最终裁决是听感**：实际生成用种子采样（temperature/top_p/top_k），不是 argmax；
  E2E 结构（收尾、时长、非 pad 码占比）与 golden 高度接近。建议用同一段文本分别在
  `--fast`、`--quant w4gptq`、`--quant w4gptq:w2` 下生成，A/B 试听后再定档。
  速度优先选 `w4gptq`（W1p，81.6 步/s，99.52%），分布保真优先选 `w4gptq:w2`（W2，tie-robust 99.74%）。
  速度余量很窄：W2 距 78 步/s 硬底线仅约 0.6%，再加 bf16 保护项前必须先重测速度。
- `--quant w8` 不推荐用于加速：本栈（torch 2.9.1/A10G）的 `_weight_int8pack_mm` 无
  融合 decode kernel，比 bf16 慢 3.5 倍，仅作 10.5 GiB 显存档保留（实验性）。

## 看门狗 v2（生成失控兜底）

W4 档在极少数种子下会出现"非终止近静音"生成（verdict-1 §五：RTN 3/12）。`generate_fast`
内置两道规则，默认开启（`--no-watchdog` 关闭）：

| 规则 | 触发条件 | 参数 |
|---|---|---|
| (a) 静音 | 连续 ≥N 帧低能量（ch0 ∈ 实测低能量码集合） | `--watchdog-silence-frames`，默认 **64 帧（5.12 s）**；文本含 `[pause Xs]` 时自动抬高到覆盖该停顿 |
| (b) 段长兜底 | 当前段 > 640 帧（51.2 s）**且**当前尾部已连续 ≥32 帧（2.56 s）静音 | `--watchdog-max-segment-frames`，默认 **640** |

阈值来源：verdict 100 段语料中正常生成的连续低能量 ch0 游程 ≤44 帧（99 分位 44，golden ≤8），
3 段失控分别达 190/204/319 帧 → 64 帧阈值落在中间（正常上限的 1.45×，最短失控的 1/3）；
段长地板 640 相对正常最长段 464 帧留 38% 余量，且规则 (b) 额外要求尾部已静音，
因此**持续出声的健康长段永不误触发**。触发时强制收尾并置 `watchdog_triggered`，
CLI 打印 `WARNING: production watchdog triggered`。回归见
`python3 -m moss_tts_lite.tests.test_fast_watchdog`（四相：静音/段长/正例/持续发声）。

**GPTQ 档已不需要看门狗兜底**（0/12 失控），但看门狗默认保留作为所有档位的最后防线。

## top-25 数字的口径（**报数必须注明，否则数字无意义**）

“音频 top-25 命中率”（参考 argmax 是否落在候选的 25 大之内）在**并列（tie）**分布上
不是一个唯一定义的量：参考分布中 63.9% 的声道 top1==top2、20.1% 的声道有 ≥25 个值并列在
最大值。同一份 logits，仅换并列打破方式，分数可以从 80.2% 变到 100%；一个与参考**逐位相同**
的模型在 CUDA `topk` 口径下也只有 86.8%。因此本仓库从此**双口径报告**：

| 口径 | 定义 | 用途 |
|---|---|---|
| **tie-robust `cover`**（推荐） | 参考 argmax 的值 ≥ 候选第 25 大的值（并列一律算命中） | 与排序实现无关、可复现、可跨实现比较 |
| CUDA `topk` | `torch.topk` 在 CUDA 上的并列顺序（= 历史门禁口径） | 与旧报告直接对照 |

两个口径**方向一致，幅度不可互比**：tie 主导的参考会把同等真实增益放大约 7 倍
（W2 的真实语言分层增益是 +0.545 pt，在旧口径下读作 +1.25 pt）。
评估方法、并列证据与五口径对照见 `.tmp/reports/gencheck-1-lang-emo.md`；
W2 档位于 2026-09-11 由 “up 最差 4 层” 配方换成 “按绝对误差预算的 down 4 层” 配方
（同形状、同体积、同速度），依据见 `.tmp/reports/gencheck-2-alt4.md` 与
gencheck-3-w2-replacement.md。

## GPTQ 权重（离线产物）

GPTQ 档需要离线 state 文件（~4 GiB，**不入 git**）。解析顺序：
`--gptq-state PATH` > `$MOSS_TTS_GPTQ_DIR` > `<model_dir>/gptq/`。
默认内置预设：`w1p.pt`（`--quant w4gptq`，默认档）、`w2.pt`（`--quant w4gptq:w2`）、`w1.pt`（`--quant w4gptq:w1`，旧基线，保留作溯源）
各需同名 `.meta.json`（记录逐线性 group size 与 bf16 保留项）。
**`w2.pt` 的配方**：g32 全量量化 + 36 层 `v_proj` + `down_proj` 第 6/16/33/35 层保留 bf16
（按绝对误差预算选出；旧配方保 `up_proj` 第 6/7/9/35 层，已保留为 `w2_up4_legacy.pt`）。

缺失时**报错并打印再生成命令**（不静默换档；`--quant w4gptq:auto` 才回退到 `w4`）。
再生成（GPU，约 35 分钟；校准语料需先由 `.tmp/gptq_agent/calib_expand.py` 生成）：

```bash
# 1) GPTQ 量化（分两半跑；K=12288 时 Hessian+逆+17GB 模型同住会超 23.5 GiB）
python3 .tmp/gptq_agent/run_gptq.py --group-size 32 --damp 0.1 \
    --scale-from compensated --inverse-device cpu --layers 0:17 --tag g32_a
python3 .tmp/gptq_agent/run_gptq.py --group-size 32 --damp 0.1 \
    --scale-from compensated --inverse-device cpu --layers 18:35 --tag g32_b
# 2) 合并两半
python3 .tmp/gptq_agent/merge_states.py --union gptq_state_g32_a.pt,gptq_state_g32_b.pt \
    --union-group 32 --out gptq_state_g32.pt
# 3) 合成部署档（v_proj 全 36 层保 bf16 = W1）
python3 .tmp/gptq_agent/merge_states.py --base gptq_state_g32.pt --base-group 32 \
    --bf16-linears $(python3 -c "print(','.join(f'{i}:v' for i in range(36)))") \
    --out models/MOSS-TTS-v1.5/gptq/w1.pt
# 4) W2 = W1 + 6/7/9/35 层 up_proj 保 bf16
python3 .tmp/gptq_agent/merge_states.py --base gptq_state_g32.pt --base-group 32 \
    --bf16-linears $(python3 -c "print(','.join([f'{i}:v' for i in range(36)]+[f'{i}:up' for i in (6,7,9,35)]))") \
    --out models/MOSS-TTS-v1.5/gptq/w2.pt
```

## 独立量化模型（自包含目录，--model-dir 直接可用）

W1/W2 也可发布为 **自包含目录**：一个 7.1/7.4 GiB 的目录含 int4 载荷、全部 bf16
保留件、分词器、配置与模型卡，**不再需要 16.6 GB 的 bf16 基座**，也不再需要
单独的 `--gptq-state`。二进制与“基座 + state”补丁路径**逐位相同**（张量字节 / 40 步
logits / 端到端 wav / 速度与显存四个层面均已验收，见
`.tmp/reports/export-1-standalone.md`）。

### 下载后直接用（HF / ModelScope）

```bash
# HF（或 modelscope download --model OpenMOSS-Team/MOSS-TTS-v1.5-W4GPTQ-w1 --local_dir ./w1）
huggingface-cli download OpenMOSS-Team/MOSS-TTS-v1.5-W4GPTQ-w1 --local-dir ./MOSS-TTS-v1.5-W4GPTQ-w1
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer      --local-dir ./MOSS-Audio-Tokenizer

# 直接推理：目录里有 meta.json 时会自动识别量化档（无需 --quant，无需 --gptq-state）
python -m moss_tts_lite "你好，欢迎收听这段试音。" -o out.wav \
    --model-dir ./MOSS-TTS-v1.5-W4GPTQ-w1 --codec-dir ./MOSS-Audio-Tokenizer

# 英文 / 质量档（w2 目录同理）
python -m moss_tts_lite "Hello, this is a test." -o out_en.wav --language English \
    --model-dir ./MOSS-TTS-v1.5-W4GPTQ-w2 --codec-dir ./MOSS-Audio-Tokenizer
```

行为说明：

- 目录内含 `meta.json` 即走独立装配（`moss_tts_lite.export.load_standalone_model`），
  并自动启用 fast/quant 解码（终端会打印一行提示）；**非**独立目录的 `--model-dir`
  路径语义完全不变。
- 独立目录自带分词器文件，不会去读 `models/MOSS-TTS-v1.5/`。
- `--quant w4gptq[:w2]` 可与独立目录同时给出，**仅作档位校验**：档位不符会硬错误
  并列出可用档位（无静默换档）。
- `--gptq-state` 与独立目录组合是**硬错误**（量化已在该目录内）。

Python API：

```python
from moss_tts_lite.export import load_standalone_model   # -> (MossTTSModel, GptqMossTTS)
model, fast = load_standalone_model("./MOSS-TTS-v1.5-W4GPTQ-w1")
```

### 自己再导出（把基座 + state 变成独立目录）

```bash
# W1（CPU 即可，约 30 s：mmap 读基座 + 校验 16.6 GiB sha256）
python3 -m moss_tts_lite.export --model-dir models/MOSS-TTS-v1.5 \
    --gptq-state models/MOSS-TTS-v1.5/gptq/w1.pt \
    --out models_export/MOSS-TTS-v1.5-W4GPTQ-w1 --preset w1

# W2
python3 -m moss_tts_lite.export --model-dir models/MOSS-TTS-v1.5 \
    --gptq-state models/MOSS-TTS-v1.5/gptq/w2.pt \
    --out models_export/MOSS-TTS-v1.5-W4GPTQ-w2 --preset w2

# 常用开关：--dry-run（只出清单 + meta，不写文件）
#           --no-hash-base（跳过基座 sha256，快）
#           --force（覆盖已有输出）
#           --metrics-json FILE（嵌入自定义实测指标）
```

目录内容：`quantized.safetensors`（int4 载荷 `layers.{i}.{proj}.q/.qsz` +
全部 bf16 保留件，沿用原 key 名）、`meta.json`（格式/量化参数/group size/
bf16 清单/基座 sha256）、分词器与 config 全套、模型卡 `README.md`、`LICENSE`。
`models_export/` 已加入 `.gitignore`（7 GiB 级产物不入库）。

上传 HF/ModelScope 的命令模板见 `.tmp/reports/export-1-standalone.md` §7。

## Windows 部署要点

以下内容基于内核/算子分析给出；本包全部实测在 Linux + A10G 完成，
**标注【未实测】的项请在目标机先用小样验证**。

1. 依赖同上；W4 路径要求 **CUDA ≥12.0 构建 + sm_80（Ampere）及以上**，否则报
   "not available for build"。【未实测：Windows 轮子安装】
2. 显存选档：24GB→`--fast`；16GB→`--fast`（紧）或 `--quant w4`；12GB→`--quant w4`；
   8GB→`--quant w4` 依赖 CLI 的 OOM 自动回退（TTS 与 codec 分时占用），不建议
   codec 同卡常驻。
3. CUDA graph：本实现为 38 个分段子图（共享 memory pool + 独立 capture stream，
   捕获前 warmup 3 次）；**不要**把注意力卷进子图（会破坏 EXACT）。Windows WDDM 下
   capture 支持但首次更慢【未实测】。
4. 分配器环境变量：torch 2.9 起更名 `PYTORCH_ALLOC_CONF`（旧名 `PYTORCH_CUDA_ALLOC_CONF`
   会打 deprecation warning，仍生效）。`expandable_segments:True` 在 Windows 的支持
   状态见 torch 文档【未实测】，若异常可去掉该变量重试。
5. 入口必须是 `python -m moss_tts_lite`。

## 复现测试（GPU；Linux 开发环境命令，Windows 下去掉 flock 部分）

```bash
# M1：fast bf16 EXACT 门禁（phase0 bitwise + zh/en 全轨迹 EXACT + 速度）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
    python3 -m moss_tts_lite.tests.test_fast

# M4：W4 套件（文本 argmax 硬门禁 + E2E wav + 速度/VRAM/RTF）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
    python3 -m moss_tts_lite.tests.test_fast_m4

# fast-native：n1 逐位门禁 + n2 质量门 + 三个 bug 回归（见 native-1 报告）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
    python3 -m moss_tts_lite.tests.test_fast_native

# 量化质量全梯度 / int4pack 内核约定黑盒验证（探针脚本在 .tmp/perf_agent/）
flock .tmp/gpu.lock python3 .tmp/perf_agent/probe_qquality.py
flock .tmp/gpu.lock python3 .tmp/perf_agent/probe_w4bb.py
```

详细性能报告与踩坑记录见 `.tmp/reports/perf-final.md`（及 perf-m1/m3/m4 分报告）；
fast-native 档见 `.tmp/reports/native-1.md`（消融、被否决路线、CLI 冷热图代价）。
