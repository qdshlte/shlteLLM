# SHLTE LLM

一个 Rust 编写的 LLM 训练与推理工具集，支持从数据准备到模型训练的全流程，以及 GGUF 模型的本地推理。

**版本**: 3.2.2

---

## ✨ 核心功能

- **模型训练** — 完整训练流程，支持 CPU/CUDA/ROCm/MPS
- **TUI 聊天** — 交互式终端聊天界面，支持流式生成和对话历史
- **数据下载** — 自动从 HuggingFace 下载数据集，支持镜像站加速
- **数据预处理** — 支持 Parquet / JSONL / CSV / TXT 格式
- **分词器训练** — BPE / WordPiece / Unigram / SentencePiece
- **GGUF 导出** — 模型导出为 GGUF 格式，兼容 llama.cpp
- **配置验证** — 自动生成配置并提供详细的合法性检查

---

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/shlteLLM.git
cd shlteLLM

# 编译发布版本
cargo build --release
```

编译后的二进制文件位于 `target/release/shlteLLM`。

---

## 🚀 快速开始

### 1. 生成配置文件

```bash
# 生成预设配置 (tiny / small / base)
shlteLLM generate --preset base -o config.toml
```

### 2. 验证配置

```bash
shlteLLM validate -c config.toml
```

### 3. 开始训练

```bash
shlteLLM train -c config.toml -o output
```

### 4. 交互聊天

```bash
# 启动 TUI 聊天界面
shlteLLM chat

# 或在命令行中直接使用提示词
shlteLLM chat -p "你好，请介绍一下你自己"
```

---

## 📋 命令详解

### `train` — 模型训练

```bash
shlteLLM train [OPTIONS]
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-c, --config` | 配置文件路径 | `config.toml` |
| `-o, --output` | 输出目录 | `output` |
| `--resume` | 从检查点恢复 | - |
| `--preset` | 使用预设配置 | `tiny/small/base` |

### `chat` — 交互聊天

```bash
shlteLLM chat [OPTIONS]
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-p, --prompt` | 非交互模式提示词 | - |
| `-m, --model` | 模型文件路径（`.gguf`） | 自动搜索当前目录 |
| `--max-tokens` | 最大生成 token 数 | `512` |
| `--temperature` | 温度参数 (0.0-2.0) | `0.7` |
| `--top-p` | Top-p 采样参数 | `0.9` |
| `--repeat-penalty` | 重复惩罚系数 | `1.1` |
| `--context-size` | 上下文长度 | `2048` |
| `--threads` | 线程数 | `4` |

### `download` — 数据集下载

```bash
shlteLLM download -d <DATASET> -o <OUTPUT>
```

支持从 HuggingFace 及镜像站下载数据集。

### `preprocess` — 数据预处理

```bash
shlteLLM preprocess -i <INPUT> -o <OUTPUT> -t <TOKENIZER_CONFIG>
```

### `train-tokenizer` — 训练分词器

```bash
shlteLLM train-tokenizer -i <INPUT> -o <OUTPUT> --algorithm bpe --vocab_size 32000
```

支持的算法：`bpe`、`wordpiece`、`unigram`、`sentencepiece`。

### `validate` — 验证配置

```bash
shlteLLM validate -c config.toml
```

检查数据集百分比、模型维度整除性、GQA 组数、学习率、dropout 范围等。

### `generate` — 生成配置

```bash
# 生成单个预设
shlteLLM generate --preset base -o config.toml

# 批量生成所有预设
shlteLLM generate-presets -o ./presets
```

### `export` — 导出模型

```bash
shlteLLM export -i model.json -o model.gguf --context-size 2048
```

### `inspect` — 检查文件

```bash
shlteLLM inspect -p ./data/train.parquet
```

可查看文件大小、格式预览，以及 GGUF 模型元信息。

### `clean` — 清理缓存

```bash
shlteLLM clean -c ./cache
```

---

## ⚙️ 配置说明

### 数据集配置 `[dataset]`

```toml
[dataset]
size_gb = 1.0
num_shards = 4
cache_dir = "./cache"

[[dataset.mix]]
name = "OpenHermes-2.5"
percentage = 0.6
split = "train"

[[dataset.mix]]
name = "code-alpaca"
percentage = 0.4
split = "train"
```

下载源类型：

| 类型 | 说明 |
|------|------|
| `HuggingFace` | 直接从 HuggingFace Hub 下载 |
| `Mirror { url }` | 使用镜像站 |
| `CustomUrl { url }` | 自定义 URL |
| `Local` | 使用本地文件 |

### 模型配置 `[model]`

```toml
[model]
num_layers = 12
hidden_dim = 768
num_heads = 12
max_position_embeddings = 512
vocab_size = 32000
attention = { GQA = { num_groups = 4 } }
activation = "SwiGLU"
position_encoding = "RoPE"
normalization = "RMSNorm"
```

支持的注意力类型：`MHA`、`GQA { num_groups }`、`MQA`、`SlidingWindow { window_size }`、`FlashAttention`。

### 训练配置 `[training]`

```toml
[training]
learning_rate = 3e-4
batch_size = 32
micro_batch_size = 8
num_steps = 10000
warmup_steps = 500
sequence_length = 512
gradient_accumulation_steps = 4
mixed_precision = "BF16"
ema_decay = 0.999
grad_clip = 1.0

[training.lr_scheduler]
type = "Cosine"
min_lr = 1e-5

[training.optimizer]
type = "AdamW"
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

### 硬件配置 `[hardware]`

```toml
[hardware]
device = "Auto"
gpu_ids = [0, 1]
num_workers = 8
use_tf32 = true
memory_prealloc = false
```

---

## 📐 预设模型规格

| 预设 | 层数 | 隐藏维度 | 头数 | 词表大小 | 序列长度 |
|------|------|----------|------|----------|----------|
| tiny | 4 | 256 | 4 | 4,096 | 128 |
| small | 8 | 512 | 8 | 8,192 | 256 |
| base | 12 | 768 | 12 | 32,768 | 512 |

---

## 🧩 功能支持矩阵

| 组件 | 支持选项 |
|------|----------|
| 激活函数 | SwiGLU, GELU, ReLU, SiLU, GEGLU |
| 位置编码 | RoPE, ALiBi, NoPE, Learned, Sinusoidal |
| 归一化 | RMSNorm, LayerNorm, PreLayerNorm, PostLayerNorm |
| 优化器 | AdamW, Adam, SGD, LAMB |
| 学习率调度 | Linear, Cosine, CosineWithRestarts, Constant, OneCycle |
| 混合精度 | FP16, BF16, FP8 |
| 注意力机制 | MHA, GQA, MQA, SlidingWindow, FlashAttention |
| 分词算法 | BPE, WordPiece, Unigram, SentencePiece |

---

## 📁 输出结构

```
output/
├── checkpoints/
│   ├── checkpoint_step_1000_loss_4.3210/
│   │   ├── model.json
│   │   ├── ema_model.json
│   │   └── state.json
│   └── best_model/
├── logs/
│   ├── training.log
│   └── training.csv
├── tensorboard/
├── preprocessed/
├── final_model.json
├── ema_model.json
├── tokenizer.json
├── training_history.json
└── training.db
```

SQLite 数据库 (`training.db`) 记录完整训练指标、检查点信息和系统日志。

---

## 🔄 从检查点恢复

```bash
shlteLLM train \
  -c config.toml \
  -o ./output \
  --resume ./output/checkpoints/checkpoint_step_500_loss_4.3210
```

恢复内容包括：模型权重、EMA 权重、优化器状态、训练步数及损失记录。

---

## 💬 聊天界面命令

在 TUI 中使用 `/help` 查看可用命令：

| 命令 | 说明 |
|------|------|
| `/load <路径>` | 加载 GGUF 模型 |
| `/ls [目录]` | 列出可用模型 |
| `/info` | 显示模型信息 |
| `/clear` | 清空对话历史 |
| `/stop` | 停止当前生成 |
| `/params <n> <t> <p> <r>` | 设置生成参数 |
| `/save <路径>` | 保存对话历史 |
| `/loadhist <路径>` | 加载对话历史 |
| `/quit` | 退出程序 |

键盘快捷键：

| 按键 | 说明 |
|------|------|
| `i` | 进入插入模式 |
| `Esc` | 退出插入模式 |
| `j` / `k` | 向下 / 向上滚动 |
| `g` / `G` | 滚动到顶部 / 底部 |
| `q` | 退出 |
| `c` | 停止生成 |

---

## 🛠️ 构建选项

```bash
# 默认构建（包含 Parquet/Arrow 支持）
cargo build --release

# 仅构建基础功能（不包含 Parquet/Arrow）
cargo build --release --no-default-features

# 启用 llama-cpp 后端（需要 libllama）
cargo build --release --features llama-cpp
```

---

## 📜 许可证

本项目基于 MIT License 开源。

作者：QD·shlte
