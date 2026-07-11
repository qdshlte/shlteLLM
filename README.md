**SHLTE - LLM**
---
一个用 Rust 编写的高性能大语言模型训练框架，覆盖从数据下载、预处理、分词器训练到模型训练和评估的完整流程。

---

✨ 核心特性

· 模块化设计 – 数据、模型、训练、数据库完全解耦，易于扩展和定制
· 丰富的模型架构 – 支持 MHA、GQA、MQA、滑动窗口注意力、Flash Attention 等多种注意力机制
· 灵活配置系统 – 基于 TOML 的配置文件，提供 tiny、small、base 三种预设模板
· 高效数据预处理 – 支持 Parquet、JSONL、CSV、纯文本等多种输入格式
· 混合精度训练 – 支持 FP16、BF16、FP8，显著降低显存占用
· 训练状态持久化 – 完善的检查点保存与恢复机制，配合 SQLite 数据库记录全部训练指标
· 梯度累积 – 通过 micro_batch_size 支持等效大 batch 训练
· EMA 模型 – 指数移动平均，提升模型稳定性与泛化能力
· 多种优化器 – AdamW、Adam、SGD、LAMB
· 灵活的学习率调度 – Linear、Cosine、CosineWithRestarts、Constant、OneCycle

---

🚀 快速上手

安装

```bash
# 克隆仓库并编译
git clone https://github.com/yourusername/shltechat.git
cd shltechat
cargo build --release

# 编译后的二进制文件位于 target/release/shltechat
```

生成配置并开始训练

```bash
# 生成 tiny 预设配置（适合快速功能测试）
shltechat generate --preset tiny

# 开始训练
shltechat train
```

---

📖 命令详解

train – 模型训练

```bash
shltechat train [OPTIONS]
```

选项 说明 默认值
-c, --config 配置文件路径 config.toml
-o, --output 输出目录 output
--resume 从指定检查点恢复训练 -
--preset 使用预设配置 (tiny/small/base) -

download – 数据集下载

```bash
shltechat download -d <DATASET> -o <OUTPUT>
```

preprocess – 数据预处理

```bash
shltechat preprocess -i <INPUT> -o <OUTPUT> -t <TOKENIZER_CONFIG>
```

train-tokenizer – 训练分词器

```bash
shltechat train-tokenizer \
  -i <INPUT_FILE> \
  -o <OUTPUT_FILE> \
  --algorithm bpe \
  --vocab_size 32000
```

validate – 验证配置文件

```bash
shltechat validate -c config.toml
```

验证项包括：数据集百分比总和、隐藏维度整除性、GQA 组数合法性、序列长度限制、学习率正值检查、dropout 范围、滑动窗口约束等。

generate – 生成配置

```bash
# 生成指定预设
shltechat generate -o my_config.toml --preset base

# 批量生成所有预设到目录
shltechat generate-presets -o ./presets
```

其他实用命令

命令 用途
benchmark 基准测试：-c config.toml -o ./benchmark_output
inspect 检查数据：-p ./data/shard_0000.preprocessed.txt
clean 清理缓存：-c ./cache

---

⚙️ 配置详解

📊 [dataset] — 数据集配置

```toml
[dataset]
download_source = { Mirror = { url = "https://hf-mirror.com" } }
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

下载源类型 说明
HuggingFace 从 HuggingFace Hub 下载
Mirror { url } 使用镜像站
CustomUrl { url } 自定义 URL
Local 使用本地文件

🧠 [model] — 模型架构

```toml
[model]
num_layers = 12
hidden_dim = 768
num_heads = 12
max_position_embeddings = 512
attention = { GQA = { num_groups = 4 } }
activation = "SwiGLU"
position_encoding = "RoPE"
normalization = "RMSNorm"
```

注意力类型 说明
MHA 标准多头注意力
GQA { num_groups } 分组查询注意力
MQA 多查询注意力
SlidingWindow { window_size } 滑动窗口注意力
GQAWithSlidingWindow { num_groups, window_size } 组合注意力
FlashAttention Flash Attention

🎯 [training] — 训练参数

```toml
[training]
learning_rate = 3e-4
min_learning_rate = 1e-5
batch_size = 32
micro_batch_size = 8
num_steps = 10000
warmup_steps = 500
sequence_length = 512
gradient_accumulation_steps = 4
mixed_precision = "BF16"
ema_decay = 0.999
grad_clip = 1.0
lr_scheduler = { Cosine = { min_lr = 1e-5 } }
optimizer = { AdamW = { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } }
```

🔤 [tokenizer] — 分词器

```toml
[tokenizer]
algorithm = "BPE"
vocab_size = 32000
normalization = true
add_prefix_space = false
```

💻 [hardware] — 硬件配置

```toml
[hardware]
device = "Auto"
gpu_ids = [0, 1]
num_workers = 8
use_tf32 = true
memory_prealloc = false
```

📝 [logging] — 日志配置

```toml
[logging]
level = "info"
wandb_project = "my-llm-project"
wandb_entity = "my-team"
tensorboard_dir = "./logs/tensorboard"
csv_log_path = "./logs/training.csv"
```

---

📐 预设模型规格

预设 层数 隐藏维度 注意力头数 词表大小 序列长度
tiny 4 256 4 4,096 128
small 8 512 8 8,192 256
base 12 768 12 32,768 512

---

📁 输出结构

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

---

📥 数据预处理

格式 处理方式
Parquet 自动检测文本列 (text/content/data 等)
JSONL 自动提取文本字段，支持嵌套结构
CSV 自动检测文本列
TXT 逐行处理

处理流程： 读取源文件 → 提取文本 → 分词转 token ID → 按序列长度切分 → 保存预处理文件

---

🔄 从检查点恢复

```bash
shltechat train \
  -c config.toml \
  -o ./output \
  --resume ./output/checkpoints/checkpoint_step_500_loss_4.3210
```

恢复内容包括：模型权重、EMA 权重、优化器状态、训练步数及损失记录。

---

🗃️ 数据库记录

SQLite 数据库 (training.db) 记录以下完整信息：

· 数据集下载状态
· 预处理统计
· 训练运行记录
· 检查点信息
· 训练指标（loss、学习率、梯度范数、吞吐量等）
· 评估结果
· 系统事件日志
· 硬件信息

---

🧩 架构支持矩阵

组件 支持选项
激活函数 SwiGLU, GELU, ReLU, SiLU, GEGLU
位置编码 RoPE, ALiBi, NoPE, Learned, Sinusoidal
归一化 RMSNorm, LayerNorm, PreLayerNorm, PostLayerNorm
优化器 AdamW, Adam, SGD, LAMB
学习率调度器 Linear, Cosine, CosineWithRestarts, Constant, OneCycle
混合精度 FP16, BF16, FP8

---

📜 许可证

本项目基于 MIT License 开源。