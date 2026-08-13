//! ============================================================================
//! 训练流水线模块
//! ============================================================================
//!
//! 本模块实现了完整的训练流水线，包括：
//! - TensorBoard事件日志写入（Protobuf格式）
//! - CSV日志管理
//! - 数据集下载与预处理
//! - 模型训练流程编排
//! - 基准测试
//! - 模型导出与压缩
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::config::Config;
use crate::data_one::{DataLoader, DataPreprocessor};
use crate::data_two::{BatchLoader, DataSplitter};
use crate::db::Database;
use crate::error::{Result, TrainError};
use crate::model::{ModelParams, Transformer};
use crate::tokenizer::Tokenizer;
use crate::train::Trainer;
use crc32fast::Hasher;
use prost::Message;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// TensorBoard Protobuf 定义
// ============================================================================

/// TensorBoard事件文件格式（简化版）
/// 参考: https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/event.proto

#[derive(Clone, PartialEq, Message)]
struct Event {
    #[prost(double, tag = "1")]
    wall_time: f64,
    #[prost(int64, tag = "2")]
    step: i64,
    #[prost(message, optional, tag = "3")]
    summary: Option<Summary>,
}

#[derive(Clone, PartialEq, Message)]
struct Summary {
    #[prost(message, repeated, tag = "1")]
    value: Vec<SummaryValue>,
}

#[derive(Clone, PartialEq, Message)]
struct SummaryValue {
    #[prost(string, tag = "1")]
    tag: String,
    #[prost(message, optional, tag = "3")]
    simple_value: Option<SimpleValue>,
    #[prost(message, optional, tag = "4")]
    histo: Option<HistogramProto>,
    #[prost(message, optional, tag = "7")]
    tensor: Option<TensorProto>,
}

#[derive(Clone, PartialEq, Message)]
struct SimpleValue {
    #[prost(double, tag = "1")]
    value: f64,
}

#[derive(Clone, PartialEq, Message)]
struct HistogramProto {
    #[prost(double, tag = "1")]
    min: f64,
    #[prost(double, tag = "2")]
    max: f64,
    #[prost(double, tag = "3")]
    num: f64,
    #[prost(double, tag = "4")]
    sum: f64,
    #[prost(double, tag = "5")]
    sum_squares: f64,
    #[prost(double, repeated, tag = "6")]
    bucket_limit: Vec<f64>,
    #[prost(double, repeated, tag = "7")]
    bucket: Vec<f64>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(enumeration = "DataType", tag = "1")]
    dtype: i32,
    #[prost(string, repeated, tag = "6")]
    string_val: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum DataType {
    #[default]
    DtInvalid = 0,
    DtFloat = 1,
    DtDouble = 2,
    DtInt32 = 3,
    DtUint8 = 4,
    DtInt16 = 5,
    DtInt8 = 6,
    DtString = 7,
    DtComplex64 = 8,
    DtInt64 = 9,
    DtBool = 10,
    DtQint8 = 11,
    DtQuint8 = 12,
    DtQint32 = 13,
    DtBfloat16 = 14,
    DtQint16 = 15,
    DtQuint16 = 16,
    DtUint16 = 17,
    DtComplex128 = 18,
    DtHalf = 19,
    DtResource = 20,
    DtVariant = 21,
    DtUint32 = 22,
    DtUint64 = 23,
    Unknown = 24,
    Float32 = 25,
    Float64 = 26,
}

impl From<i32> for DataType {
    fn from(value: i32) -> Self {
        match value {
            0 => DataType::DtInvalid,
            1 => DataType::DtFloat,
            2 => DataType::DtDouble,
            3 => DataType::DtInt32,
            4 => DataType::DtUint8,
            5 => DataType::DtInt16,
            6 => DataType::DtInt8,
            7 => DataType::DtString,
            8 => DataType::DtComplex64,
            9 => DataType::DtInt64,
            10 => DataType::DtBool,
            11 => DataType::DtQint8,
            12 => DataType::DtQuint8,
            13 => DataType::DtQint32,
            14 => DataType::DtBfloat16,
            15 => DataType::DtQint16,
            16 => DataType::DtQuint16,
            17 => DataType::DtUint16,
            18 => DataType::DtComplex128,
            19 => DataType::DtHalf,
            20 => DataType::DtResource,
            21 => DataType::DtVariant,
            22 => DataType::DtUint32,
            23 => DataType::DtUint64,
            24 => DataType::Unknown,
            25 => DataType::Float32,
            26 => DataType::Float64,
            _ => DataType::DtInvalid,
        }
    }
}

impl From<DataType> for i32 {
    fn from(dtype: DataType) -> Self {
        dtype as i32
    }
}

// ============================================================================
// TensorBoard 事件写入器
// ============================================================================

pub struct TensorBoardWriter {
    log_dir: PathBuf,
    event_file: std::fs::File,
    current_step: i64,
}

impl TensorBoardWriter {
    /// 创建新的TensorBoard写入器
    pub fn new(log_dir: &Path) -> Result<Self> {
        fs::create_dir_all(log_dir)?;

        let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let event_file_path = log_dir.join(format!(
            "events.out.tfevents.{}.{}",
            std::process::id(),
            timestamp
        ));

        let mut event_file = fs::File::create(&event_file_path)?;

        // 写入TensorBoard文件头（8字节版本号，全0）
        let header = [0u8; 8];
        event_file.write_all(&header)?;

        println!("📊 TensorBoard日志目录: {}", log_dir.display());
        println!("   查看命令: tensorboard --logdir {}", log_dir.display());

        Ok(TensorBoardWriter {
            log_dir: log_dir.to_path_buf(),
            event_file,
            current_step: 0,
        })
    }

    /// 计算CRC32C校验和（TensorBoard使用CRC32C）
    fn crc32c(data: &[u8]) -> u32 {
        let mut hasher = Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    /// 写入事件记录
    fn write_event(&mut self, event: &Event) -> Result<()> {
        // 序列化事件
        let mut buf = Vec::new();
        event.encode(&mut buf).map_err(|e| {
            TrainError::Serialization(serde_json::Error::io(std::io::Error::other(format!(
                "Protobuf编码失败: {}",
                e
            ))))
        })?;
        // 写入长度（uint64，小端序）
        let len = buf.len() as u64;
        self.event_file.write_all(&len.to_le_bytes())?;

        // 写入数据的CRC32C
        let data_crc = Self::crc32c(&buf);
        self.event_file.write_all(&data_crc.to_le_bytes())?;

        // 写入数据
        self.event_file.write_all(&buf)?;

        // 写入掩码长度（0）和CRC
        let mask_len = 0u64;
        self.event_file.write_all(&mask_len.to_le_bytes())?;
        let mask_crc = Self::crc32c(&[]);
        self.event_file.write_all(&mask_crc.to_le_bytes())?;

        self.event_file.flush()?;
        Ok(())
    }

    /// 记录标量值
    pub fn log_scalar(&mut self, tag: &str, value: f64, step: i64) -> Result<()> {
        self.current_step = step;

        let event = Event {
            wall_time: chrono::Utc::now().timestamp() as f64,
            step,
            summary: Some(Summary {
                value: vec![SummaryValue {
                    tag: tag.to_string(),
                    simple_value: Some(SimpleValue { value }),
                    histo: None,
                    tensor: None,
                }],
            }),
        };

        self.write_event(&event)
    }

    /// 记录多个标量
    pub fn log_scalars(&mut self, tags: &[(&str, f64)], step: i64) -> Result<()> {
        self.current_step = step;

        let values: Vec<SummaryValue> = tags
            .iter()
            .map(|(tag, value)| SummaryValue {
                tag: tag.to_string(),
                simple_value: Some(SimpleValue { value: *value }),
                histo: None,
                tensor: None,
            })
            .collect();

        let event = Event {
            wall_time: chrono::Utc::now().timestamp() as f64,
            step,
            summary: Some(Summary { value: values }),
        };

        self.write_event(&event)
    }

    /// 记录直方图
    pub fn log_histogram(&mut self, tag: &str, values: &[f32], step: i64) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0] as f64;
        let max = sorted[sorted.len() - 1] as f64;
        let sum: f64 = sorted.iter().map(|&v| v as f64).sum();
        let sum_squares: f64 = sorted.iter().map(|&v| (v as f64) * (v as f64)).sum();

        // 创建30个桶的直方图
        let num_bins = 30;
        let bucket_size = (max - min) / num_bins as f64;
        let mut buckets = vec![0.0; num_bins];

        for &val in &sorted {
            let idx = ((val as f64 - min) / bucket_size).floor() as usize;
            if idx < num_bins {
                buckets[idx] += 1.0;
            }
        }

        // 计算桶边界
        let bucket_limit: Vec<f64> = (1..=num_bins)
            .map(|i| min + i as f64 * bucket_size)
            .collect();

        let event = Event {
            wall_time: chrono::Utc::now().timestamp() as f64,
            step,
            summary: Some(Summary {
                value: vec![SummaryValue {
                    tag: tag.to_string(),
                    simple_value: None,
                    histo: Some(HistogramProto {
                        min,
                        max,
                        num: values.len() as f64,
                        sum,
                        sum_squares,
                        bucket_limit,
                        bucket: buckets,
                    }),
                    tensor: None,
                }],
            }),
        };

        self.write_event(&event)
    }

    /// 记录文本
    pub fn log_text(&mut self, tag: &str, text: &str, step: i64) -> Result<()> {
        let event = Event {
            wall_time: chrono::Utc::now().timestamp() as f64,
            step,
            summary: Some(Summary {
                value: vec![SummaryValue {
                    tag: tag.to_string(),
                    simple_value: None,
                    histo: None,
                    tensor: Some(TensorProto {
                        dtype: DataType::DtString as i32,
                        string_val: vec![text.to_string()],
                    }),
                }],
            }),
        };

        self.write_event(&event)
    }

    /// 刷新缓冲区
    pub fn flush(&mut self) -> Result<()> {
        self.event_file.flush()?;
        Ok(())
    }
}

// ============================================================================
// CSV日志管理器
// ============================================================================

pub struct CsvLogger {
    writer: csv::Writer<std::fs::File>,
    file_path: PathBuf,
}

// ===== 训练步指标 =====

#[derive(Debug, Clone, Default)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f64,
    pub eval_loss: Option<f64>,
    pub learning_rate: f64,
    pub tokens_per_second: f64,
    pub epoch: usize,
    pub gradient_norm: Option<f64>,
    pub gpu_memory_mb: Option<f64>,
}

impl CsvLogger {
    /// 创建新的CSV日志器
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record([
            "step",
            "loss",
            "eval_loss",
            "learning_rate",
            "tokens_per_second",
            "epoch",
            "gradient_norm",
            "gpu_memory_mb",
            "timestamp",
        ])?;
        writer.flush()?;

        Ok(CsvLogger {
            writer,
            file_path: path.to_path_buf(),
        })
    }

    /// 记录训练步数
    pub fn log_step(&mut self, metrics: &StepMetrics) -> Result<()> {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

        self.writer.write_record(&[
            metrics.step.to_string(),
            format!("{:.6}", metrics.loss),
            metrics.eval_loss.map(|l| format!("{:.6}", l)).unwrap_or_default(),
            format!("{:.8}", metrics.learning_rate),
            format!("{:.2}", metrics.tokens_per_second),
            metrics.epoch.to_string(),
            metrics.gradient_norm
                .map(|g| format!("{:.6}", g))
                .unwrap_or_default(),
            metrics.gpu_memory_mb
                .map(|m| format!("{:.1}", m))
                .unwrap_or_default(),
            timestamp.to_string(),
        ])?;
        self.writer.flush()?;
        Ok(())
    }

    /// 批量记录多个步数
    pub fn log_steps(&mut self, steps: &[(usize, f64, f64, f64)]) -> Result<()> {
        for (step, loss, lr, tps) in steps {
            self.log_step(&StepMetrics {
                step: *step,
                loss: *loss,
                eval_loss: None,
                learning_rate: *lr,
                tokens_per_second: *tps,
                epoch: 0,
                gradient_norm: None,
                gpu_memory_mb: None,
            })?;
        }
        Ok(())
    }

    /// 刷新缓冲区
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// 获取日志文件路径
    pub fn path(&self) -> &Path {
        &self.file_path
    }
}

// ============================================================================
// 下载数据集
// ============================================================================

pub fn download_dataset(dataset_name: &str, output_dir: &Path) -> Result<()> {
    println!("📥 下载数据集: {}", dataset_name);
    println!("   目标目录: {}", output_dir.display());

    let mut config = Config::tiny_preset();
    config.dataset.mix[0].name = dataset_name.to_string();
    config.dataset.cache_dir = output_dir.to_string_lossy().to_string();

    let loader = DataLoader::new(config.dataset);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| TrainError::Unknown(format!("创建运行时失败: {}", e)))?;
    let paths = rt.block_on(async { loader.download().await })?;

    println!("✅ 下载完成，文件数: {}", paths.len());
    for path in paths {
        println!("   - {}", path.display());
    }
    Ok(())
}

// ============================================================================
// 预处理数据
// ============================================================================

pub fn preprocess_data(input_path: &Path, output_path: &Path, tokenizer: &Tokenizer) -> Result<()> {
    println!("🔧 预处理数据...");
    println!("   输入: {}", input_path.display());
    println!("   输出: {}", output_path.display());

    let preprocessor = DataPreprocessor::new(tokenizer.clone(), 512);

    if input_path.is_dir() {
        preprocessor.preprocess_directory(input_path, output_path)?;
    } else {
        preprocessor.preprocess_file(input_path, output_path)?;
    }

    println!("✅ 预处理完成");
    Ok(())
}

// ============================================================================
// 完整训练流程
// ============================================================================

pub fn run_training(
    config: Config,
    output_dir: PathBuf,
    resume_from: Option<PathBuf>,
) -> Result<()> {
    println!("🚀 启动 SHLTE LLM 训练流程 v3.2.2");
    println!("================================================");

    // ========================================================================
    // 创建输出目录
    // ========================================================================
    fs::create_dir_all(&output_dir)?;
    let log_dir = output_dir.join("logs");
    let checkpoint_dir = output_dir.join("checkpoints");
    let tensorboard_dir = output_dir.join("tensorboard");
    let preprocessed_dir = output_dir.join("preprocessed");

    fs::create_dir_all(&log_dir)?;
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&tensorboard_dir)?;
    fs::create_dir_all(&preprocessed_dir)?;

    // ========================================================================
    // 初始化TensorBoard写入器
    // ========================================================================
    let mut tb_writer = TensorBoardWriter::new(&tensorboard_dir)?;
    tb_writer.log_text(
        "system/info",
        &format!("Training started at {}", chrono::Local::now()),
        0,
    )?;

    // ========================================================================
    // 初始化数据库
    // ========================================================================
    let db_path = output_dir.join("training.db");
    let db = Database::open(&db_path)?;
    println!("📁 数据库: {}", db_path.display());

    // ========================================================================
    // 步骤1: 下载数据集
    // ========================================================================
    println!("\n📥 步骤1/7: 下载数据集");
    let data_loader = DataLoader::new(config.dataset.clone());

    let rt = tokio::runtime::Runtime::new()?;
    let data_paths = rt.block_on(async { data_loader.download_streaming().await })?;

    // 记录数据集到数据库
    let mut dataset_ids = Vec::new();
    for source in config.dataset.mix.iter() {
        match db.add_dataset(
            &source.name,
            &config.dataset.download_source.to_string(),
            config.dataset.size_gb,
            config.dataset.num_shards,
        ) {
            Ok(id) => {
                dataset_ids.push(id);
                println!("   📋 数据集已记录: {} (ID: {})", source.name, id);
            }
            Err(e) => println!("   ⚠️ 记录数据集失败: {}", e),
        }
    }

    // ========================================================================
    // 步骤2: 准备分词器
    // ========================================================================
    println!("\n🔤 步骤2/7: 准备分词器");
    let tokenizer_path = output_dir.join("tokenizer.json");

    let tokenizer = if tokenizer_path.exists() {
        println!("   📂 加载已有分词器: {}", tokenizer_path.display());
        Tokenizer::load(&tokenizer_path)?
    } else {
        println!("   🔨 训练新分词器...");
        let mut tokenizer = create_default_tokenizer(&config)?;

        if !data_paths.is_empty() {
            let sample_texts = load_sample_texts(&data_paths[0], 50000)?;
            println!("   📖 使用 {} 条文本训练分词器", sample_texts.len());
            tokenizer.train_on_texts(&sample_texts)?;
        }

        tokenizer.save(&tokenizer_path)?;
        println!("   ✅ 分词器已保存: {}", tokenizer_path.display());
        tokenizer
    };

    println!("   📊 词表大小: {}", tokenizer.vocab_size());

    // ========================================================================
    // 步骤3: 预处理数据
    // ========================================================================
    println!("\n🔧 步骤3/7: 预处理数据");

    let preprocessor = DataPreprocessor::new(tokenizer.clone(), config.training.sequence_length);

    let mut preprocessed_paths = Vec::new();

    for (i, data_path) in data_paths.iter().enumerate() {
        let output_path = preprocessed_dir.join(format!("shard_{:04}.preprocessed.txt", i));

        if output_path.exists() {
            println!("   📂 使用已有预处理文件: {}", output_path.display());
        } else {
            println!("   📝 处理: {}", data_path.display());
            let stats = preprocessor.preprocess_file(data_path, &output_path)?;

            let dataset_id = dataset_ids.get(i).copied().unwrap_or(0);
            let _ = db.add_preprocessed_data(
                dataset_id,
                stats.vocab_size,
                stats.num_tokens,
                stats.num_sequences,
                stats.avg_sequence_length,
            );

            println!(
                "      Token数: {}, 序列数: {}",
                stats.num_tokens, stats.num_sequences
            );
        }

        preprocessed_paths.push(output_path);
    }

    // ========================================================================
    // 步骤4: 划分数据集
    // ========================================================================
    println!("\n📊 步骤4/7: 划分数据集");
    let val_ratio = 0.05;
    let (train_paths, val_paths) =
        DataSplitter::train_val_split(&preprocessed_paths, val_ratio, 42)?;

    println!("   训练集: {} 个文件", train_paths.len());
    println!("   验证集: {} 个文件", val_paths.len());

    // ========================================================================
    // 步骤5: 创建模型
    // ========================================================================
    println!("\n🧠 步骤5/7: 创建模型");
    let vocab_size = tokenizer.vocab_size();
    let model_params = ModelParams::from_config(&config.model, vocab_size);
    let model = Transformer::new(model_params);

    let num_params = model.num_parameters();
    println!("   📊 模型参数:");
    println!("      层数: {}", model.params.num_layers);
    println!("      隐藏维度: {}", model.params.hidden_dim);
    println!("      注意力头数: {}", model.params.num_heads);
    println!("      KV头数: {}", model.params.num_key_value_heads);
    println!("      中间层维度: {}", model.params.intermediate_dim);
    println!("      词表大小: {}", model.params.vocab_size);
    println!(
        "      最大位置编码: {}",
        model.params.max_position_embeddings
    );
    println!("      总参数量: {:.2}M", num_params as f64 / 1e6);

    // 记录超参数到TensorBoard
    tb_writer.log_scalar("config/learning_rate", config.training.learning_rate, 0)?;
    tb_writer.log_scalar("config/batch_size", config.training.batch_size as f64, 0)?;
    tb_writer.log_scalar("config/num_layers", config.model.num_layers as f64, 0)?;
    tb_writer.log_scalar("config/hidden_dim", config.model.hidden_dim as f64, 0)?;
    tb_writer.log_scalar("config/num_heads", config.model.num_heads as f64, 0)?;
    tb_writer.log_scalar("config/num_params", num_params as f64, 0)?;

    // ========================================================================
    // 步骤6: 准备数据加载器
    // ========================================================================
    println!("\n📊 步骤6/7: 准备数据加载器");
    let mut train_loader = BatchLoader::new(
        train_paths,
        config.training.batch_size,
        config.training.sequence_length,
    );
    train_loader.set_shuffle(true);
    train_loader.load_all()?;

    let mut eval_loader = BatchLoader::new(
        val_paths,
        config.training.batch_size,
        config.training.sequence_length,
    );
    eval_loader.set_shuffle(false);
    eval_loader.load_all()?;

    println!("   训练批次数: {}", train_loader.num_batches());
    println!("   验证批次数: {}", eval_loader.num_batches());

    // ========================================================================
    // 创建CSV日志器
    // ========================================================================
    let mut csv_logger = if let Some(ref csv_path) = config.logging.csv_log_path {
        let csv_path = PathBuf::from(csv_path);
        if let Some(parent) = csv_path.parent() {
            fs::create_dir_all(parent)?;
        }
        Some(CsvLogger::new(&csv_path)?)
    } else {
        None
    };

    // ========================================================================
    // 步骤7: 开始训练
    // ========================================================================
    println!("\n🚀 步骤7/7: 开始训练");
    println!("================================================");

    let mut trainer = Trainer::new(config.clone(), model, output_dir.clone());

    // 从检查点恢复
    if let Some(ref checkpoint_path) = resume_from {
        println!("📂 从检查点恢复: {}", checkpoint_path.display());
        match trainer.restore_checkpoint(checkpoint_path) {
            Ok(()) => {
                println!("   ✅ 检查点恢复成功");
            }
            Err(e) => {
                println!("⚠️  检查点加载失败: {}", e);
                println!("   从头开始训练");
            }
        }
    }

    // 记录训练开始到数据库
    let run_id = db.start_training_run(
        &format!(
            "{:x}",
            md5::compute(serde_json::to_string(&config).unwrap_or_default())
        ),
        &serde_json::to_string(&config.model).unwrap_or_default(),
        &serde_json::to_string(&config.training).unwrap_or_default(),
        config.training.num_steps,
        &format!("{:?}", config.hardware.device),
    )?;

    println!("   📋 训练运行ID: {}", run_id);

    // 执行训练
    trainer.train(&mut train_loader, &mut eval_loader)?;

    // ========================================================================
    // 记录训练指标到TensorBoard
    // ========================================================================
    println!("\n📊 写入TensorBoard日志...");

    for (step, &loss) in trainer.train_losses.iter().enumerate() {
        let step_i64 = step as i64;
        tb_writer.log_scalar("train/loss", loss, step_i64)?;

        if step < trainer.learning_rates.len() {
            tb_writer.log_scalar(
                "train/learning_rate",
                trainer.learning_rates[step],
                step_i64,
            )?;
        }
    }

    for (step, &loss) in trainer.eval_losses.iter().enumerate() {
        tb_writer.log_scalar("eval/loss", loss, (step * 100) as i64)?;
    }

    tb_writer.log_scalar(
        "train/best_loss",
        trainer.best_loss,
        trainer.current_step as i64,
    )?;
    tb_writer.flush()?;
    println!("   ✅ TensorBoard日志已写入: {}", tensorboard_dir.display());

    // ========================================================================
    // 记录CSV日志
    // ========================================================================
    if let Some(ref mut csv) = csv_logger {
        for (step, &loss) in trainer.train_losses.iter().enumerate() {
            let lr = if step < trainer.learning_rates.len() {
                trainer.learning_rates[step]
            } else {
                0.0
            };
            csv.log_step(&StepMetrics {
                step: step + 1,
                loss,
                eval_loss: None,
                learning_rate: lr,
                tokens_per_second: 0.0,
                epoch: 0,
                gradient_norm: None,
                gpu_memory_mb: None,
            })?;
        }
        csv.flush()?;
        println!("📄 CSV日志已写入: {}", csv.path().display());
    }

    // ========================================================================
    // 保存最终模型
    // ========================================================================
    println!("\n💾 保存最终模型");

    let final_model = trainer.get_model();
    let final_model_path = output_dir.join("final_model.json");
    final_model.save(&final_model_path)?;
    println!("   最终模型: {}", final_model_path.display());

    if let Some(ema_model) = trainer.get_ema_model() {
        let ema_path = output_dir.join("ema_model.json");
        ema_model.save(&ema_path)?;
        println!("   EMA模型: {}", ema_path.display());
    }

    // ========================================================================
    // 生成训练报告
    // ========================================================================
    println!("\n📊 生成训练报告");
    let report_path = output_dir.join("training_report.txt");
    let mut report = fs::File::create(&report_path)?;

    writeln!(report, "========================================")?;
    writeln!(report, "SHLTE LLM 训练报告 v3.2.2")?;
    writeln!(report, "========================================\n")?;
    writeln!(report, "训练统计:")?;
    writeln!(report, "  训练步数: {}", trainer.current_step)?;
    writeln!(report, "  最佳损失: {:.4}", trainer.best_loss)?;
    writeln!(
        report,
        "  最终损失: {:.4}",
        trainer.train_losses.last().unwrap_or(&0.0)
    )?;
    writeln!(report, "  总参数量: {:.2}M", num_params as f64 / 1e6)?;
    writeln!(report, "\n模型配置:")?;
    writeln!(report, "  层数: {}", config.model.num_layers)?;
    writeln!(report, "  隐藏维度: {}", config.model.hidden_dim)?;
    writeln!(report, "  注意力头数: {}", config.model.num_heads)?;
    writeln!(report, "  激活函数: {:?}", config.model.activation)?;
    writeln!(report, "  位置编码: {:?}", config.model.position_encoding)?;
    writeln!(report, "\n训练配置:")?;
    writeln!(report, "  学习率: {:.2e}", config.training.learning_rate)?;
    writeln!(report, "  批次大小: {}", config.training.batch_size)?;
    writeln!(report, "  序列长度: {}", config.training.sequence_length)?;
    writeln!(report, "  权重衰减: {}", config.training.weight_decay)?;
    writeln!(report, "  梯度裁剪: {}", config.training.grad_clip)?;

    println!("   📄 训练报告: {}", report_path.display());

    // 更新数据库
    let final_loss = *trainer.train_losses.last().unwrap_or(&0.0);
    db.complete_training_run(run_id, final_loss)?;

    // ========================================================================
    // 完成
    // ========================================================================
    println!("\n✨ 训练流程完成!");
    println!("   输出目录: {}", output_dir.display());
    println!("   检查点目录: {}", checkpoint_dir.display());
    println!("   日志目录: {}", log_dir.display());
    println!(
        "   TensorBoard: tensorboard --logdir {}",
        tensorboard_dir.display()
    );

    Ok(())
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 创建默认分词器
fn create_default_tokenizer(config: &Config) -> Result<Tokenizer> {
    let tokenizer_path = PathBuf::from("tokenizer.toml");

    if !tokenizer_path.exists() {
        let tokenizer_config = crate::config::TokenizerConfig {
            algorithm: config.tokenizer.algorithm.clone(),
            vocab_size: config.tokenizer.vocab_size,
            special_tokens: config.tokenizer.special_tokens.clone(),
            normalization: config.tokenizer.normalization,
            add_prefix_space: config.tokenizer.add_prefix_space,
        };

        let content = toml::to_string_pretty(&tokenizer_config)
            .map_err(|e| TrainError::Tokenizer(format!("无法序列化分词器配置: {}", e)))?;

        fs::write(&tokenizer_path, content)?;
        println!("✅ 已生成默认分词器配置: {}", tokenizer_path.display());
    }

    Tokenizer::from_file(&tokenizer_path)
}

/// 加载样本文本用于训练分词器
fn load_sample_texts(data_path: &Path, max_samples: usize) -> Result<Vec<String>> {
    let mut samples = Vec::new();

    let content = fs::read_to_string(data_path)?;

    for line in content.lines().take(max_samples * 2) {
        let line = line.trim();
        if !line.is_empty() {
            samples.push(line.to_string());
        }

        if samples.len() >= max_samples {
            break;
        }
    }

    Ok(samples)
}

// ============================================================================
// 验证与清理功能
// ============================================================================

/// 验证数据集完整性
pub fn validate_dataset(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 {
        return Ok(false);
    }

    if let Ok(content) = fs::read_to_string(path) {
        if let Some(first_line) = content.lines().next() {
            return Ok(!first_line.trim().is_empty());
        }
    }

    Ok(false)
}

/// 验证预处理数据
pub fn validate_preprocessed_data(path: &Path, expected_seq_len: usize) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens.len() != expected_seq_len + 1 {
            println!(
                "⚠️  第{}行长度不匹配: {} (期望{})",
                i + 1,
                tokens.len(),
                expected_seq_len + 1
            );
            return Ok(false);
        }

        for token in &tokens {
            if token.parse::<usize>().is_err() {
                println!("⚠️  第{}行包含无效token: {}", i + 1, token);
                return Ok(false);
            }
        }

        if i >= 10 {
            break;
        }
    }

    Ok(true)
}

/// 清理缓存
pub fn clean_cache(cache_dir: &Path) -> Result<()> {
    if cache_dir.exists() {
        println!("🗑️  清理缓存: {}", cache_dir.display());
        let size = dir_size(cache_dir)?;
        fs::remove_dir_all(cache_dir)?;
        println!("   释放空间: {:.2} MB", size as f64 / 1_048_576.0);
    } else {
        println!("   缓存目录不存在: {}", cache_dir.display());
    }
    Ok(())
}

/// 计算目录大小
fn dir_size(path: &Path) -> Result<u64> {
    let mut total_size = 0u64;

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                total_size += entry.metadata()?.len();
            } else if path.is_dir() {
                total_size += dir_size(&path)?;
            }
        }
    }

    Ok(total_size)
}

// ============================================================================
// 基准测试
// ============================================================================

/// 基准测试结果统计
#[derive(Debug, Clone)]
struct BenchStats {
    runs: Vec<std::time::Duration>,
    config_name: String,
    num_params: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl BenchStats {
    fn avg_ms(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        self.runs.iter().map(|d| d.as_secs_f64() * 1000.0).sum::<f64>() / self.runs.len() as f64
    }

    fn min_ms(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        self.runs.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min)
    }

    fn max_ms(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        self.runs.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::NEG_INFINITY, f64::max)
    }

    fn p50_ms(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        let mut times: Vec<f64> = self.runs.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = times.len() / 2;
        if times.len() % 2 == 0 {
            (times[mid - 1] + times[mid]) / 2.0
        } else {
            times[mid]
        }
    }

    fn p99_ms(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        let mut times: Vec<f64> = self.runs.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (times.len() as f64 * 0.99).ceil() as usize;
        times[idx.min(times.len() - 1)]
    }

    fn tokens_per_sec(&self, seq_len: usize) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        let avg_us = self.avg_ms() * 1000.0;
        if avg_us <= 0.0 { return 0.0; }
        (seq_len as f64 / avg_us) * 1_000_000.0
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": self.config_name,
            "num_parameters": self.num_params,
            "vocab_size": self.vocab_size,
            "sequence_length": self.seq_len,
            "avg_ms": format!("{:.3}", self.avg_ms()),
            "min_ms": format!("{:.3}", self.min_ms()),
            "max_ms": format!("{:.3}", self.max_ms()),
            "p50_ms": format!("{:.3}", self.p50_ms()),
            "p99_ms": format!("{:.3}", self.p99_ms()),
            "tokens_per_sec": format!("{:.1}", self.tokens_per_sec(self.seq_len)),
            "runs": self.runs.len()
        })
    }
}

/// 收集系统硬件信息
fn get_hardware_info() -> serde_json::Value {
    let cpu_count = num_cpus::get();
    let mem_info = std::fs::read_to_string("/proc/meminfo")
        .map(|s| {
            let mut total_kb = 0u64;
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        total_kb = parts[1].parse::<u64>().unwrap_or(0);
                    }
                    break;
                }
            }
            total_kb
        })
        .unwrap_or(0);
    let total_mem_gb = mem_info as f64 / 1024.0 / 1024.0;

    serde_json::json!({
        "cpu_cores": cpu_count,
        "total_memory_gb": format!("{:.1}", total_mem_gb),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS
    })
}

/// 运行单次基准测试
fn run_single_bench(
    name: &str,
    config: &Config,
    vocab_size: usize,
    seq_len: usize,
    num_runs: usize,
) -> BenchStats {
    println!("   ──────────────────────────────────────────────");
    println!("   配置: {} | 层数:{} 维:{} 头:{} 词表:{} 序列:{}",
        name, config.model.num_layers, config.model.hidden_dim,
        config.model.num_heads, vocab_size, seq_len);

    let model_params = ModelParams::from_config(&config.model, vocab_size);
    let model = Transformer::new(model_params);
    let num_params = model.num_parameters();

    let input_ids: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();

    // 预热
    for _ in 0..3 {
        let _ = model.forward(&input_ids, false);
    }

    // 正式测试
    let mut times: Vec<std::time::Duration> = Vec::new();
    for _ in 0..num_runs {
        let start = std::time::Instant::now();
        let _ = model.forward(&input_ids, false);
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    BenchStats {
        runs: times,
        config_name: name.to_string(),
        num_params,
        vocab_size,
        seq_len,
    }
}

/// 运行反向传播基准
fn run_backward_bench(
    name: &str,
    config: &Config,
    vocab_size: usize,
    seq_len: usize,
    num_runs: usize,
) -> BenchStats {
    println!("   ──────────────────────────────────────────────");
    println!("   反向传播: {} | 层数:{} 维:{} 词表:{} 序列:{}",
        name, config.model.num_layers, config.model.hidden_dim, vocab_size, seq_len);

    let model_params = ModelParams::from_config(&config.model, vocab_size);
    let model = Transformer::new(model_params);
    let num_params = model.num_parameters();

    let input_ids: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();
    let targets: Vec<usize> = (0..seq_len).map(|i| (i + 1) % vocab_size).collect();

    // 前向传播获取 logits
    let logits = match model.forward(&input_ids, false) {
        Ok(l) => l,
        Err(e) => {
            println!("   ⚠️  前向传播失败: {}", e);
            return BenchStats {
                runs: vec![],
                config_name: name.to_string(),
                num_params,
                vocab_size,
                seq_len,
            };
        }
    };

    // 预热
    for _ in 0..3 {
        let _ = model.backward(&logits, &targets);
    }

    // 正式测试
    let mut times: Vec<std::time::Duration> = Vec::new();
    for _ in 0..num_runs {
        let logits = model.forward(&input_ids, false).unwrap();
        let start = std::time::Instant::now();
        let _ = model.backward(&logits, &targets);
        let elapsed = start.elapsed();
        times.push(elapsed);
    }

    BenchStats {
        runs: times,
        config_name: name.to_string(),
        num_params,
        vocab_size,
        seq_len,
    }
}

/// 运行基准测试
pub fn benchmark(_config: &Config, output_dir: &Path) -> Result<()> {
    println!("\n🔬 开始系统基准测试...\n");

    // 创建输出目录
    fs::create_dir_all(output_dir)?;

    // 硬件信息
    let hw_info = get_hardware_info();
    println!("💻 硬件信息:");
    println!("   CPU 核心: {}", hw_info["cpu_cores"]);
    println!("   总内存: {} GB", hw_info["total_memory_gb"]);
    println!("   架构: {} / {}", hw_info["arch"], hw_info["os"]);
    println!();

    // 测试配置列表
    let test_configs = vec![
        ("Tiny (MHA)", Config::tiny_preset(), 4096, 128),
        ("Small (SwiGLU+RoPE)", Config::small_preset(), 8192, 256),
        ("Base (GQA+BF16)", Config::base_preset(), 32768, 512),
    ];

    let num_runs = 10;
    let mut forward_results: Vec<BenchStats> = Vec::new();
    let mut backward_results: Vec<BenchStats> = Vec::new();

    let total_start = std::time::Instant::now();

    // 前向传播基准
    println!("📊 前向传播基准测试 ({} 次迭代/配置)", num_runs);
    for (name, bench_config, vocab_size, seq_len) in &test_configs {
        let stats = run_single_bench(name, bench_config, *vocab_size, *seq_len, num_runs);
        forward_results.push(stats);
    }
    println!();

    // 反向传播基准
    println!("📊 反向传播基准测试 ({} 次迭代/配置)", num_runs);
    for (name, bench_config, vocab_size, seq_len) in &test_configs {
        let stats = run_backward_bench(name, bench_config, *vocab_size, *seq_len, num_runs);
        backward_results.push(stats);
    }
    println!();

    let total_time = total_start.elapsed();

    // 输出前向传播结果
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    前向传播 性能结果                        ║");
    println!("╠══════════════════╦═══════╦═══════╦═══════╦═══════╦═══════╣");
    println!("║ 配置             ║ 平均  ║ 最小  ║ 最大  ║ P50   ║ P99   ║");
    println!("╠══════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╣");
    for stats in &forward_results {
        println!("║ {:<18} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║",
            &stats.config_name[..18.min(stats.config_name.len())],
            stats.avg_ms(), stats.min_ms(), stats.max_ms(),
            stats.p50_ms(), stats.p99_ms());
    }
    println!("╚══════════════════╩═══════╩═══════╩═══════╩═══════╩═══════╝");
    println!("   单位: ms/次  |  吞吐量单位: tokens/sec\n");

    // 输出反向传播结果
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   反向传播 性能结果                         ║");
    println!("╠══════════════════╦═══════╦═══════╦═══════╦═══════╦═══════╣");
    println!("║ 配置             ║ 平均  ║ 最小  ║ 最大  ║ P50   ║ P99   ║");
    println!("╠══════════════════╬═══════╬═══════╬═══════╬═══════╬═══════╣");
    for stats in &backward_results {
        println!("║ {:<18} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║ {:>5.2} ║",
            &stats.config_name[..18.min(stats.config_name.len())],
            stats.avg_ms(), stats.min_ms(), stats.max_ms(),
            stats.p50_ms(), stats.p99_ms());
    }
    println!("╚══════════════════╩═══════╩═══════╩═══════╩═══════╩═══════╝");
    println!();

    // 计算总参数和吞吐量
    let total_params: usize = forward_results.iter().map(|s| s.num_params).sum();
    println!("📈 汇总:");
    println!("   测试配置数: {}", forward_results.len());
    println!("   总参数量: {:.2}M", total_params as f64 / 1e6);
    println!("   总耗时: {:?}", total_time);

    // 各配置吞吐量
    for stats in &forward_results {
        let tps = stats.tokens_per_sec(stats.seq_len);
        println!("   {} 吞吐量: {:.1} tokens/sec", stats.config_name, tps);
    }
    println!();

    // 保存 JSON 结果
    let bench_data = serde_json::json!({
        "hardware": hw_info,
        "total_time_ms": total_time.as_secs_f64() * 1000.0,
        "forward_pass": forward_results.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
        "backward_pass": backward_results.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
        "summary": {
            "configs_tested": forward_results.len(),
            "total_parameters": total_params,
            "avg_forward_ms": forward_results.iter().map(|s| s.avg_ms()).collect::<Vec<_>>(),
            "avg_backward_ms": backward_results.iter().map(|s| s.avg_ms()).collect::<Vec<_>>()
        }
    });

    let bench_path = output_dir.join("benchmark_results.json");
    fs::write(&bench_path, bench_data.to_string())?;
    println!("✅ 基准测试结果已保存: {}", bench_path.display());

    // 同时保存可读文本报告
    let report_path = output_dir.join("benchmark_report.txt");
    let mut report = String::new();
    report.push_str("============================================================\n");
    report.push_str("           shlteLLM 基准测试报告\n");
    report.push_str("============================================================\n\n");
    report.push_str(&format!("硬件配置:\n"));
    report.push_str(&format!("  CPU 核心: {}\n", hw_info["cpu_cores"]));
    report.push_str(&format!("  总内存: {} GB\n", hw_info["total_memory_gb"]));
    report.push_str(&format!("  架构: {}/{}\n\n", hw_info["arch"], hw_info["os"]));
    report.push_str(&format!("测试耗时: {:?}\n\n", total_time));

    report.push_str("前向传播结果:\n");
    report.push_str("------------------------------------------------------------\n");
    for stats in &forward_results {
        report.push_str(&format!(
            "  {}: avg={:.2}ms min={:.2}ms max={:.2}ms p50={:.2}ms p99={:.2}ms | {:.1} tok/s | {:.2}M params\n",
            stats.config_name,
            stats.avg_ms(), stats.min_ms(), stats.max_ms(), stats.p50_ms(), stats.p99_ms(),
            stats.tokens_per_sec(stats.seq_len),
            stats.num_params as f64 / 1e6
        ));
    }

    report.push_str("\n反向传播结果:\n");
    report.push_str("------------------------------------------------------------\n");
    for stats in &backward_results {
        report.push_str(&format!(
            "  {}: avg={:.2}ms min={:.2}ms max={:.2}ms p50={:.2}ms p99={:.2}ms | {:.2}M params\n",
            stats.config_name,
            stats.avg_ms(), stats.min_ms(), stats.max_ms(), stats.p50_ms(), stats.p99_ms(),
            stats.num_params as f64 / 1e6
        ));
    }

    fs::write(&report_path, report)?;
    println!("📄 文本报告已保存: {}", report_path.display());

    Ok(())
}

// ============================================================================
// 模型导出与压缩
// ============================================================================

/// 导出模型为ONNX格式
pub fn export_onnx(_model: &Transformer, output_path: &Path) -> Result<()> {
    println!("📤 导出模型为ONNX格式: {}", output_path.display());

    #[cfg(feature = "onnx")]
    {
        // ONNX导出实现需要添加onnxruntime依赖
        // 这里仅作占位
        println!("   ONNX导出功能需要完成实现");
    }

    #[cfg(not(feature = "onnx"))]
    {
        println!("   (需要启用 'onnx' feature 并使用 --features onnx 编译)");
    }

    Ok(())
}

/// 模型量化压缩（8位量化）
pub fn compress_model(model: &mut Transformer, bits: usize) -> Result<()> {
    println!("🗜️  压缩模型 ({}位量化)...", bits);

    let before_params = model.num_parameters();
    let quantize_factor = (1 << bits) - 1;

    // 对权重进行量化
    let quantize_weight = |w: &mut [Vec<f32>]| {
        for row in w.iter_mut() {
            for val in row.iter_mut() {
                let quantized = (*val * quantize_factor as f32).round();
                *val = quantized / quantize_factor as f32;
            }
        }
    };

    quantize_weight(&mut model.embedding);

    for layer in model.layers.iter_mut() {
        quantize_weight(&mut layer.attention.q_proj);
        quantize_weight(&mut layer.attention.k_proj);
        quantize_weight(&mut layer.attention.v_proj);
        quantize_weight(&mut layer.attention.o_proj);
        quantize_weight(&mut layer.feed_forward.up_proj);
        quantize_weight(&mut layer.feed_forward.down_proj);
        if let Some(ref mut gate) = layer.feed_forward.gate_proj {
            quantize_weight(gate);
        }
    }

    if let Some(ref mut lm_head) = model.lm_head {
        quantize_weight(lm_head);
    }

    let after_params = model.num_parameters();
    println!("   压缩前: {:.2}M 参数", before_params as f64 / 1e6);
    println!("   压缩后: {:.2}M 参数", after_params as f64 / 1e6);
    println!(
        "   压缩比: {:.1}%",
        (1.0 - after_params as f64 / before_params as f64) * 100.0
    );

    Ok(())
}