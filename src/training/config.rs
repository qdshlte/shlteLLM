//! ============================================================================
//! 配置管理模块
//! ============================================================================
//!
//! 本模块定义了整个训练框架的配置结构，支持TOML格式的配置文件读写。
//! 包含训练配置、模型配置、数据集配置、硬件配置等多个子配置项。
//! 提供预设配置（tiny/small/base）以及配置验证功能。
//!
//! 修复内容（P2-2）：
//! - 添加更完善的配置验证
//! - 学习率调度器参数合理性检查
//! - 优化器参数范围检查
//! - 硬件配置与实际硬件匹配性检查
//! - 数据路径存在性检查
//!
//! ============================================================================

// ML 领域常见缩写（BPE, GELU, GEGLU, MHA, GQA, MQA, SGD, LAMB, CPU, CUDA, MPS）
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::extra_unused_lifetimes)]

// ============================================================================
// 标准库导入
// ============================================================================

use crate::error::{Result, TrainError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;
use std::collections::HashSet;

// ============================================================================
// 主配置结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dataset: DatasetConfig,
    pub preprocessing: PreprocessingConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub tokenizer: TokenizerConfig,
    pub hardware: HardwareConfig,
    pub logging: LoggingConfig,
}

// ============================================================================
// 数据集配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// 数据集混合配置: [("dataset_name", percentage), ...]
    pub mix: Vec<DatasetSource>,
    /// 下载源
    pub download_source: DownloadSource,
    /// 数据集大小（GB）
    pub size_gb: f64,
    /// 分片数量
    pub num_shards: usize,
    /// 本地数据路径
    pub local_path: Option<String>,
    /// 自定义URL
    pub custom_url: Option<String>,
    /// 缓存目录
    pub cache_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSource {
    pub name: String,
    pub percentage: f64, // 0.0 - 1.0
    pub split: Option<String>,
    pub subset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DownloadSource {
    HuggingFace,
    Mirror { url: String },
    CustomUrl { url: String },
    Local,
}

// ============================================================================
// 预处理配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// 分词算法
    pub tokenization_algorithm: TokenizationAlgorithm,
    /// 词表大小
    pub vocab_size: usize,
    /// 最小词频
    pub min_frequency: usize,
    /// 最大序列长度
    pub max_sequence_length: usize,
    /// 特殊token
    pub special_tokens: SpecialTokens,
    /// 是否小写
    pub lowercase: bool,
    /// 是否移除重音符号
    pub remove_accents: bool,
    /// 字节级编码
    pub byte_level: bool,
    /// 是否添加前缀空格
    pub add_prefix_space: bool,
    /// 字节回退
    pub byte_fallback: bool,
}

// ============================================================================
// 分词算法枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizationAlgorithm {
    BPE,
    WordPiece,
    Unigram,
    SentencePiece,
}

// ============================================================================
// 特殊Token配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,
    pub unk_token: String,
    pub mask_token: Option<String>,
    pub sep_token: Option<String>,
    pub cls_token: Option<String>,
    pub additional_tokens: Vec<String>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad_token: "<|pad|>".to_string(),
            bos_token: "<|im_start|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            unk_token: "<|unk|>".to_string(),
            mask_token: Some("<|mask|>".to_string()),
            sep_token: Some("<|sep|>".to_string()),
            cls_token: Some("<|cls|>".to_string()),
            additional_tokens: vec![
                "<|im_end|>".to_string(),
                "<|user|>".to_string(),
                "<|assistant|>".to_string(),
                "<|system|>".to_string(),
            ],
        }
    }
}

// ============================================================================
// 模型配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 层数
    pub num_layers: usize,
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 激活函数
    pub activation: ActivationFunction,
    /// 位置编码
    pub position_encoding: PositionEncoding,
    /// 归一化方式
    pub normalization: NormalizationType,
    /// 注意力机制
    pub attention: AttentionType,
    /// QKV偏置
    pub use_qkv_bias: bool,
    /// MLP偏置
    pub use_mlp_bias: bool,
    /// 词表嵌入与输出头共享权重
    pub tied_embedding: bool,
    /// Dropout率
    pub dropout: f64,
    /// 随机深度
    pub stochastic_depth: Option<f64>,
    /// 中间层维度（FFN）
    pub intermediate_dim: Option<usize>,
    /// 滑动窗口大小（用于滑动窗口注意力）
    pub sliding_window: Option<usize>,
    /// GQA组数
    pub num_key_value_heads: Option<usize>,
    /// RoPE基础频率
    pub rope_theta: Option<f64>,
    /// RMSNorm epsilon
    pub rms_norm_eps: Option<f64>,
    /// 最大位置编码长度
    pub max_position_embeddings: usize,
    /// 词表大小（从tokenizer获取，此处可覆盖）
    pub vocab_size_override: Option<usize>,
}

// ============================================================================
// 激活函数枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationFunction {
    SwiGLU,
    GELU,
    ReLU,
    SiLU,
    GEGLU,
}

// ============================================================================
// 位置编码枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionEncoding {
    RoPE,
    ALiBi,
    NoPE,
    Learned,
    Sinusoidal,
}

// ============================================================================
// 归一化类型枚举
// ============================================================================

/// 归一化类型
///
/// 各变体说明：
/// - `Rms`: RMS归一化 (RMS Layer Normalization)
/// - `Layer`: 层归一化 (Layer Normalization)
/// - `PreLayer`: 前置层归一化 (Pre-Layer Normalization), 在子层前应用归一化
/// - `PostLayer`: 后置层归一化 (Post-Layer Normalization), 在子层后应用归一化
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormalizationType {
    Rms,
    Layer,
    PreLayer,
    PostLayer,
}

// ============================================================================
// 注意力类型枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttentionType {
    MHA,
    GQA { num_groups: usize },
    MQA,
    SlidingWindow { window_size: usize },
    GQAWithSlidingWindow {
        num_groups: usize,
        window_size: usize,
    },
    FlashAttention,
}

// ============================================================================
// 训练配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// 学习率
    pub learning_rate: f64,
    /// 最小学习率
    pub min_learning_rate: Option<f64>,
    /// 学习率调度器
    pub lr_scheduler: LRScheduler,
    /// 权重衰减
    pub weight_decay: f64,
    /// 梯度裁剪
    pub grad_clip: f64,
    /// 批大小
    pub batch_size: usize,
    /// 微批大小（梯度累积）
    pub micro_batch_size: Option<usize>,
    /// 训练步数
    pub num_steps: usize,
    /// 热身步数
    pub warmup_steps: usize,
    /// 序列长度
    pub sequence_length: usize,
    /// 评估间隔
    pub eval_interval: usize,
    /// 保存间隔
    pub save_interval: usize,
    /// 日志间隔
    pub log_interval: usize,
    /// 优化器
    pub optimizer: OptimizerType,
    /// 混合精度
    pub mixed_precision: Option<MixedPrecision>,
    /// 梯度累积步数
    pub gradient_accumulation_steps: usize,
    /// EMA衰减
    pub ema_decay: Option<f64>,
    /// 最大检查点数量
    pub max_checkpoints: usize,
}

// ============================================================================
// 学习率调度器枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduler {
    Linear,
    Cosine {
        min_lr: f64,
    },
    CosineWithRestarts {
        min_lr: f64,
        restart_interval: usize,
    },
    Constant,
    OneCycle {
        max_lr: f64,
        pct_start: f64,
    },
}

// ============================================================================
// 优化器类型枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    SGD {
        momentum: f64,
    },
    LAMB {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
}

// ============================================================================
// 混合精度枚举
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum MixedPrecision {
    FP16,
    BF16,
    FP8,
}

// ============================================================================
// 硬件配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// 设备类型
    pub device: Device,
    /// GPU ID列表
    pub gpu_ids: Vec<usize>,
    /// 数据加载线程数
    pub num_workers: usize,
    /// 是否使用TF32
    pub use_tf32: bool,
    /// 内存预分配
    pub memory_prealloc: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Device {
    CPU,
    CUDA,
    MPS,
    ROCm,
    Auto,
}

// ============================================================================
// 分词器配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub algorithm: TokenizationAlgorithm,
    pub vocab_size: usize,
    pub special_tokens: SpecialTokens,
    pub normalization: bool,
    pub add_prefix_space: bool,
}

// ============================================================================
// 日志配置
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// 日志级别
    pub level: String,
    /// WandB项目名
    pub wandb_project: Option<String>,
    /// WandB实体
    pub wandb_entity: Option<String>,
    /// Tensorboard目录
    pub tensorboard_dir: Option<String>,
    /// CSV日志路径
    pub csv_log_path: Option<String>,
}

// ============================================================================
// 配置验证结果
// ============================================================================

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub suggestions: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        ValidationResult {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            suggestions: Vec::new(),
        }
    }
    
    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }
    
    pub fn add_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
        self.is_valid = false;
    }
    
    pub fn add_suggestion(&mut self, suggestion: &str) {
        self.suggestions.push(suggestion.to_string());
    }
    
    pub fn print(&self) {
        if !self.warnings.is_empty() {
            println!("\n⚠️ 警告:");
            for warning in &self.warnings {
                println!("   - {}", warning);
            }
        }
        
        if !self.errors.is_empty() {
            println!("\n❌ 错误:");
            for error in &self.errors {
                println!("   - {}", error);
            }
        }
        
        if !self.suggestions.is_empty() {
            println!("\n💡 建议:");
            for suggestion in &self.suggestions {
                println!("   - {}", suggestion);
            }
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Config 实现
// ============================================================================

impl Config {
    /// 从文件加载配置
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            TrainError::Config(format!("无法读取配置文件 {}: {}", path.display(), e))
        })?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| TrainError::Config(format!("配置文件解析错误: {}", e)))?;

        Ok(config)
    }

    /// 生成默认配置文件
    pub fn generate_default(path: &Path) -> Result<()> {
        if path.exists() {
            println!("⚠️  配置文件已存在: {}", path.display());
            println!("   如需重新生成，请删除现有文件或使用不同的路径。");
            return Ok(());
        }

        let config = Config::tiny_preset();
        let content = toml::to_string_pretty(&config)
            .map_err(|e| TrainError::Config(format!("无法序列化默认配置: {}", e)))?;

        fs::write(path, content)
            .map_err(|e| TrainError::Config(format!("无法写入默认配置文件: {}", e)))?;

        println!("✅ 已生成默认配置文件: {}", path.display());
        Ok(())
    }

    /// 验证配置合法性（增强版）
    pub fn validate(&self) -> Result<()> {
        let result = self.validate_detailed();
        
        result.print();
        
        if !result.is_valid {
            return Err(TrainError::Validation(format!(
                "配置验证失败，发现 {} 个错误",
                result.errors.len()
            )));
        }
        
        Ok(())
    }
    
    /// 详细验证（返回结构化结果）
    pub fn validate_detailed(&self) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // ====================================================================
        // 验证数据集配置
        // ====================================================================
        self.validate_dataset_config(&mut result);
        
        // ====================================================================
        // 验证模型配置
        // ====================================================================
        self.validate_model_config(&mut result);
        
        // ====================================================================
        // 验证训练配置
        // ====================================================================
        self.validate_training_config(&mut result);
        
        // ====================================================================
        // 验证硬件配置
        // ====================================================================
        self.validate_hardware_config(&mut result);
        
        // ====================================================================
        // 验证日志配置
        // ====================================================================
        self.validate_logging_config(&mut result);
        
        result
    }
    
    // ========================================================================
    // 数据集配置验证
    // ========================================================================
    
    fn validate_dataset_config(&self, result: &mut ValidationResult) {
        // 验证数据集百分比总和
        let total_percentage: f64 = self.dataset.mix.iter().map(|ds| ds.percentage).sum();
        
        if (total_percentage - 1.0).abs() > 1e-6 {
            result.add_error(&format!(
                "数据集百分比总和必须为100%，当前为{}%",
                total_percentage * 100.0
            ));
        }
        
        if self.dataset.mix.is_empty() {
            result.add_error("至少需要一个数据集");
        }
        
        // 验证数据集名称
        let mut seen_names = HashSet::new();
        for source in &self.dataset.mix {
            if source.name.is_empty() {
                result.add_error("数据集名称不能为空");
            }
            if seen_names.contains(&source.name) {
                result.add_warning(&format!("数据集名称 '{}' 重复出现", source.name));
            }
            seen_names.insert(&source.name);
            
            if source.percentage < 0.0 || source.percentage > 1.0 {
                result.add_error(&format!(
                    "数据集 '{}' 的百分比必须在[0,1]范围内，当前为{}",
                    source.name, source.percentage
                ));
            }
        }
        
        // 验证数据集大小
        if self.dataset.size_gb <= 0.0 {
            result.add_warning(&format!("数据集大小为 {} GB，可能过小", self.dataset.size_gb));
        }
        
        if self.dataset.size_gb > 1000.0 {
            result.add_warning(&format!(
                "数据集大小为 {} GB，可能需要大量磁盘空间",
                self.dataset.size_gb
            ));
        }
        
        // 验证分片数量
        if self.dataset.num_shards == 0 {
            result.add_error("分片数量不能为0");
        }
        
        // 验证本地路径
        if let DownloadSource::Local = &self.dataset.download_source {
            if let Some(local_path) = &self.dataset.local_path {
                let path = Path::new(local_path);
                if !path.exists() {
                    result.add_error(&format!("本地数据路径不存在: {}", local_path));
                }
            } else {
                result.add_error("使用 Local 下载源时必须指定 local_path");
            }
        }
        
        // 验证自定义URL
        if let DownloadSource::CustomUrl { url } = &self.dataset.download_source {
            if !url.starts_with("http://") && !url.starts_with("https://") {
                result.add_warning(&format!("自定义URL可能无效: {}", url));
            }
        }
        
        // 验证缓存目录
        let cache_path = Path::new(&self.dataset.cache_dir);
        if cache_path.exists() && !cache_path.is_dir() {
            result.add_error(&format!("缓存目录路径存在但不是目录: {}", self.dataset.cache_dir));
        }
    }
    
    // ========================================================================
    // 模型配置验证
    // ========================================================================
    
    fn validate_model_config(&self, result: &mut ValidationResult) {
        // 基础参数验证
        if self.model.num_layers == 0 {
            result.add_error("层数不能为0");
        } else if self.model.num_layers > 100 {
            result.add_warning(&format!("层数 {} 可能过大，需要大量GPU内存", self.model.num_layers));
        }
        
        if self.model.hidden_dim == 0 {
            result.add_error("隐藏层维度不能为0");
        } else if self.model.hidden_dim > 16384 {
            result.add_warning(&format!("隐藏层维度 {} 可能过大", self.model.hidden_dim));
        }
        
        if self.model.num_heads == 0 {
            result.add_error("注意力头数不能为0");
        }
        
        // hidden_dim 必须能被 num_heads 整除
        if self.model.hidden_dim > 0 && self.model.num_heads > 0
            && !self.model.hidden_dim.is_multiple_of(self.model.num_heads) {
                result.add_error(&format!(
                    "隐藏层维度({})必须能被注意力头数({})整除",
                    self.model.hidden_dim, self.model.num_heads
                ));
                result.add_suggestion(&format!(
                    "建议: 设置 hidden_dim = {} 或 num_heads = {}",
                    self.model.hidden_dim.next_multiple_of(self.model.num_heads),
                    self.model.hidden_dim
                ));
            }
        
        // 验证GQA配置
        match &self.model.attention {
            AttentionType::GQA { num_groups }
            | AttentionType::GQAWithSlidingWindow {
                num_groups,
                window_size: _,
            } => {
                if *num_groups == 0 {
                    result.add_error("GQA组数不能为0");
                }
                if self.model.num_heads > 0 && !self.model.num_heads.is_multiple_of(*num_groups) {
                    result.add_error(&format!(
                        "GQA组数({})必须能整除注意力头数({})",
                        num_groups, self.model.num_heads
                    ));
                }
                if let Some(kv_heads) = self.model.num_key_value_heads {
                    if kv_heads != *num_groups {
                        result.add_warning(&format!(
                            "num_key_value_heads({}) 与 GQA num_groups({}) 不一致",
                            kv_heads, num_groups
                        ));
                    }
                }
            }
            _ => {}
        }
        
        // 验证dropout
        if self.model.dropout < 0.0 || self.model.dropout >= 1.0 {
            result.add_error(&format!(
                "Dropout率必须在[0.0, 1.0)范围内，当前为{}",
                self.model.dropout
            ));
        } else if self.model.dropout > 0.5 {
            result.add_warning(&format!("Dropout率 {} 较高，可能影响收敛", self.model.dropout));
        }
        
        // 验证随机深度
        if let Some(sd) = self.model.stochastic_depth {
            if !(0.0..1.0).contains(&sd) {
                result.add_error(&format!(
                    "随机深度概率必须在[0.0, 1.0)范围内，当前为{}",
                    sd
                ));
            }
        }
        
        // 验证滑动窗口
        match &self.model.attention {
            AttentionType::SlidingWindow { window_size }
            | AttentionType::GQAWithSlidingWindow {
                num_groups: _,
                window_size,
            } => {
                if *window_size == 0 {
                    result.add_error("滑动窗口大小不能为0");
                }
                if *window_size > self.model.max_position_embeddings {
                    result.add_error(&format!(
                        "滑动窗口大小({})不能超过最大位置编码长度({})",
                        window_size, self.model.max_position_embeddings
                    ));
                }
            }
            _ => {}
        }
        
        // 验证位置编码
        match self.model.position_encoding {
            PositionEncoding::RoPE => {
                if let Some(theta) = self.model.rope_theta {
                    if theta <= 0.0 {
                        result.add_error(&format!("RoPE theta 必须大于0，当前为{}", theta));
                    } else if !(1000.0..=100000.0).contains(&theta) {
                        result.add_warning(&format!("RoPE theta = {} 可能不是最优值，建议使用10000", theta));
                    }
                }
            }
            PositionEncoding::ALiBi => {
                result.add_suggestion("ALiBi 位置编码通常与滑动窗口注意力配合使用");
            }
            _ => {}
        }
        
        // 验证归一化
        if self.model.normalization == NormalizationType::Rms {
            if let Some(eps) = self.model.rms_norm_eps {
                if eps <= 0.0 || eps >= 1.0 {
                    result.add_error(&format!("RMSNorm eps 必须在(0,1)范围内，当前为{}", eps));
                }
            }
        }
        
        // 验证最大位置编码长度
        if self.model.max_position_embeddings == 0 {
            result.add_error("最大位置编码长度不能为0");
        } else if self.model.max_position_embeddings > 128000 {
            result.add_warning(&format!(
                "最大位置编码长度 {} 可能需要大量内存",
                self.model.max_position_embeddings
            ));
        }
    }
    
    // ========================================================================
    // 训练配置验证
    // ========================================================================
    
    fn validate_training_config(&self, result: &mut ValidationResult) {
        // 学习率验证
        if self.training.learning_rate <= 0.0 {
            result.add_error(&format!("学习率必须大于0，当前为{}", self.training.learning_rate));
        } else if self.training.learning_rate > 1.0 {
            result.add_warning(&format!("学习率 {} 可能过大，建议使用0.001~0.0001范围", self.training.learning_rate));
        } else if self.training.learning_rate < 1e-7 {
            result.add_warning(&format!("学习率 {} 可能过小，训练会很慢", self.training.learning_rate));
        }
        
        // 最小学习率验证
        if let Some(min_lr) = self.training.min_learning_rate {
            if min_lr < 0.0 {
                result.add_error(&format!("最小学习率不能为负数，当前为{}", min_lr));
            } else if min_lr > self.training.learning_rate {
                result.add_error(&format!(
                    "最小学习率({})不能大于学习率({})",
                    min_lr, self.training.learning_rate
                ));
            }
        }
        
        // 学习率调度器参数验证
        match &self.training.lr_scheduler {
            LRScheduler::Cosine { min_lr } => {
                if *min_lr < 0.0 {
                    result.add_error(&format!("余弦衰减的最小学习率不能为负数，当前为{}", min_lr));
                }
            }
            LRScheduler::CosineWithRestarts { min_lr, restart_interval } => {
                if *min_lr < 0.0 {
                    result.add_error(&format!("余弦重启的最小学习率不能为负数，当前为{}", min_lr));
                }
                if *restart_interval == 0 {
                    result.add_error("余弦重启的重启间隔不能为0");
                }
            }
            LRScheduler::OneCycle { max_lr, pct_start } => {
                if *max_lr < self.training.learning_rate {
                    result.add_error(&format!(
                        "OneCycle的最大学习率({})不能小于基础学习率({})",
                        max_lr, self.training.learning_rate
                    ));
                }
                if *pct_start <= 0.0 || *pct_start >= 1.0 {
                    result.add_error(&format!(
                        "OneCycle的上升比例必须在(0,1)范围内，当前为{}",
                        pct_start
                    ));
                }
            }
            _ => {}
        }
        
        // 优化器参数验证
        match &self.training.optimizer {
            OptimizerType::AdamW { beta1, beta2, epsilon }
            | OptimizerType::Adam { beta1, beta2, epsilon }
            | OptimizerType::LAMB { beta1, beta2, epsilon } => {
                if !(0.0..1.0).contains(beta1) {
                    result.add_error(&format!("Adam beta1 必须在[0,1)范围内，当前为{}", beta1));
                }
                if !(0.0..1.0).contains(beta2) {
                    result.add_error(&format!("Adam beta2 必须在[0,1)范围内，当前为{}", beta2));
                }
                if *epsilon <= 0.0 || *epsilon >= 1.0 {
                    result.add_warning(&format!("Adam epsilon={} 可能不是最优值", epsilon));
                }
            }
            OptimizerType::SGD { momentum } => {
                if *momentum < 0.0 || *momentum >= 1.0 {
                    result.add_error(&format!("SGD动量必须在[0,1)范围内，当前为{}", momentum));
                }
            }
        }
        
        // 权重衰减验证
        if self.training.weight_decay < 0.0 {
            result.add_error(&format!("权重衰减不能为负数，当前为{}", self.training.weight_decay));
        } else if self.training.weight_decay > 0.1 {
            result.add_warning(&format!("权重衰减 {} 较大，可能导致欠拟合", self.training.weight_decay));
        }
        
        // 梯度裁剪验证
        if self.training.grad_clip < 0.0 {
            result.add_error(&format!("梯度裁剪阈值不能为负数，当前为{}", self.training.grad_clip));
        } else if self.training.grad_clip == 0.0 {
            result.add_warning("梯度裁剪阈值为0，将禁用梯度裁剪");
        }
        
        // 批大小验证
        if self.training.batch_size == 0 {
            result.add_error("批大小不能为0");
        } else if self.training.batch_size < 4 {
            result.add_warning("批大小过小，可能导致训练不稳定");
        }
        
        // 微批大小验证
        if let Some(micro_batch) = self.training.micro_batch_size {
            if micro_batch == 0 {
                result.add_error("微批大小不能为0");
            } else if micro_batch > self.training.batch_size {
                result.add_warning(&format!(
                    "微批大小({})大于批大小({})，将使用批大小作为微批大小",
                    micro_batch, self.training.batch_size
                ));
            }
        }
        
        // 训练步数验证
        if self.training.num_steps == 0 {
            result.add_error("训练步数必须大于0");
        }
        
        // 热身步数验证
        if self.training.warmup_steps > self.training.num_steps {
            result.add_error(&format!(
                "热身步数({})不能超过总训练步数({})",
                self.training.warmup_steps, self.training.num_steps
            ));
        } else if self.training.warmup_steps > self.training.num_steps / 2 {
            result.add_warning(&format!(
                "热身步数({})超过总步数的一半，可能影响收敛",
                self.training.warmup_steps
            ));
        }
        
        // 序列长度验证
        if self.training.sequence_length == 0 {
            result.add_error("序列长度不能为0");
        } else if self.training.sequence_length > self.model.max_position_embeddings {
            result.add_error(&format!(
                "序列长度({})不能超过最大位置编码长度({})",
                self.training.sequence_length, self.model.max_position_embeddings
            ));
        }
        
        // 评估间隔验证
        if self.training.eval_interval > 0 && self.training.eval_interval < self.training.log_interval {
            result.add_warning("评估间隔小于日志间隔，可能影响性能");
        }
        
        // 保存间隔验证
        if self.training.save_interval > 0 && self.training.save_interval < self.training.eval_interval {
            result.add_warning("保存间隔小于评估间隔，可能产生过多检查点");
        }
        
        // 梯度累积步数验证
        if self.training.gradient_accumulation_steps == 0 {
            result.add_error("梯度累积步数不能为0");
        } else if self.training.gradient_accumulation_steps > 32 {
            result.add_warning(&format!(
                "梯度累积步数 {} 较大，可能导致训练变慢",
                self.training.gradient_accumulation_steps
            ));
        }
        
        // EMA衰减验证
        if let Some(ema_decay) = self.training.ema_decay {
            if !(0.9..1.0).contains(&ema_decay) {
                result.add_warning(&format!(
                    "EMA衰减 {} 可能不是最优值，建议使用0.99~0.999",
                    ema_decay
                ));
            }
        }
        
        // 混合精度验证
        if let Some(mp) = &self.training.mixed_precision {
            match mp {
                MixedPrecision::BF16 => {
                    // BF16需要硬件支持
                    result.add_suggestion("BF16混合精度需要Ampere或更新的GPU，或支持AVX512的CPU");
                }
                MixedPrecision::FP8 => {
                    result.add_warning("FP8混合精度需要H100或更新的GPU，请确保硬件支持");
                }
                MixedPrecision::FP16 => {
                    // FP16是通用选项
                }
            }
        }
        
        // 最大检查点数验证
        if self.training.max_checkpoints == 0 {
            result.add_warning("max_checkpoints=0 将不保存任何检查点");
        }
    }
    
    // ========================================================================
    // 硬件配置验证
    // ========================================================================
    
    fn validate_hardware_config(&self, result: &mut ValidationResult) {
        // 验证GPU ID
        if !self.hardware.gpu_ids.is_empty() {
            let max_gpu_id = self.hardware.gpu_ids.iter().max().copied().unwrap_or(0);
            if max_gpu_id >= 8 {
                result.add_warning(&format!("GPU ID {} 可能超出实际GPU数量", max_gpu_id));
            }
            
            // 检查重复的GPU ID
            let unique_ids: HashSet<_> = self.hardware.gpu_ids.iter().collect();
            if unique_ids.len() != self.hardware.gpu_ids.len() {
                result.add_warning("GPU ID列表中存在重复");
            }
        }
        
        // 验证数据加载线程数
        let cpu_count = num_cpus::get();
        if self.hardware.num_workers > cpu_count * 2 {
            result.add_warning(&format!(
                "num_workers={} 超过CPU核心数的2倍({})，可能影响性能",
                self.hardware.num_workers,
                cpu_count * 2
            ));
        } else if self.hardware.num_workers == 0 {
            result.add_warning("num_workers=0 将使用主线程加载数据");
        }
        
        // 验证设备兼容性
        match self.hardware.device {
            Device::CUDA => {
                #[cfg(not(target_os = "windows"))]
                {
                    // 检查CUDA是否可用（运行时检测）
                    result.add_suggestion("确保已安装CUDA驱动和CUDA工具包");
                }
            }
            Device::MPS => {
                #[cfg(not(target_os = "macos"))]
                {
                    result.add_warning("MPS仅在macOS上可用");
                }
            }
            Device::ROCm => {
                result.add_suggestion("ROCm支持需要特定硬件和驱动");
            }
            Device::Auto => {
                // 自动检测是安全的
            }
            Device::CPU => {
                // CPU总是可用
            }
        }
        
        // TF32建议
        if self.hardware.use_tf32 {
            result.add_suggestion("TF32需要Ampere或更新的GPU");
        }
    }
    
    // ========================================================================
    // 日志配置验证
    // ========================================================================
    
    fn validate_logging_config(&self, result: &mut ValidationResult) {
        // 验证日志级别
        let valid_levels = ["debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.to_lowercase().as_str()) {
            result.add_warning(&format!(
                "日志级别 '{}' 可能无效，有效值: {:?}",
                self.logging.level, valid_levels
            ));
        }
        
        // 验证TensorBoard目录
        if let Some(tb_dir) = &self.logging.tensorboard_dir {
            let path = Path::new(tb_dir);
            if path.exists() && !path.is_dir() {
                result.add_error(&format!("TensorBoard路径存在但不是目录: {}", tb_dir));
            }
        }
        
        // 验证CSV日志路径
        if let Some(csv_path) = &self.logging.csv_log_path {
            let path = Path::new(csv_path);
            if path.exists() && path.is_dir() {
                result.add_error(&format!("CSV日志路径是目录而不是文件: {}", csv_path));
            }
        }
    }

    // ========================================================================
    // 预设配置
    // ========================================================================

    /// 获取预设配置列表
    pub fn presets() -> Vec<(&'static str, Config)> {
        vec![
            ("tiny", Config::tiny_preset()),
            ("small", Config::small_preset()),
            ("base", Config::base_preset()),
        ]
    }

    /// 保存预设配置到目录
    pub fn save_presets(dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        for (name, preset) in Config::presets() {
            let path = dir.join(format!("{}.toml", name));
            let content = toml::to_string_pretty(&preset)?;
            fs::write(&path, content)?;
            println!("✅ 已保存预设配置: {}", path.display());
        }

        Ok(())
    }
    
    /// 获取推荐的配置（根据硬件）
    pub fn recommend_for_hardware() -> Config {
        let cpu_count = num_cpus::get();
        let memory = sysinfo::System::new_all();
        
        let mut config = Config::tiny_preset();
        
        // 根据CPU核心数调整数据加载线程
        if cpu_count >= 16 {
            config.hardware.num_workers = 8;
        } else if cpu_count >= 8 {
            config.hardware.num_workers = 4;
        } else {
            config.hardware.num_workers = 2;
        }
        
        // 根据可用内存调整模型大小
        let available_memory = memory.available_memory();
        if available_memory > 16_000_000_000 {  // > 16GB
            config = Config::base_preset();
        } else if available_memory > 8_000_000_000 {  // > 8GB
            config = Config::small_preset();
        }
        // 否则使用tiny
        
        config
    }

    // ========================================================================
    // Tiny预设配置
    // ========================================================================

    pub fn tiny_preset() -> Config {
        Config {
            dataset: DatasetConfig {
                mix: vec![DatasetSource {
                    name: "FineWeb-Edu".to_string(),
                    percentage: 1.0,
                    split: Some("train".to_string()),
                    subset: None,
                }],
                download_source: DownloadSource::Mirror {
                    url: "https://hf-mirror.com".to_string(),
                },
                size_gb: 0.1,
                num_shards: 1,
                local_path: None,
                custom_url: None,
                cache_dir: "./cache".to_string(),
            },
            preprocessing: PreprocessingConfig {
                tokenization_algorithm: TokenizationAlgorithm::BPE,
                vocab_size: 4096,
                min_frequency: 2,
                max_sequence_length: 128,
                special_tokens: SpecialTokens::default(),
                lowercase: true,
                remove_accents: true,
                byte_level: true,
                add_prefix_space: false,
                byte_fallback: false,
            },
            model: ModelConfig {
                num_layers: 4,
                hidden_dim: 256,
                num_heads: 4,
                activation: ActivationFunction::GELU,
                position_encoding: PositionEncoding::NoPE,
                normalization: NormalizationType::Layer,
                attention: AttentionType::MHA,
                use_qkv_bias: true,
                use_mlp_bias: true,
                tied_embedding: true,
                dropout: 0.1,
                stochastic_depth: None,
                intermediate_dim: Some(1024),
                sliding_window: None,
                num_key_value_heads: None,
                rope_theta: None,
                rms_norm_eps: None,
                max_position_embeddings: 128,
                vocab_size_override: None,
            },
            training: TrainingConfig {
                learning_rate: 1e-3,
                min_learning_rate: Some(1e-5),
                lr_scheduler: LRScheduler::Cosine { min_lr: 1e-5 },
                weight_decay: 0.01,
                grad_clip: 1.0,
                batch_size: 8,
                micro_batch_size: None,
                num_steps: 1000,
                warmup_steps: 100,
                sequence_length: 128,
                eval_interval: 100,
                save_interval: 500,
                log_interval: 10,
                optimizer: OptimizerType::AdamW {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
                mixed_precision: None,
                gradient_accumulation_steps: 1,
                ema_decay: None,
                max_checkpoints: 5,
            },
            tokenizer: TokenizerConfig {
                algorithm: TokenizationAlgorithm::BPE,
                vocab_size: 4096,
                special_tokens: SpecialTokens::default(),
                normalization: true,
                add_prefix_space: false,
            },
            hardware: HardwareConfig {
                device: Device::Auto,
                gpu_ids: vec![0],
                num_workers: 2,
                use_tf32: true,
                memory_prealloc: false,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                wandb_project: None,
                wandb_entity: None,
                tensorboard_dir: Some("./logs/tensorboard".to_string()),
                csv_log_path: Some("./logs/training.csv".to_string()),
            },
        }
    }

    // ========================================================================
    // Small预设配置
    // ========================================================================

    pub fn small_preset() -> Config {
        let mut config = Config::tiny_preset();
        config.model.num_layers = 8;
        config.model.hidden_dim = 512;
        config.model.num_heads = 8;
        config.model.intermediate_dim = Some(2048);
        config.preprocessing.vocab_size = 8192;
        config.tokenizer.vocab_size = 8192;
        config.training.learning_rate = 5e-4;
        config.training.batch_size = 16;
        config.preprocessing.max_sequence_length = 256;
        config.model.max_position_embeddings = 256;
        config.training.sequence_length = 256;
        config.training.num_steps = 5000;
        config.model.activation = ActivationFunction::SwiGLU;
        config.model.position_encoding = PositionEncoding::RoPE;
        config.model.normalization = NormalizationType::Rms;
        config.model.rope_theta = Some(10000.0);
        config.model.rms_norm_eps = Some(1e-6);
        config
    }

    // ========================================================================
    // Base预设配置
    // ========================================================================

    pub fn base_preset() -> Config {
        let mut config = Config::small_preset();
        config.model.num_layers = 12;
        config.model.hidden_dim = 768;
        config.model.num_heads = 12;
        config.model.intermediate_dim = Some(3072);
        config.preprocessing.vocab_size = 32768;
        config.tokenizer.vocab_size = 32768;
        config.training.learning_rate = 3e-4;
        config.training.batch_size = 32;
        config.preprocessing.max_sequence_length = 512;
        config.model.max_position_embeddings = 512;
        config.training.sequence_length = 512;
        config.training.num_steps = 10000;
        config.model.attention = AttentionType::GQA { num_groups: 4 };
        config.model.tied_embedding = false;
        config.training.mixed_precision = Some(MixedPrecision::BF16);
        config.training.ema_decay = Some(0.999);
        config
    }
}

// ============================================================================
// DownloadSource 转换为字符串
// ============================================================================

impl fmt::Display for DownloadSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadSource::HuggingFace => write!(f, "https://huggingface.co"),
            DownloadSource::Mirror { url } => write!(f, "{}", url),
            DownloadSource::CustomUrl { url } => write!(f, "{}", url),
            DownloadSource::Local => write!(f, "local"),
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_preset_validation() {
        let config = Config::tiny_preset();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_preset_validation() {
        let config = Config::small_preset();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_base_preset_validation() {
        let config = Config::base_preset();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_percentage() {
        let mut config = Config::tiny_preset();
        config.dataset.mix[0].percentage = 0.5;
        let result = config.validate_detailed();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_invalid_sequence_length() {
        let mut config = Config::tiny_preset();
        config.training.sequence_length = 200;
        config.model.max_position_embeddings = 128;
        let result = config.validate_detailed();
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_invalid_learning_rate() {
        let mut config = Config::tiny_preset();
        config.training.learning_rate = -0.001;
        let result = config.validate_detailed();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("学习率")));
    }
    
    #[test]
    fn test_invalid_hidden_dim_divisibility() {
        let mut config = Config::tiny_preset();
        config.model.hidden_dim = 257;
        config.model.num_heads = 8;
        let result = config.validate_detailed();
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("必须能被")));
    }
    
    #[test]
    fn test_warning_on_large_dropout() {
        let mut config = Config::tiny_preset();
        config.model.dropout = 0.6;
        let result = config.validate_detailed();
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }
    
    #[test]
    fn test_recommend_for_hardware() {
        let config = Config::recommend_for_hardware();
        assert!(config.validate().is_ok());
    }
}