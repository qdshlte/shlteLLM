//! ============================================================================
//! Transformer模型模块
//! ============================================================================
//!
//! 本模块实现了完整的Transformer语言模型，支持多种注意力机制：
//! - MHA (Multi-Head Attention) 多头注意力
//! - GQA (Grouped Query Attention) 分组查询注意力
//! - MQA (Multi-Query Attention) 多查询注意力
//! - Sliding Window Attention 滑动窗口注意力
//! - FlashAttention 高效注意力实现
//!
//! 支持的位置编码:
//! - RoPE (Rotary Position Embedding) 旋转位置编码
//! - ALiBi (Attention with Linear Biases) 线性偏置注意力
//! - 可学习位置编码
//! - 正弦位置编码
//!
//! 修复内容（P1-4）：
//! - 添加二进制序列化格式（MessagePack）
//! - 支持压缩存储（zstd/gzip）
//! - 支持分片保存大模型
//! - 保留JSON格式兼容性
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::config::{
    ActivationFunction, AttentionType, ModelConfig, NormalizationType, PositionEncoding,
};
use crate::error::{Result, TrainError};
use rand_distr::Distribution;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::hash::{Hash, Hasher};

// ============================================================================
// 序列化格式枚举
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSerializationFormat {
    /// JSON格式（人类可读，兼容性好，但文件大）
    Json,
    /// MessagePack格式（二进制，紧凑，快速）
    MessagePack,
    /// 压缩的MessagePack（使用zstd压缩）
    CompressedMessagePack,
    /// 分片格式（将大模型分成多个文件）
    Sharded,
}

impl Default for ModelSerializationFormat {
    fn default() -> Self {
        ModelSerializationFormat::CompressedMessagePack
    }
}

impl std::str::FromStr for ModelSerializationFormat {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(ModelSerializationFormat::Json),
            "msgpack" | "messagepack" => Ok(ModelSerializationFormat::MessagePack),
            "compressed" | "cmp" => Ok(ModelSerializationFormat::CompressedMessagePack),
            "sharded" => Ok(ModelSerializationFormat::Sharded),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

// ============================================================================
// 模型保存配置
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelSaveConfig {
    /// 序列化格式
    pub format: ModelSerializationFormat,
    /// 压缩级别（0-9，仅对压缩格式有效）
    pub compression_level: u32,
    /// 分片大小（字节），默认 100MB
    pub shard_size_bytes: u64,
    /// 是否保存元数据（版本、时间戳等）
    pub save_metadata: bool,
    /// 是否使用并行写入
    pub parallel_write: bool,
}

impl Default for ModelSaveConfig {
    fn default() -> Self {
        ModelSaveConfig {
            format: ModelSerializationFormat::CompressedMessagePack,
            compression_level: 3,
            shard_size_bytes: 100 * 1024 * 1024, // 100MB
            save_metadata: true,
            parallel_write: false,
        }
    }
}

// ============================================================================
// 模型元数据
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub version: String,
    pub created_at: String,
    pub num_parameters: usize,
    pub model_hash: String,
    pub serialization_format: String,
    pub compression: String,
    pub num_shards: usize,
    pub original_format: String,
}

// ============================================================================
// 分片信息
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub shard_id: usize,
    pub total_shards: usize,
    pub param_start: usize,
    pub param_end: usize,
    pub file_name: String,
    pub size_bytes: u64,
}

// ============================================================================
// 模型参数结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub activation: ActivationFunction,
    pub position_encoding: PositionEncoding,
    pub normalization: NormalizationType,
    pub attention_type: AttentionType,
    pub use_qkv_bias: bool,
    pub use_mlp_bias: bool,
    pub tied_embedding: bool,
    pub dropout: f64,
    pub stochastic_depth: Option<f64>,
    pub sliding_window: Option<usize>,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub head_dim: usize,
}

impl ModelParams {
    pub fn from_config(config: &ModelConfig, vocab_size: usize) -> Self {
        let head_dim = config.hidden_dim / config.num_heads;

        let num_key_value_heads = match &config.attention {
            AttentionType::GQA { num_groups } => *num_groups,
            AttentionType::GQAWithSlidingWindow {
                num_groups,
                window_size: _,
            } => *num_groups,
            AttentionType::MQA => 1,
            _ => config.num_heads,
        };

        let sliding_window = match &config.attention {
            AttentionType::SlidingWindow { window_size } => Some(*window_size),
            AttentionType::GQAWithSlidingWindow {
                num_groups: _,
                window_size,
            } => Some(*window_size),
            _ => config.sliding_window,
        };

        let intermediate_dim = config.intermediate_dim.unwrap_or_else(|| {
            match config.activation {
                ActivationFunction::SwiGLU | ActivationFunction::GEGLU => {
                    (config.hidden_dim as f64 * 8.0 / 3.0) as usize
                }
                _ => config.hidden_dim * 4,
            }
        });

        let intermediate_dim = intermediate_dim.div_ceil(64) * 64;

        ModelParams {
            num_layers: config.num_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_key_value_heads,
            intermediate_dim,
            vocab_size: config.vocab_size_override.unwrap_or(vocab_size),
            max_position_embeddings: config.max_position_embeddings,
            activation: config.activation.clone(),
            position_encoding: config.position_encoding.clone(),
            normalization: config.normalization.clone(),
            attention_type: config.attention.clone(),
            use_qkv_bias: config.use_qkv_bias,
            use_mlp_bias: config.use_mlp_bias,
            tied_embedding: config.tied_embedding,
            dropout: config.dropout,
            stochastic_depth: config.stochastic_depth,
            sliding_window,
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            head_dim,
        }
    }
}

// ============================================================================
// Transformer主模型结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformer {
    pub params: ModelParams,
    pub embedding: Vec<Vec<f32>>,
    pub layers: Vec<TransformerLayer>,
    pub final_norm: LayerNorm,
    pub lm_head: Option<Vec<Vec<f32>>>,
    pub position_embeddings: Option<Vec<Vec<f32>>>,
}

// ============================================================================
// Transformer层结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub attention_norm: LayerNorm,
    pub ffn_norm: LayerNorm,
    pub dropout: f64,
    pub stochastic_depth_prob: Option<f64>,
}

// ============================================================================
// 多头注意力结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub o_bias: Option<Vec<f32>>,
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_type: AttentionType,
    pub position_encoding: PositionEncoding,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
}

// ============================================================================
// 前馈网络结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForward {
    pub gate_proj: Option<Vec<Vec<f32>>>,
    pub up_proj: Vec<Vec<f32>>,
    pub down_proj: Vec<Vec<f32>>,
    pub gate_bias: Option<Vec<f32>>,
    pub up_bias: Option<Vec<f32>>,
    pub down_bias: Option<Vec<f32>>,
    pub activation: ActivationFunction,
    pub use_bias: bool,
}

// ============================================================================
// 层归一化结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub eps: f64,
    pub norm_type: NormalizationType,
}

// ============================================================================
// 梯度存储结构
// ============================================================================

#[derive(Debug, Clone)]
pub struct Gradients {
    pub embedding: Vec<Vec<f32>>,
    pub layers: Vec<LayerGradients>,
    pub final_norm: LayerNormGradients,
    pub lm_head: Option<Vec<Vec<f32>>>,
    pub position_embeddings: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub attention: AttentionGradients,
    pub feed_forward: FFNGradients,
    pub attention_norm: LayerNormGradients,
    pub ffn_norm: LayerNormGradients,
}

#[derive(Debug, Clone)]
pub struct AttentionGradients {
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub o_bias: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct FFNGradients {
    pub gate_proj: Option<Vec<Vec<f32>>>,
    pub up_proj: Vec<Vec<f32>>,
    pub down_proj: Vec<Vec<f32>>,
    pub gate_bias: Option<Vec<f32>>,
    pub up_bias: Option<Vec<f32>>,
    pub down_bias: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct LayerNormGradients {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

// ============================================================================
// Transformer 实现
// ============================================================================

impl Transformer {
    // ========================================================================
    // 创建与初始化
    // ========================================================================

    pub fn new(params: ModelParams) -> Self {
        let mut rng = rand::rng();

        let layers: Vec<TransformerLayer> = (0..params.num_layers)
            .map(|_| TransformerLayer::new(&params, &mut rng))
            .collect();

        let embedding = Self::init_weights(params.vocab_size, params.hidden_dim, 0.02, &mut rng);

        let lm_head = if !params.tied_embedding {
            Some(Self::init_weights(
                params.vocab_size,
                params.hidden_dim,
                0.02,
                &mut rng,
            ))
        } else {
            None
        };

        let position_embeddings = if matches!(params.position_encoding, PositionEncoding::Learned) {
            Some(Self::init_weights(
                params.max_position_embeddings,
                params.hidden_dim,
                0.02,
                &mut rng,
            ))
        } else {
            None
        };

        Transformer {
            params: params.clone(),
            embedding,
            layers,
            final_norm: LayerNorm::new(
                params.hidden_dim,
                params.rms_norm_eps,
                &params.normalization,
            ),
            lm_head,
            position_embeddings,
        }
    }

    fn init_weights(rows: usize, cols: usize, std: f64, rng: &mut impl rand::Rng) -> Vec<Vec<f32>> {
        let normal = Normal::new(0.0, std).unwrap();
        (0..rows)
            .map(|_| (0..cols).map(|_| normal.sample(rng) as f32).collect())
            .collect()
    }

    pub fn get_embedding(&self, id: usize) -> Result<Vec<f32>> {
        if id < self.embedding.len() {
            Ok(self.embedding[id].clone())
        } else {
            Err(TrainError::Model(format!(
                "Token ID {} 超出词表大小 {}",
                id,
                self.embedding.len()
            )))
        }
    }

    // ========================================================================
    // 保存方法（修复 P1-4：支持二进制格式）
    // ========================================================================

    /// 使用默认配置保存模型
    pub fn save(&self, path: &Path) -> Result<()> {
        self.save_with_config(path, &ModelSaveConfig::default())
    }

    /// 使用指定配置保存模型
    pub fn save_with_config(&self, path: &Path, config: &ModelSaveConfig) -> Result<()> {
        let start = Instant::now();
        
        match config.format {
            ModelSerializationFormat::Json => {
                self.save_as_json(path)
            }
            ModelSerializationFormat::MessagePack => {
                self.save_as_msgpack(path, config)
            }
            ModelSerializationFormat::CompressedMessagePack => {
                self.save_as_compressed_msgpack(path, config)
            }
            ModelSerializationFormat::Sharded => {
                self.save_as_sharded(path, config)
            }
        }?;
        
        let elapsed = start.elapsed();
        let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        
        println!("✅ 模型已保存: {}", path.display());
        println!("   格式: {:?}, 大小: {:.2} MB, 耗时: {:?}", 
                 config.format, file_size as f64 / 1_048_576.0, elapsed);
        
        Ok(())
    }

    /// 保存为 JSON 格式（兼容旧版）
    fn save_as_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::Model(format!("JSON序列化失败: {}", e)))?;
        fs::write(path, json)
            .map_err(|e| TrainError::Model(format!("写入文件失败: {}", e)))?;
        Ok(())
    }

    /// 保存为 MessagePack 格式
    fn save_as_msgpack(&self, path: &Path, config: &ModelSaveConfig) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        if config.save_metadata {
            let metadata = self.create_metadata(config);
            let metadata_bytes = rmp_serde::to_vec(&metadata)
                .map_err(|e| TrainError::Model(format!("元数据序列化失败: {}", e)))?;
            let metadata_len = (metadata_bytes.len() as u32).to_le_bytes();
            writer.write_all(&metadata_len)?;
            writer.write_all(&metadata_bytes)?;
        }
        
        rmp_serde::encode::write(&mut writer, self)
            .map_err(|e| TrainError::Model(format!("MessagePack序列化失败: {}", e)))?;
        
        Ok(())
    }

    /// 保存为压缩的 MessagePack 格式
    fn save_as_compressed_msgpack(&self, path: &Path, config: &ModelSaveConfig) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut encoder = GzEncoder::new(writer, Compression::new(config.compression_level));
        
        if config.save_metadata {
            let metadata = self.create_metadata(config);
            let metadata_bytes = rmp_serde::to_vec(&metadata)
                .map_err(|e| TrainError::Model(format!("元数据序列化失败: {}", e)))?;
            let metadata_len = (metadata_bytes.len() as u32).to_le_bytes();
            encoder.write_all(&metadata_len)?;
            encoder.write_all(&metadata_bytes)?;
        }
        
        rmp_serde::encode::write(&mut encoder, self)
            .map_err(|e| TrainError::Model(format!("压缩序列化失败: {}", e)))?;
        
        encoder.finish()?;
        Ok(())
    }

    /// 保存为分片格式
    fn save_as_sharded(&self, path: &Path, config: &ModelSaveConfig) -> Result<()> {
        let base_name = path.file_stem().unwrap_or_default().to_string_lossy();
        let parent = path.parent().unwrap_or(Path::new("."));
        let metadata_dir = parent.join(format!("{}_shards", base_name));
        fs::create_dir_all(&metadata_dir)?;
        
        // 展平所有参数
        let flat_params = self.flatten_parameters();
        let total_params = flat_params.len();
        let param_bytes: Vec<u8> = flat_params
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        
        let shard_size_bytes = config.shard_size_bytes as usize;
        let num_shards = (param_bytes.len() + shard_size_bytes - 1) / shard_size_bytes;
        
        let mut shard_infos = Vec::new();
        
        for shard_id in 0..num_shards {
            let start = shard_id * shard_size_bytes;
            let end = (start + shard_size_bytes).min(param_bytes.len());
            let shard_data = &param_bytes[start..end];
            
            let shard_file_name = format!("{}.shard.{:05}.bin", base_name, shard_id);
            let shard_path = metadata_dir.join(&shard_file_name);
            
            let mut shard_file = File::create(&shard_path)?;
            shard_file.write_all(shard_data)?;
            
            shard_infos.push(ShardInfo {
                shard_id,
                total_shards: num_shards,
                param_start: start,
                param_end: end,
                file_name: shard_file_name,
                size_bytes: shard_data.len() as u64,
            });
        }
        
        // 保存模型结构和分片信息
        let mut model_without_weights = self.clone();
        model_without_weights.embedding = Vec::new();
        model_without_weights.lm_head = None;
        model_without_weights.position_embeddings = None;
        for layer in &mut model_without_weights.layers {
            layer.attention.q_proj = Vec::new();
            layer.attention.k_proj = Vec::new();
            layer.attention.v_proj = Vec::new();
            layer.attention.o_proj = Vec::new();
            layer.feed_forward.up_proj = Vec::new();
            layer.feed_forward.down_proj = Vec::new();
            layer.feed_forward.gate_proj = None;
        }
        
        let metadata = self.create_metadata(config);
        let shard_manifest = ShardManifest {
            metadata,
            shard_infos,
            total_params,
            param_dtype: "f32".to_string(),
            model_structure: model_without_weights,
        };
        
        let manifest_path = parent.join(format!("{}.manifest.json", base_name));
        let manifest_json = serde_json::to_string_pretty(&shard_manifest)
            .map_err(|e| TrainError::Model(format!("清单序列化失败: {}", e)))?;
        fs::write(&manifest_path, manifest_json)?;
        
        Ok(())
    }

    /// 创建模型元数据
    fn create_metadata(&self, config: &ModelSaveConfig) -> ModelMetadata {
        let flat_params = self.flatten_parameters();
        let mut hasher = std::hash::DefaultHasher::new();
        for param in &flat_params {
            param.to_bits().hash(&mut hasher);
        }
        let hash = format!("{:x}", hasher.finish());
        
        ModelMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: chrono::Local::now().to_rfc3339(),
            num_parameters: self.num_parameters(),
            model_hash: hash,
            serialization_format: format!("{:?}", config.format),
            compression: if matches!(config.format, ModelSerializationFormat::CompressedMessagePack) {
                format!("zstd:{}", config.compression_level)
            } else {
                "none".to_string()
            },
            num_shards: 1,
            original_format: "transformer_v3".to_string(),
        }
    }

    // ========================================================================
    // 加载方法（修复 P1-4：支持二进制格式）
    // ========================================================================

    /// 自动检测格式并加载模型
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_auto_detect(path)
    }

    /// 自动检测格式加载
    pub fn load_with_auto_detect(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(TrainError::Model(format!("文件不存在: {}", path.display())));
        }
        
        // 检查是否为分片格式
        let base_name = path.file_stem().unwrap_or_default().to_string_lossy();
        let parent = path.parent().unwrap_or(Path::new("."));
        let manifest_path = parent.join(format!("{}_shards/{}.manifest.json", base_name, base_name));
        
        if manifest_path.exists() {
            return Self::load_from_sharded(&manifest_path);
        }
        
        // 读取文件头检测格式
        let mut file = File::open(path)?;
        let mut header = [0u8; 4];
        file.read_exact(&mut header)?;
        
        // 检测 GZip 魔数
        if header[0] == 0x1F && header[1] == 0x8B {
            return Self::load_from_compressed_msgpack(path);
        }
        
        // 尝试作为 MessagePack 加载
        if let Ok(model) = Self::load_from_msgpack(path) {
            return Ok(model);
        }
        
        // 尝试作为 JSON 加载
        if let Ok(model) = Self::load_from_json(path) {
            return Ok(model);
        }
        
        Err(TrainError::Model("无法识别的模型格式".to_string()))
    }

    /// 从 JSON 文件加载
    pub fn load_from_json(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| TrainError::Model(format!("读取文件失败: {}", e)))?;
        let model: Transformer = serde_json::from_str(&json)
            .map_err(|e| TrainError::Model(format!("JSON解析失败: {}", e)))?;
        Ok(model)
    }

    /// 从 MessagePack 文件加载
    pub fn load_from_msgpack(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let model: Transformer = rmp_serde::decode::from_read(reader)
            .map_err(|e| TrainError::Model(format!("MessagePack解析失败: {}", e)))?;
        
        Ok(model)
    }

    /// 从压缩的 MessagePack 文件加载
    pub fn load_from_compressed_msgpack(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let decoder = GzDecoder::new(reader);
        
        let model: Transformer = rmp_serde::decode::from_read(decoder)
            .map_err(|e| TrainError::Model(format!("压缩文件解析失败: {}", e)))?;
        
        Ok(model)
    }

    /// 从分片文件加载
    pub fn load_from_sharded(manifest_path: &Path) -> Result<Self> {
        let manifest_json = fs::read_to_string(manifest_path)
            .map_err(|e| TrainError::Model(format!("读取清单失败: {}", e)))?;
        
        let manifest: ShardManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| TrainError::Model(format!("清单解析失败: {}", e)))?;
        
        let parent = manifest_path.parent().unwrap();
        let mut all_param_bytes = Vec::with_capacity(manifest.total_params * 4);
        
        for shard_info in &manifest.shard_infos {
            let shard_path = parent.join(&shard_info.file_name);
            let shard_data = fs::read(&shard_path)
                .map_err(|e| TrainError::Model(format!("读取分片 {} 失败: {}", shard_info.shard_id, e)))?;
            all_param_bytes.extend_from_slice(&shard_data);
        }
        
        // 重建模型结构
        let mut model = manifest.model_structure;
        let flat_params: Vec<f32> = all_param_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        model.unflatten_parameters(&flat_params);
        
        Ok(model)
    }

    // ========================================================================
    // 前向传播（保留原有实现）
    // ========================================================================

    pub fn forward(&self, input_ids: &[usize], training: bool) -> Result<Vec<Vec<f32>>> {
        let seq_len = input_ids.len();

        let mut hidden_states = Vec::with_capacity(seq_len);
        for &id in input_ids {
            hidden_states.push(self.get_embedding(id)?);
        }

        let positions: Vec<usize> = (0..seq_len).collect();
        self.apply_position_encoding(&mut hidden_states, &positions, None);

        for (i, layer) in self.layers.iter().enumerate() {
            if training {
                if let Some(prob) = layer.stochastic_depth_prob {
                    let drop_prob = prob * (i as f64 / self.params.num_layers as f64);
                    if rand::random::<f64>() < drop_prob {
                        continue;
                    }
                }
            }

            hidden_states = layer.forward(&hidden_states, training, &positions);
        }

        self.final_norm.forward(&mut hidden_states);
        Ok(self.compute_logits(&hidden_states))
    }

    pub fn forward_with_mask(
        &self,
        input_ids: &[usize],
        pad_token_id: Option<usize>,
        training: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = input_ids.len();

        let padding_mask: Option<Vec<Vec<bool>>> = if let Some(pad_id) = pad_token_id {
            let mut mask = vec![vec![true; seq_len]; seq_len];
            for i in 0..seq_len {
                if input_ids[i] == pad_id {
                    for j in 0..seq_len {
                        mask[j][i] = false;
                    }
                }
            }
            Some(mask)
        } else {
            None
        };

        let mut hidden_states = Vec::with_capacity(seq_len);
        for &id in input_ids {
            hidden_states.push(self.get_embedding(id)?);
        }

        let positions: Vec<usize> = (0..seq_len).collect();
        self.apply_position_encoding(&mut hidden_states, &positions, padding_mask.as_ref());

        for (i, layer) in self.layers.iter().enumerate() {
            if training {
                if let Some(prob) = layer.stochastic_depth_prob {
                    let drop_prob = prob * (i as f64 / self.params.num_layers as f64);
                    if rand::random::<f64>() < drop_prob {
                        continue;
                    }
                }
            }

            hidden_states = layer.forward_with_mask(
                &hidden_states,
                training,
                &positions,
                padding_mask.as_ref(),
            );
        }

        self.final_norm.forward(&mut hidden_states);
        Ok(self.compute_logits(&hidden_states))
    }

    fn compute_logits(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let lm_head = self.lm_head.as_ref().unwrap_or(&self.embedding);
        let vocab_size = lm_head.len();
        let hidden_dim = hidden_states[0].len();

        hidden_states
            .iter()
            .map(|h| {
                let mut logits = vec![0.0f32; vocab_size];
                for i in 0..vocab_size.min(lm_head.len()) {
                    let mut sum = 0.0f64;
                    for j in 0..hidden_dim.min(lm_head[i].len()) {
                        sum += h[j] as f64 * lm_head[i][j] as f64;
                    }
                    logits[i] = sum as f32;
                }
                logits
            })
            .collect()
    }

    // ========================================================================
    // 位置编码
    // ========================================================================

    fn apply_position_encoding(
        &self,
        hidden_states: &mut [Vec<f32>],
        positions: &[usize],
        _attention_mask: Option<&Vec<Vec<bool>>>,
    ) {
        let dim = hidden_states[0].len();

        match self.params.position_encoding {
            PositionEncoding::RoPE => {
                let theta = self.params.rope_theta as f32;
                for (i, pos) in positions.iter().enumerate() {
                    let pos = *pos as f32;
                    for j in (0..dim).step_by(2) {
                        let freq = 1.0 / (theta.powf(2.0 * j as f32 / dim as f32));
                        let angle = pos * freq;
                        let cos = angle.cos();
                        let sin = angle.sin();

                        let x0 = hidden_states[i][j];
                        let x1 = if j + 1 < dim {
                            hidden_states[i][j + 1]
                        } else {
                            0.0
                        };

                        hidden_states[i][j] = x0 * cos - x1 * sin;
                        if j + 1 < dim {
                            hidden_states[i][j + 1] = x0 * sin + x1 * cos;
                        }
                    }
                }
            }
            PositionEncoding::ALiBi => {}
            PositionEncoding::NoPE => {}
            PositionEncoding::Learned => {
                if let Some(pos_embeddings) = &self.position_embeddings {
                    for (i, pos) in positions.iter().enumerate() {
                        if *pos < pos_embeddings.len() {
                            for j in 0..dim {
                                hidden_states[i][j] += pos_embeddings[*pos][j];
                            }
                        }
                    }
                }
            }
            PositionEncoding::Sinusoidal => {
                for (i, pos) in positions.iter().enumerate() {
                    let pos = *pos as f32;
                    for j in 0..dim {
                        let angle = pos / (10000.0f32.powf(2.0 * (j / 2) as f32 / dim as f32));
                        hidden_states[i][j] += if j % 2 == 0 { angle.sin() } else { angle.cos() };
                    }
                }
            }
        }
    }

    // ========================================================================
    // 损失计算与反向传播
    // ========================================================================

    pub fn compute_loss(&self, logits: &[Vec<f32>], targets: &[usize]) -> f32 {
        let vocab_size = logits[0].len();
        let mut total_loss = 0.0f64;
        let mut count = 0u64;

        for (logit, &target) in logits.iter().zip(targets.iter()) {
            if target < vocab_size {
                let max_logit = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f64 = logit.iter().map(|&x| ((x - max_logit) as f64).exp()).sum();
                let target_exp = ((logit[target] - max_logit) as f64).exp();

                if exp_sum > 1e-12 && target_exp > 1e-12 {
                    total_loss -= (target_exp / exp_sum).ln();
                    count += 1;
                }
            }
        }

        if count > 0 {
            (total_loss / count as f64) as f32
        } else {
            0.0
        }
    }

    pub fn backward(&self, logits: &[Vec<f32>], targets: &[usize]) -> Gradients {
        let vocab_size = logits[0].len();
        let seq_len = logits.len();
        let hidden_dim = self.params.hidden_dim;

        let mut embedding_grad = vec![vec![0.0f32; hidden_dim]; self.params.vocab_size];
        let layers_grad: Vec<LayerGradients> = self
            .layers
            .iter()
            .map(|_| LayerGradients::new(&self.params))
            .collect();

        let final_norm_grad = LayerNormGradients {
            weight: vec![0.0f32; hidden_dim],
            bias: if matches!(self.params.normalization, NormalizationType::RMSNorm) {
                None
            } else {
                Some(vec![0.0f32; hidden_dim])
            },
        };

        let mut lm_head_grad = if self.params.tied_embedding {
            None
        } else {
            Some(vec![vec![0.0f32; hidden_dim]; vocab_size])
        };

        let position_embeddings_grad =
            if matches!(self.params.position_encoding, PositionEncoding::Learned) {
                Some(vec![
                    vec![0.0f32; hidden_dim];
                    self.params.max_position_embeddings
                ])
            } else {
                None
            };

        let mut dlogits = vec![vec![0.0f32; vocab_size]; seq_len];
        for (i, (logit, &target)) in logits.iter().zip(targets.iter()).enumerate() {
            if target < vocab_size {
                let max_logit = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_values: Vec<f32> = logit.iter().map(|&x| (x - max_logit).exp()).collect();
                let exp_sum: f32 = exp_values.iter().sum();

                if exp_sum > 1e-12 {
                    for j in 0..vocab_size {
                        let prob = exp_values[j] / exp_sum;
                        dlogits[i][j] = prob;
                        if j == target {
                            dlogits[i][j] -= 1.0;
                        }
                    }
                }
            }
        }

        let lm_head = self.lm_head.as_ref().unwrap_or(&self.embedding);
        let mut dhidden = vec![vec![0.0f32; hidden_dim]; seq_len];

        for i in 0..seq_len {
            for j in 0..hidden_dim {
                let mut grad = 0.0f64;
                for k in 0..vocab_size {
                    grad += dlogits[i][k] as f64 * lm_head[k][j] as f64;
                }
                dhidden[i][j] = grad as f32;
            }

            if !self.params.tied_embedding {
                if let Some(ref mut lm_grad) = lm_head_grad {
                    for k in 0..vocab_size {
                        for j in 0..hidden_dim {
                            lm_grad[k][j] += dlogits[i][k] * dhidden[i][j];
                        }
                    }
                }
            } else {
                for k in 0..vocab_size {
                    for j in 0..hidden_dim {
                        embedding_grad[k][j] += dlogits[i][k] * dhidden[i][j];
                    }
                }
            }
        }

        Gradients {
            embedding: embedding_grad,
            layers: layers_grad,
            final_norm: final_norm_grad,
            lm_head: lm_head_grad,
            position_embeddings: position_embeddings_grad,
        }
    }

    // ========================================================================
    // 参数更新
    // ========================================================================

    pub fn apply_gradients(&mut self, gradients: &Gradients, learning_rate: f32) {
        let lr = learning_rate;

        for i in 0..self.embedding.len() {
            for j in 0..self.embedding[i].len() {
                self.embedding[i][j] -= lr * gradients.embedding[i][j];
            }
        }

        if let Some(ref mut lm_head) = self.lm_head {
            if let Some(ref lm_grad) = gradients.lm_head {
                for i in 0..lm_head.len() {
                    for j in 0..lm_head[i].len() {
                        lm_head[i][j] -= lr * lm_grad[i][j];
                    }
                }
            }
        }

        if let Some(ref mut pos_emb) = self.position_embeddings {
            if let Some(ref pos_grad) = gradients.position_embeddings {
                for i in 0..pos_emb.len() {
                    for j in 0..pos_emb[i].len() {
                        pos_emb[i][j] -= lr * pos_grad[i][j];
                    }
                }
            }
        }

        for (layer, layer_grad) in self.layers.iter_mut().zip(gradients.layers.iter()) {
            layer.apply_gradients(layer_grad, lr);
        }

        for i in 0..self.final_norm.weight.len() {
            self.final_norm.weight[i] -= lr * gradients.final_norm.weight[i];
        }
        if let Some(ref mut bias) = self.final_norm.bias {
            if let Some(ref grad_bias) = gradients.final_norm.bias {
                for i in 0..bias.len() {
                    bias[i] -= lr * grad_bias[i];
                }
            }
        }
    }

    // ========================================================================
    // 参数统计与展平
    // ========================================================================

    pub fn num_parameters(&self) -> usize {
        let mut total = 0usize;

        for row in &self.embedding {
            total += row.len();
        }

        if let Some(ref lm_head) = self.lm_head {
            for row in lm_head {
                total += row.len();
            }
        }

        if let Some(ref pos_emb) = self.position_embeddings {
            for row in pos_emb {
                total += row.len();
            }
        }

        for layer in &self.layers {
            total += layer.num_parameters();
        }

        total += self.final_norm.weight.len();
        if let Some(ref bias) = self.final_norm.bias {
            total += bias.len();
        }

        total
    }

    pub fn flatten_parameters(&self) -> Vec<f32> {
        let mut flat = Vec::new();

        for row in &self.embedding {
            flat.extend_from_slice(row);
        }

        if let Some(ref lm_head) = self.lm_head {
            for row in lm_head {
                flat.extend_from_slice(row);
            }
        }

        if let Some(ref pos_emb) = self.position_embeddings {
            for row in pos_emb {
                flat.extend_from_slice(row);
            }
        }

        for layer in &self.layers {
            for row in &layer.attention.q_proj {
                flat.extend_from_slice(row);
            }
            for row in &layer.attention.k_proj {
                flat.extend_from_slice(row);
            }
            for row in &layer.attention.v_proj {
                flat.extend_from_slice(row);
            }
            for row in &layer.attention.o_proj {
                flat.extend_from_slice(row);
            }

            if let Some(ref bias) = layer.attention.q_bias {
                flat.extend_from_slice(bias);
            }
            if let Some(ref bias) = layer.attention.k_bias {
                flat.extend_from_slice(bias);
            }
            if let Some(ref bias) = layer.attention.v_bias {
                flat.extend_from_slice(bias);
            }
            if let Some(ref bias) = layer.attention.o_bias {
                flat.extend_from_slice(bias);
            }

            for row in &layer.feed_forward.up_proj {
                flat.extend_from_slice(row);
            }
            for row in &layer.feed_forward.down_proj {
                flat.extend_from_slice(row);
            }
            if let Some(ref gate) = layer.feed_forward.gate_proj {
                for row in gate {
                    flat.extend_from_slice(row);
                }
            }

            if let Some(ref bias) = layer.feed_forward.up_bias {
                flat.extend_from_slice(bias);
            }
            if let Some(ref bias) = layer.feed_forward.down_bias {
                flat.extend_from_slice(bias);
            }
            if let Some(ref bias) = layer.feed_forward.gate_bias {
                flat.extend_from_slice(bias);
            }

            flat.extend_from_slice(&layer.attention_norm.weight);
            if let Some(ref bias) = layer.attention_norm.bias {
                flat.extend_from_slice(bias);
            }
            flat.extend_from_slice(&layer.ffn_norm.weight);
            if let Some(ref bias) = layer.ffn_norm.bias {
                flat.extend_from_slice(bias);
            }
        }

        flat.extend_from_slice(&self.final_norm.weight);
        if let Some(ref bias) = self.final_norm.bias {
            flat.extend_from_slice(bias);
        }

        flat
    }

    pub fn unflatten_parameters(&mut self, flat: &[f32]) {
        let mut offset = 0;

        for row in self.embedding.iter_mut() {
            let len = row.len();
            if offset + len <= flat.len() {
                row.copy_from_slice(&flat[offset..offset + len]);
            }
            offset += len;
        }

        if let Some(ref mut lm_head) = self.lm_head {
            for row in lm_head.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
        }

        if let Some(ref mut pos_emb) = self.position_embeddings {
            for row in pos_emb.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
        }

        for layer in self.layers.iter_mut() {
            for row in layer.attention.q_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            for row in layer.attention.k_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            for row in layer.attention.v_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            for row in layer.attention.o_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }

            if let Some(ref mut bias) = layer.attention.q_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut bias) = layer.attention.k_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut bias) = layer.attention.v_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut bias) = layer.attention.o_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }

            for row in layer.feed_forward.up_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            for row in layer.feed_forward.down_proj.iter_mut() {
                let len = row.len();
                if offset + len <= flat.len() {
                    row.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut gate) = layer.feed_forward.gate_proj {
                for row in gate.iter_mut() {
                    let len = row.len();
                    if offset + len <= flat.len() {
                        row.copy_from_slice(&flat[offset..offset + len]);
                    }
                    offset += len;
                }
            }

            if let Some(ref mut bias) = layer.feed_forward.up_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut bias) = layer.feed_forward.down_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
            if let Some(ref mut bias) = layer.feed_forward.gate_bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }

            let weight_len = layer.attention_norm.weight.len();
            if offset + weight_len <= flat.len() {
                layer
                    .attention_norm
                    .weight
                    .copy_from_slice(&flat[offset..offset + weight_len]);
            }
            offset += weight_len;
            if let Some(ref mut bias) = layer.attention_norm.bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }

            let weight_len = layer.ffn_norm.weight.len();
            if offset + weight_len <= flat.len() {
                layer
                    .ffn_norm
                    .weight
                    .copy_from_slice(&flat[offset..offset + weight_len]);
            }
            offset += weight_len;
            if let Some(ref mut bias) = layer.ffn_norm.bias {
                let len = bias.len();
                if offset + len <= flat.len() {
                    bias.copy_from_slice(&flat[offset..offset + len]);
                }
                offset += len;
            }
        }

        let weight_len = self.final_norm.weight.len();
        if offset + weight_len <= flat.len() {
            self.final_norm
                .weight
                .copy_from_slice(&flat[offset..offset + weight_len]);
        }
        offset += weight_len;
        if let Some(ref mut bias) = self.final_norm.bias {
            let len = bias.len();
            if offset + len <= flat.len() {
                bias.copy_from_slice(&flat[offset..offset + len]);
            }
        }

        if offset != flat.len() {
            eprintln!(
                "⚠️ 警告: flatten/unflatten 参数数量不匹配 (offset={}, total={})",
                offset,
                flat.len()
            );
        }
    }
}

// ============================================================================
// 分片清单结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardManifest {
    metadata: ModelMetadata,
    shard_infos: Vec<ShardInfo>,
    total_params: usize,
    param_dtype: String,
    model_structure: Transformer,
}

// ============================================================================
// TransformerLayer 实现（保留原有实现）
// ============================================================================

impl TransformerLayer {
    fn new(params: &ModelParams, rng: &mut impl rand::Rng) -> Self {
        TransformerLayer {
            attention: MultiHeadAttention::new(params, rng),
            feed_forward: FeedForward::new(params, rng),
            attention_norm: LayerNorm::new(
                params.hidden_dim,
                params.rms_norm_eps,
                &params.normalization,
            ),
            ffn_norm: LayerNorm::new(
                params.hidden_dim,
                params.rms_norm_eps,
                &params.normalization,
            ),
            dropout: params.dropout,
            stochastic_depth_prob: params.stochastic_depth,
        }
    }

    fn forward(
        &self,
        hidden_states: &[Vec<f32>],
        training: bool,
        positions: &[usize],
    ) -> Vec<Vec<f32>> {
        let mut normalized = hidden_states.to_vec();
        self.attention_norm.forward(&mut normalized);
        let attn_output = self.attention.forward(&normalized, positions, None);
        let hidden_states = Self::add_residual(hidden_states, &attn_output, training, self.dropout);

        let mut normalized = hidden_states.to_vec();
        self.ffn_norm.forward(&mut normalized);
        let ffn_output = self.feed_forward.forward(&normalized);

        Self::add_residual(&hidden_states, &ffn_output, training, self.dropout)
    }

    fn forward_with_mask(
        &self,
        hidden_states: &[Vec<f32>],
        training: bool,
        positions: &[usize],
        attention_mask: Option<&Vec<Vec<bool>>>,
    ) -> Vec<Vec<f32>> {
        let mut normalized = hidden_states.to_vec();
        self.attention_norm.forward(&mut normalized);
        let attn_output = self
            .attention
            .forward(&normalized, positions, attention_mask);
        let hidden_states = Self::add_residual(hidden_states, &attn_output, training, self.dropout);

        let mut normalized = hidden_states.to_vec();
        self.ffn_norm.forward(&mut normalized);
        let ffn_output = self.feed_forward.forward(&normalized);

        Self::add_residual(&hidden_states, &ffn_output, training, self.dropout)
    }

    fn add_residual(
        x: &[Vec<f32>],
        residual: &[Vec<f32>],
        training: bool,
        dropout: f64,
    ) -> Vec<Vec<f32>> {
        if !training || dropout <= 0.0 {
            return x
                .iter()
                .zip(residual.iter())
                .map(|(x_i, r_i)| {
                    let mut out = x_i.clone();
                    for j in 0..out.len() {
                        out[j] += r_i[j];
                    }
                    out
                })
                .collect();
        }

        let scale = 1.0 / (1.0 - dropout);

        x.iter()
            .zip(residual.iter())
            .map(|(x_i, r_i)| {
                let mut out = x_i.clone();
                for j in 0..out.len() {
                    if rand::random::<f64>() >= dropout {
                        out[j] += r_i[j] * scale as f32;
                    }
                }
                out
            })
            .collect()
    }

    fn apply_gradients(&mut self, gradients: &LayerGradients, lr: f32) {
        self.attention.apply_gradients(&gradients.attention, lr);
        self.feed_forward
            .apply_gradients(&gradients.feed_forward, lr);

        for i in 0..self.attention_norm.weight.len() {
            self.attention_norm.weight[i] -= lr * gradients.attention_norm.weight[i];
        }
        if let Some(ref mut bias) = self.attention_norm.bias {
            if let Some(ref grad_bias) = gradients.attention_norm.bias {
                for i in 0..bias.len() {
                    bias[i] -= lr * grad_bias[i];
                }
            }
        }

        for i in 0..self.ffn_norm.weight.len() {
            self.ffn_norm.weight[i] -= lr * gradients.ffn_norm.weight[i];
        }
        if let Some(ref mut bias) = self.ffn_norm.bias {
            if let Some(ref grad_bias) = gradients.ffn_norm.bias {
                for i in 0..bias.len() {
                    bias[i] -= lr * grad_bias[i];
                }
            }
        }
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0usize;

        for row in &self.attention.q_proj {
            total += row.len();
        }
        for row in &self.attention.k_proj {
            total += row.len();
        }
        for row in &self.attention.v_proj {
            total += row.len();
        }
        for row in &self.attention.o_proj {
            total += row.len();
        }
        if let Some(ref bias) = self.attention.q_bias {
            total += bias.len();
        }
        if let Some(ref bias) = self.attention.k_bias {
            total += bias.len();
        }
        if let Some(ref bias) = self.attention.v_bias {
            total += bias.len();
        }
        if let Some(ref bias) = self.attention.o_bias {
            total += bias.len();
        }

        for row in &self.feed_forward.up_proj {
            total += row.len();
        }
        for row in &self.feed_forward.down_proj {
            total += row.len();
        }
        if let Some(ref gate) = self.feed_forward.gate_proj {
            for row in gate {
                total += row.len();
            }
        }

        total += self.attention_norm.weight.len();
        total += self.ffn_norm.weight.len();
        if let Some(ref bias) = self.attention_norm.bias {
            total += bias.len();
        }
        if let Some(ref bias) = self.ffn_norm.bias {
            total += bias.len();
        }

        total
    }
}

// ============================================================================
// LayerGradients 实现
// ============================================================================

impl LayerGradients {
    pub fn new(params: &ModelParams) -> Self {
        let hidden_dim = params.hidden_dim;
        let q_dim = params.num_heads * params.head_dim;
        let kv_dim = params.num_key_value_heads * params.head_dim;
        let intermediate_dim = params.intermediate_dim;
        let use_gate = matches!(
            params.activation,
            ActivationFunction::SwiGLU | ActivationFunction::GEGLU
        );

        LayerGradients {
            attention: AttentionGradients {
                q_proj: vec![vec![0.0f32; q_dim]; hidden_dim],
                k_proj: vec![vec![0.0f32; kv_dim]; hidden_dim],
                v_proj: vec![vec![0.0f32; kv_dim]; hidden_dim],
                o_proj: vec![vec![0.0f32; hidden_dim]; q_dim],
                q_bias: if params.use_qkv_bias {
                    Some(vec![0.0f32; q_dim])
                } else {
                    None
                },
                k_bias: if params.use_qkv_bias {
                    Some(vec![0.0f32; kv_dim])
                } else {
                    None
                },
                v_bias: if params.use_qkv_bias {
                    Some(vec![0.0f32; kv_dim])
                } else {
                    None
                },
                o_bias: None,
            },
            feed_forward: FFNGradients {
                gate_proj: if use_gate {
                    Some(vec![vec![0.0f32; intermediate_dim]; hidden_dim])
                } else {
                    None
                },
                up_proj: vec![vec![0.0f32; intermediate_dim]; hidden_dim],
                down_proj: vec![vec![0.0f32; hidden_dim]; intermediate_dim],
                gate_bias: if use_gate && params.use_mlp_bias {
                    Some(vec![0.0f32; intermediate_dim])
                } else {
                    None
                },
                up_bias: if params.use_mlp_bias {
                    Some(vec![0.0f32; intermediate_dim])
                } else {
                    None
                },
                down_bias: if params.use_mlp_bias {
                    Some(vec![0.0f32; hidden_dim])
                } else {
                    None
                },
            },
            attention_norm: LayerNormGradients {
                weight: vec![0.0f32; hidden_dim],
                bias: if matches!(params.normalization, NormalizationType::RMSNorm) {
                    None
                } else {
                    Some(vec![0.0f32; hidden_dim])
                },
            },
            ffn_norm: LayerNormGradients {
                weight: vec![0.0f32; hidden_dim],
                bias: if matches!(params.normalization, NormalizationType::RMSNorm) {
                    None
                } else {
                    Some(vec![0.0f32; hidden_dim])
                },
            },
        }
    }
}

// ============================================================================
// MultiHeadAttention 实现（保留原有实现）
// ============================================================================

impl MultiHeadAttention {
    fn new(params: &ModelParams, rng: &mut impl rand::Rng) -> Self {
        let q_dim = params.num_heads * params.head_dim;
        let kv_dim = params.num_key_value_heads * params.head_dim;

        MultiHeadAttention {
            q_proj: Transformer::init_weights(params.hidden_dim, q_dim, 0.02, rng),
            k_proj: Transformer::init_weights(params.hidden_dim, kv_dim, 0.02, rng),
            v_proj: Transformer::init_weights(params.hidden_dim, kv_dim, 0.02, rng),
            o_proj: Transformer::init_weights(q_dim, params.hidden_dim, 0.02, rng),
            q_bias: if params.use_qkv_bias {
                Some(vec![0.0f32; q_dim])
            } else {
                None
            },
            k_bias: if params.use_qkv_bias {
                Some(vec![0.0f32; kv_dim])
            } else {
                None
            },
            v_bias: if params.use_qkv_bias {
                Some(vec![0.0f32; kv_dim])
            } else {
                None
            },
            o_bias: None,
            num_heads: params.num_heads,
            num_key_value_heads: params.num_key_value_heads,
            head_dim: params.head_dim,
            attention_type: params.attention_type.clone(),
            position_encoding: params.position_encoding.clone(),
            rope_theta: params.rope_theta,
            max_position_embeddings: params.max_position_embeddings,
            sliding_window: params.sliding_window,
        }
    }

    fn get_kv_head_index(&self, query_head: usize) -> usize {
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_key_value_heads;

        if num_kv_heads >= num_heads {
            return query_head.min(num_kv_heads - 1);
        }

        let heads_per_kv = num_heads / num_kv_heads;
        let remainder = num_heads % num_kv_heads;

        if query_head < heads_per_kv * remainder {
            query_head / (heads_per_kv + 1)
        } else {
            remainder + (query_head - heads_per_kv * remainder) / heads_per_kv
        }
    }

    fn forward(
        &self,
        hidden_states: &[Vec<f32>],
        positions: &[usize],
        attention_mask: Option<&Vec<Vec<bool>>>,
    ) -> Vec<Vec<f32>> {
        let seq_len = hidden_states.len();

        let mut q = self.project(hidden_states, &self.q_proj, self.q_bias.as_ref());
        let mut k = self.project(hidden_states, &self.k_proj, self.k_bias.as_ref());
        let v = self.project(hidden_states, &self.v_proj, self.v_bias.as_ref());

        if self.position_encoding == PositionEncoding::RoPE {
            self.apply_rope(&mut q, positions);
            self.apply_rope(&mut k, positions);
        }

        let attn_output = match self.attention_type {
            AttentionType::FlashAttention => {
                self.flash_attention(&q, &k, &v, seq_len, attention_mask)
            }
            _ => self.scaled_dot_product_attention(&q, &k, &v, seq_len, attention_mask),
        };

        self.project(&attn_output, &self.o_proj, self.o_bias.as_ref())
    }

    fn project(
        &self,
        input: &[Vec<f32>],
        weight: &[Vec<f32>],
        bias: Option<&Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let in_dim = weight.len();
        let out_dim = weight[0].len();

        let mut output = vec![vec![0.0f32; out_dim]; seq_len];

        for i in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = 0.0f64;
                for k in 0..in_dim.min(input[i].len()) {
                    sum += input[i][k] as f64 * weight[k][j] as f64;
                }
                output[i][j] = sum as f32;
                if let Some(b) = bias {
                    if j < b.len() {
                        output[i][j] += b[j];
                    }
                }
            }
        }

        output
    }

    fn apply_rope(&self, x: &mut [Vec<f32>], positions: &[usize]) {
        let dim = x[0].len();
        let head_dim = self.head_dim;
        let theta = self.rope_theta as f32;

        for (i, pos) in positions.iter().enumerate() {
            let pos = *pos as f32;
            for j in (0..head_dim.min(dim)).step_by(2) {
                let freq = theta.powf(2.0 * j as f32 / head_dim as f32);
                let angle = pos / freq;
                let cos = angle.cos();
                let sin = angle.sin();

                let x0 = x[i][j];
                let x1 = if j + 1 < dim { x[i][j + 1] } else { 0.0 };

                x[i][j] = x0 * cos - x1 * sin;
                if j + 1 < dim {
                    x[i][j + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    fn flash_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        seq_len: usize,
        attention_mask: Option<&Vec<Vec<bool>>>,
    ) -> Vec<Vec<f32>> {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_key_value_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let total_q_dim = num_heads * head_dim;
        let mut output = vec![vec![0.0f32; total_q_dim]; seq_len];

        let block_size = 32;

        for h in 0..num_heads {
            let kv_head = self.get_kv_head_index(h);
            
            debug_assert!(
                kv_head < num_kv_heads,
                "KV头索引 {} 超出范围 {}",
                kv_head,
                num_kv_heads
            );
            
            let use_causal = match self.attention_type {
                AttentionType::SlidingWindow { window_size: _ } => false,
                AttentionType::GQAWithSlidingWindow {
                    num_groups: _,
                    window_size: _,
                } => false,
                _ => true,
            };

            for block_start in (0..seq_len).step_by(block_size) {
                let block_end = (block_start + block_size).min(seq_len);
                let block_len = block_end - block_start;

                let mut max_vals = vec![f32::NEG_INFINITY; block_len];
                let mut sum_exp = vec![0.0f64; block_len];
                let mut block_output = vec![vec![0.0f32; head_dim]; block_len];

                for kv_start in 0..seq_len {
                    let kv_end = (kv_start + block_size).min(seq_len);
                    let kv_len = kv_end - kv_start;

                    let mut scores = vec![vec![0.0f32; kv_len]; block_len];

                    for i in 0..block_len {
                        let seq_idx = block_start + i;
                        let q_offset = h * head_dim;

                        for j in 0..kv_len {
                            let kv_idx = kv_start + j;
                            let k_offset = kv_head * head_dim;

                            let mut score = 0.0f64;
                            for d in 0..head_dim {
                                score += q[seq_idx][q_offset + d] as f64
                                    * k[kv_idx][k_offset + d] as f64;
                            }
                            scores[i][j] = (score as f32) * scale;
                        }
                    }

                    for i in 0..block_len {
                        let seq_idx = block_start + i;
                        for j in 0..kv_len {
                            let kv_idx = kv_start + j;

                            if use_causal && kv_idx > seq_idx {
                                scores[i][j] = f32::NEG_INFINITY;
                            }

                            if let Some(window) = self.sliding_window {
                                let distance = seq_idx.abs_diff(kv_idx);
                                if distance > window {
                                    scores[i][j] = f32::NEG_INFINITY;
                                }
                            }

                            if let Some(mask) = attention_mask {
                                if !mask[seq_idx][kv_idx] {
                                    scores[i][j] = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }

                    for i in 0..block_len {
                        for j in 0..kv_len {
                            let score = scores[i][j];
                            if score != f32::NEG_INFINITY && !score.is_nan() {
                                let current_max = max_vals[i];
                                if score > current_max {
                                    let scale_factor = (current_max - score).exp();
                                    for d in 0..head_dim {
                                        block_output[i][d] *= scale_factor;
                                    }
                                    sum_exp[i] *= scale_factor as f64;
                                    max_vals[i] = score;
                                }

                                let exp_val = (score - max_vals[i]).exp();
                                sum_exp[i] += exp_val as f64;

                                let v_offset = kv_head * head_dim;
                                for d in 0..head_dim {
                                    block_output[i][d] += exp_val * v[kv_start + j][v_offset + d];
                                }
                            }
                        }
                    }
                }

                for i in 0..block_len {
                    let seq_idx = block_start + i;
                    let q_offset = h * head_dim;
                    if sum_exp[i] > 1e-12 {
                        let inv_sum = 1.0 / sum_exp[i];
                        for d in 0..head_dim {
                            output[seq_idx][q_offset + d] = block_output[i][d] * inv_sum as f32;
                        }
                    }
                }
            }
        }

        output
    }

    fn scaled_dot_product_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        seq_len: usize,
        attention_mask: Option<&Vec<Vec<bool>>>,
    ) -> Vec<Vec<f32>> {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let total_q_dim = num_heads * head_dim;
        let mut output = vec![vec![0.0f32; total_q_dim]; seq_len];

        for h in 0..num_heads {
            let kv_head = self.get_kv_head_index(h);

            let use_causal = match self.attention_type {
                AttentionType::SlidingWindow { window_size: _ } => false,
                AttentionType::GQAWithSlidingWindow {
                    num_groups: _,
                    window_size: _,
                } => false,
                _ => true,
            };

            for i in 0..seq_len {
                let mut scores = vec![0.0f32; seq_len];
                let q_offset = h * head_dim;
                let k_offset = kv_head * head_dim;
                let v_offset = kv_head * head_dim;

                for j in 0..seq_len {
                    let mut score = 0.0f64;
                    for d in 0..head_dim {
                        score += q[i][q_offset + d] as f64 * k[j][k_offset + d] as f64;
                    }
                    scores[j] = (score as f32) * scale;
                }

                if use_causal {
                    for j in (i + 1)..seq_len {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }

                if let Some(window) = self.sliding_window {
                    for j in 0..seq_len {
                        let distance = i.abs_diff(j);
                        if distance > window {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                }

                if let Some(mask) = attention_mask {
                    for j in 0..seq_len {
                        if !mask[i][j] {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                }

                if let PositionEncoding::ALiBi = self.position_encoding {
                    let slope = 2.0f32.powi(-8 - h as i32);
                    for j in 0..seq_len {
                        if j < i {
                            scores[j] -= slope * (i - j) as f32;
                        }
                    }
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f64;
                let mut exp_values = vec![0.0f32; seq_len];

                for j in 0..seq_len {
                    if scores[j] != f32::NEG_INFINITY && !scores[j].is_nan() {
                        let exp_val = (scores[j] - max_score).exp();
                        exp_values[j] = exp_val;
                        exp_sum += exp_val as f64;
                    }
                }

                for d in 0..head_dim {
                    let mut weighted_sum = 0.0f64;
                    if exp_sum > 1e-12 {
                        for j in 0..seq_len {
                            if exp_values[j] > 0.0 {
                                let weight = exp_values[j] as f64 / exp_sum;
                                weighted_sum += weight * v[j][v_offset + d] as f64;
                            }
                        }
                    }
                    output[i][q_offset + d] = weighted_sum as f32;
                }
            }
        }

        output
    }

    fn apply_gradients(&mut self, gradients: &AttentionGradients, lr: f32) {
        let update_matrix = |a: &mut [Vec<f32>], b: &[Vec<f32>]| {
            for i in 0..a.len().min(b.len()) {
                for j in 0..a[i].len().min(b[i].len()) {
                    a[i][j] -= lr * b[i][j];
                }
            }
        };

        let update_vector = |a: &mut Option<Vec<f32>>, b: &Option<Vec<f32>>| {
            if let (Some(ref mut av), Some(ref bv)) = (a, b) {
                for i in 0..av.len().min(bv.len()) {
                    av[i] -= lr * bv[i];
                }
            }
        };

        update_matrix(&mut self.q_proj, &gradients.q_proj);
        update_matrix(&mut self.k_proj, &gradients.k_proj);
        update_matrix(&mut self.v_proj, &gradients.v_proj);
        update_matrix(&mut self.o_proj, &gradients.o_proj);

        update_vector(&mut self.q_bias, &gradients.q_bias);
        update_vector(&mut self.k_bias, &gradients.k_bias);
        update_vector(&mut self.v_bias, &gradients.v_bias);
        update_vector(&mut self.o_bias, &gradients.o_bias);
    }
}

// ============================================================================
// FeedForward 实现（保留原有实现）
// ============================================================================

impl FeedForward {
    fn new(params: &ModelParams, rng: &mut impl rand::Rng) -> Self {
        let use_gate = matches!(
            params.activation,
            ActivationFunction::SwiGLU | ActivationFunction::GEGLU
        );

        FeedForward {
            gate_proj: if use_gate {
                Some(Transformer::init_weights(
                    params.hidden_dim,
                    params.intermediate_dim,
                    0.02,
                    rng,
                ))
            } else {
                None
            },
            up_proj: Transformer::init_weights(
                params.hidden_dim,
                params.intermediate_dim,
                0.02,
                rng,
            ),
            down_proj: Transformer::init_weights(
                params.intermediate_dim,
                params.hidden_dim,
                0.02,
                rng,
            ),
            gate_bias: if use_gate && params.use_mlp_bias {
                Some(vec![0.0f32; params.intermediate_dim])
            } else {
                None
            },
            up_bias: if params.use_mlp_bias {
                Some(vec![0.0f32; params.intermediate_dim])
            } else {
                None
            },
            down_bias: if params.use_mlp_bias {
                Some(vec![0.0f32; params.hidden_dim])
            } else {
                None
            },
            activation: params.activation.clone(),
            use_bias: params.use_mlp_bias,
        }
    }

    fn forward(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let up = self.project(hidden_states, &self.up_proj, self.up_bias.as_ref());

        let gated: Vec<Vec<f32>> = if let Some(gate_proj) = &self.gate_proj {
            let gate = self.project(hidden_states, gate_proj, self.gate_bias.as_ref());
            gate.iter()
                .zip(up.iter())
                .map(|(g, u)| {
                    g.iter()
                        .zip(u.iter())
                        .map(|(&g_val, &u_val)| match self.activation {
                            ActivationFunction::SwiGLU => {
                                let silu_gate = g_val / (1.0 + (-g_val).exp());
                                silu_gate * u_val
                            }
                            ActivationFunction::GEGLU => {
                                let gelu_gate = Self::gelu(g_val);
                                gelu_gate * u_val
                            }
                            _ => u_val,
                        })
                        .collect()
                })
                .collect()
        } else {
            up.iter()
                .map(|x| x.iter().map(|&v| self.apply_activation(v)).collect())
                .collect()
        };

        self.project(&gated, &self.down_proj, self.down_bias.as_ref())
    }

    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation {
            ActivationFunction::SwiGLU | ActivationFunction::SiLU => x / (1.0 + (-x).exp()),
            ActivationFunction::GEGLU | ActivationFunction::GELU => Self::gelu(x),
            ActivationFunction::ReLU => x.max(0.0),
        }
    }

    fn project(
        &self,
        input: &[Vec<f32>],
        weight: &[Vec<f32>],
        bias: Option<&Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let in_dim = weight.len();
        let out_dim = weight[0].len();

        let mut output = vec![vec![0.0f32; out_dim]; seq_len];

        for i in 0..seq_len {
            for j in 0..out_dim {
                let mut sum = 0.0f64;
                for k in 0..in_dim.min(input[i].len()) {
                    sum += input[i][k] as f64 * weight[k][j] as f64;
                }
                output[i][j] = sum as f32;
                if let Some(b) = bias {
                    if j < b.len() {
                        output[i][j] += b[j];
                    }
                }
            }
        }

        output
    }

    fn apply_gradients(&mut self, gradients: &FFNGradients, lr: f32) {
        let update_matrix = |a: &mut [Vec<f32>], b: &[Vec<f32>]| {
            for i in 0..a.len().min(b.len()) {
                for j in 0..a[i].len().min(b[i].len()) {
                    a[i][j] -= lr * b[i][j];
                }
            }
        };

        let update_optional_matrix = |a: &mut Option<Vec<Vec<f32>>>, b: &Option<Vec<Vec<f32>>>| {
            if let (Some(ref mut am), Some(ref bm)) = (a, b) {
                update_matrix(am, bm);
            }
        };

        let update_vector = |a: &mut Option<Vec<f32>>, b: &Option<Vec<f32>>| {
            if let (Some(ref mut av), Some(ref bv)) = (a, b) {
                for i in 0..av.len().min(bv.len()) {
                    av[i] -= lr * bv[i];
                }
            }
        };

        update_optional_matrix(&mut self.gate_proj, &gradients.gate_proj);
        update_matrix(&mut self.up_proj, &gradients.up_proj);
        update_matrix(&mut self.down_proj, &gradients.down_proj);

        update_vector(&mut self.gate_bias, &gradients.gate_bias);
        update_vector(&mut self.up_bias, &gradients.up_bias);
        update_vector(&mut self.down_bias, &gradients.down_bias);
    }
}

// ============================================================================
// LayerNorm 实现（保留原有实现）
// ============================================================================

impl LayerNorm {
    fn new(dim: usize, eps: f64, norm_type: &NormalizationType) -> Self {
        match norm_type {
            NormalizationType::RMSNorm => LayerNorm {
                weight: vec![1.0f32; dim],
                bias: None,
                eps,
                norm_type: norm_type.clone(),
            },
            _ => LayerNorm {
                weight: vec![1.0f32; dim],
                bias: Some(vec![0.0f32; dim]),
                eps,
                norm_type: norm_type.clone(),
            },
        }
    }

    fn forward(&self, x: &mut [Vec<f32>]) {
        match self.norm_type {
            NormalizationType::RMSNorm => {
                for vec in x.iter_mut() {
                    let mean_square: f64 =
                        vec.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / vec.len() as f64;
                    let rms = ((mean_square + self.eps).sqrt()) as f32;
                    if rms > 1e-12 {
                        for (v, &w) in vec.iter_mut().zip(self.weight.iter()) {
                            *v = *v / rms * w;
                        }
                    }
                }
            }
            NormalizationType::LayerNorm
            | NormalizationType::PreLayerNorm
            | NormalizationType::PostLayerNorm => {
                for vec in x.iter_mut() {
                    let mean: f64 = vec.iter().map(|&v| v as f64).sum::<f64>() / vec.len() as f64;
                    let variance: f64 = vec
                        .iter()
                        .map(|&v| {
                            let diff = v as f64 - mean;
                            diff * diff
                        })
                        .sum::<f64>()
                        / vec.len() as f64;
                    let std = ((variance + self.eps).sqrt()) as f32;

                    if std > 1e-12 {
                        for (i, v) in vec.iter_mut().enumerate() {
                            *v = ((*v as f64 - mean) / std as f64) as f32 * self.weight[i];
                            if let Some(bias) = &self.bias {
                                *v += bias[i];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use tempfile::tempdir;

    fn create_test_model() -> Transformer {
        let config = ModelConfig {
            num_layers: 2,
            hidden_dim: 64,
            num_heads: 4,
            activation: ActivationFunction::GELU,
            position_encoding: PositionEncoding::RoPE,
            normalization: NormalizationType::RMSNorm,
            attention: AttentionType::MHA,
            use_qkv_bias: true,
            use_mlp_bias: true,
            tied_embedding: true,
            dropout: 0.0,
            stochastic_depth: None,
            intermediate_dim: Some(256),
            sliding_window: None,
            num_key_value_heads: None,
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-6),
            max_position_embeddings: 128,
            vocab_size_override: None,
        };

        let params = ModelParams::from_config(&config, 1000);
        Transformer::new(params)
    }

    #[test]
    fn test_json_serialization() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");
        
        model.save_as_json(&path).unwrap();
        assert!(path.exists());
        
        let loaded = Transformer::load_from_json(&path).unwrap();
        assert_eq!(loaded.params.num_layers, model.params.num_layers);
    }

    #[test]
    fn test_msgpack_serialization() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.msgpack");
        
        let config = ModelSaveConfig {
            format: ModelSerializationFormat::MessagePack,
            ..Default::default()
        };
        model.save_with_config(&path, &config).unwrap();
        assert!(path.exists());
        
        let loaded = Transformer::load_from_msgpack(&path).unwrap();
        assert_eq!(loaded.params.num_layers, model.params.num_layers);
    }

    #[test]
    fn test_compressed_serialization() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.cmp");
        
        model.save_as_compressed_msgpack(&path, &ModelSaveConfig::default()).unwrap();
        assert!(path.exists());
        
        let loaded = Transformer::load_from_compressed_msgpack(&path).unwrap();
        assert_eq!(loaded.params.num_layers, model.params.num_layers);
        
        // 验证压缩后文件更小
        let json_path = dir.path().join("model.json");
        model.save_as_json(&json_path).unwrap();
        let json_size = fs::metadata(&json_path).unwrap().len();
        let cmp_size = fs::metadata(&path).unwrap().len();
        assert!(cmp_size < json_size);
    }

    #[test]
    fn test_sharded_serialization() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("sharded_model");
        
        let config = ModelSaveConfig {
            format: ModelSerializationFormat::Sharded,
            shard_size_bytes: 1024,
            ..Default::default()
        };
        model.save_with_config(&path, &config).unwrap();
        
        let manifest_path = dir.path().join("sharded_model.manifest.json");
        assert!(manifest_path.exists());
        
        let loaded = Transformer::load_from_sharded(&manifest_path).unwrap();
        assert_eq!(loaded.params.num_layers, model.params.num_layers);
    }

    #[test]
    fn test_auto_detect_format() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        
        // 测试 JSON 自动检测
        let json_path = dir.path().join("model.json");
        model.save_as_json(&json_path).unwrap();
        let loaded_json = Transformer::load(&json_path).unwrap();
        assert_eq!(loaded_json.params.num_layers, model.params.num_layers);
        
        // 测试 MessagePack 自动检测
        let msgpack_path = dir.path().join("model.msgpack");
        let config = ModelSaveConfig {
            format: ModelSerializationFormat::MessagePack,
            ..Default::default()
        };
        model.save_with_config(&msgpack_path, &config).unwrap();
        let loaded_msgpack = Transformer::load(&msgpack_path).unwrap();
        assert_eq!(loaded_msgpack.params.num_layers, model.params.num_layers);
    }

    #[test]
    fn test_model_creation() {
        let model = create_test_model();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.embedding.len(), 1000);
        assert!(model.final_norm.bias.is_none());
    }

    #[test]
    fn test_forward_pass() -> Result<()> {
        let model = create_test_model();
        let input_ids = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&input_ids, false)?;

        assert_eq!(logits.len(), 5);
        assert_eq!(logits[0].len(), 1000);
        Ok(())
    }
}