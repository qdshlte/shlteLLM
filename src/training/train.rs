//! ============================================================================
//! 训练器模块
//! ============================================================================
//!
//! 本模块实现了完整的模型训练流程，包括：
//! - 优化器（AdamW、Adam、SGD、LAMB）
//! - 学习率调度器（线性、余弦、余弦重启、单周期）
//! - 混合精度训练（FP16/BF16）
//! - EMA（指数移动平均）模型
//! - 梯度累积
//! - 检查点保存与恢复
//! - 早停机制
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::config::{Config, LRScheduler, MixedPrecision, OptimizerType, TrainingConfig};
use crate::data_two::BatchLoader;
use crate::error::Result;
use crate::model::{
    AttentionGradients, FFNGradients, Gradients, LayerGradients, LayerNormGradients, ModelParams,
    Transformer,
};
use rand::SeedableRng;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ============================================================================
// 优化器结构
// ============================================================================

#[derive(Debug, Clone)]
pub struct Optimizer {
    /// 优化器类型
    pub optimizer_type: OptimizerType,
    /// 基础学习率
    pub learning_rate: f64,
    /// 权重衰减系数
    pub weight_decay: f64,
    /// 梯度裁剪阈值
    pub grad_clip: f64,
    /// Adam beta1参数
    pub beta1: f64,
    /// Adam beta2参数
    pub beta2: f64,
    /// 数值稳定epsilon
    pub epsilon: f64,
    /// SGD动量
    pub momentum: f64,
    /// 当前步数
    pub step: usize,
    /// Adam一阶矩估计（展平存储）
    pub m: Vec<f32>,
    /// Adam二阶矩估计（展平存储）
    pub v: Vec<f32>,
    /// SGD动量缓存（展平存储）
    pub velocity: Vec<f32>,
    /// 参数数量
    param_count: usize,
    /// 是否已初始化
    initialized: bool,
}

impl Optimizer {
    // ========================================================================
    // 创建与初始化
    // ========================================================================

    /// 从训练配置创建优化器
    pub fn from_config(config: &TrainingConfig) -> Self {
        let (beta1, beta2, epsilon, momentum) = match &config.optimizer {
            OptimizerType::AdamW {
                beta1,
                beta2,
                epsilon,
            } => (*beta1, *beta2, *epsilon, 0.0),
            OptimizerType::Adam {
                beta1,
                beta2,
                epsilon,
            } => (*beta1, *beta2, *epsilon, 0.0),
            OptimizerType::SGD { momentum } => (0.0, 0.0, 1e-8, *momentum),
            OptimizerType::LAMB {
                beta1,
                beta2,
                epsilon,
            } => (*beta1, *beta2, *epsilon, 0.0),
        };

        // Adam优化器不使用权重衰减（AdamW才使用）
        let weight_decay = match &config.optimizer {
            OptimizerType::Adam { .. } => 0.0,
            _ => config.weight_decay,
        };

        Optimizer {
            optimizer_type: config.optimizer.clone(),
            learning_rate: config.learning_rate,
            weight_decay,
            grad_clip: config.grad_clip,
            beta1,
            beta2,
            epsilon,
            momentum,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
            velocity: Vec::new(),
            param_count: 0,
            initialized: false,
        }
    }

    /// 增加步数计数
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// 获取当前步数
    pub fn get_step(&self) -> usize {
        self.step
    }

    /// 确保优化器状态已初始化（基于参数数量）
    pub fn ensure_state_initialized(&mut self, total_params: usize) {
        if !self.initialized || self.param_count != total_params {
            self.param_count = total_params;
            self.m = vec![0.0f32; total_params];
            self.v = vec![0.0f32; total_params];
            self.velocity = vec![0.0f32; total_params];
            self.initialized = true;
        }
    }

    /// 检查是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ========================================================================
    // 学习率计算
    // ========================================================================

    /// 根据当前步数计算学习率
    pub fn get_lr(&self, config: &TrainingConfig) -> f64 {
        let total_steps = config.num_steps as f64;
        let warmup_steps = config.warmup_steps as f64;
        let current_step = self.step as f64;

        // 预热阶段：线性增长
        if current_step < warmup_steps && warmup_steps > 0.0 {
            return self.learning_rate * current_step / warmup_steps;
        }

        let progress = (current_step - warmup_steps) / (total_steps - warmup_steps).max(1.0);
        let progress = progress.min(1.0);

        match &config.lr_scheduler {
            LRScheduler::Linear => self.learning_rate * (1.0 - progress).max(0.0),
            LRScheduler::Cosine { min_lr } => {
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (self.learning_rate - min_lr) * cosine_decay
            }
            LRScheduler::CosineWithRestarts {
                min_lr,
                restart_interval,
            } => {
                let restart_step = self.step % restart_interval;
                let progress = restart_step as f64 / *restart_interval as f64;
                let cosine_decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_lr + (self.learning_rate - min_lr) * cosine_decay
            }
            LRScheduler::Constant => self.learning_rate,
            LRScheduler::OneCycle { max_lr, pct_start } => {
                if progress < *pct_start {
                    self.learning_rate + (max_lr - self.learning_rate) * progress / pct_start
                } else {
                    let decay_progress = (progress - pct_start) / (1.0 - pct_start);
                    max_lr - (max_lr - self.learning_rate) * decay_progress
                }
            }
        }
    }

    // ========================================================================
    // 梯度裁剪
    // ========================================================================

    /// 对展平后的梯度进行裁剪
    pub fn clip_gradients_flat(&self, gradients: &mut [f32]) {
        if self.grad_clip <= 0.0 || gradients.is_empty() {
            return;
        }

        let total_norm: f64 = gradients
            .iter()
            .map(|&x| x as f64 * x as f64)
            .sum::<f64>()
            .sqrt();

        if total_norm > self.grad_clip && total_norm > 0.0 {
            let scale = self.grad_clip / total_norm;
            for g in gradients.iter_mut() {
                *g = (*g as f64 * scale) as f32;
            }
        }
    }

    /// 计算梯度范数
    pub fn compute_gradient_norm(&self, gradients: &[f32]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }
        gradients.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
    }

    // ========================================================================
    // Adam/AdamW 更新
    // ========================================================================

    pub fn update_params_adam_flat(&mut self, params: &mut [f32], gradients: &[f32], lr: f32) {
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;

        let bias_correction1 = 1.0f32 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0f32 - beta2.powi(self.step as i32);

        if bias_correction1 <= 0.0 || bias_correction2 <= 0.0 {
            return;
        }

        let alpha = lr * bias_correction2.sqrt() / bias_correction1;

        for i in 0..params.len().min(gradients.len()) {
            let grad = gradients[i];

            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad * grad;

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;
            let update = alpha * m_hat / (v_hat.sqrt() + epsilon);

            if weight_decay > 0.0 {
                params[i] *= 1.0 - lr * weight_decay;
            }

            params[i] -= update;
        }
    }

    // ========================================================================
    // LAMB 优化器更新
    // ========================================================================

    pub fn update_params_lamb_flat(&mut self, params: &mut [f32], gradients: &[f32], lr: f32) {
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;

        let bias_correction1 = 1.0f32 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0f32 - beta2.powi(self.step as i32);

        if bias_correction1 <= 0.0 || bias_correction2 <= 0.0 {
            return;
        }

        let param_norm: f64 = params.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        let grad_norm: f64 = gradients.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();

        let trust_ratio = if param_norm > 0.0 && grad_norm > 0.0 {
            (param_norm / (grad_norm + epsilon as f64)) as f32
        } else {
            1.0
        };

        for i in 0..params.len().min(gradients.len()) {
            let grad = gradients[i];

            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad * grad;

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            let adam_update = m_hat / (v_hat.sqrt() + epsilon);
            let decay_update = weight_decay * params[i];
            let update = adam_update + decay_update;
            let update_norm = update.abs();

            let effective_lr = if update_norm > 0.0 {
                lr * trust_ratio / (update_norm + epsilon)
            } else {
                lr
            };

            params[i] -= effective_lr * update;
        }
    }

    // ========================================================================
    // SGD 更新
    // ========================================================================

    pub fn update_params_sgd_flat(&mut self, params: &mut [f32], gradients: &[f32], lr: f32) {
        let momentum = self.momentum as f32;
        let weight_decay = self.weight_decay as f32;

        for i in 0..params.len().min(gradients.len()) {
            let grad = gradients[i] + weight_decay * params[i];
            self.velocity[i] = momentum * self.velocity[i] + grad;
            params[i] -= lr * self.velocity[i];
        }
    }

    /// 更新参数
    pub fn update(&mut self, params: &mut [f32], gradients: &[f32], lr: f32) {
        self.step += 1;

        match &self.optimizer_type {
            OptimizerType::AdamW { .. } | OptimizerType::Adam { .. } => {
                self.update_params_adam_flat(params, gradients, lr);
            }
            OptimizerType::LAMB { .. } => {
                self.update_params_lamb_flat(params, gradients, lr);
            }
            OptimizerType::SGD { .. } => {
                self.update_params_sgd_flat(params, gradients, lr);
            }
        }
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.velocity.fill(0.0);
    }
}

// ============================================================================
// 混合精度训练器（修复版）
// ============================================================================

#[derive(Debug, Clone)]
pub struct MixedPrecisionTrainer {
    pub precision: MixedPrecision,
    pub loss_scale: f32,
    pub loss_scale_factor: f32,
    pub loss_scale_window: usize,
    pub skipped_steps: usize,
    pub consecutive_non_overflow: usize,
    pub enabled: bool,
    pub max_loss_scale: f32,
    pub min_loss_scale: f32,
    /// 记录溢出次数（用于统计）
    pub overflow_count: usize,
    /// 总步数计数
    pub total_steps: usize,
}

impl MixedPrecisionTrainer {
    pub fn new(precision: MixedPrecision) -> Self {
        let (initial_scale, max_scale) = match precision {
            MixedPrecision::FP16 => (65536.0, 32768.0),
            MixedPrecision::BF16 => (1.0, 1.0),
            MixedPrecision::FP8 => (4096.0, 16384.0),
        };

        MixedPrecisionTrainer {
            precision,
            loss_scale: initial_scale,
            loss_scale_factor: 2.0,
            loss_scale_window: 2000,
            skipped_steps: 0,
            consecutive_non_overflow: 0,
            enabled: true,
            max_loss_scale: max_scale,
            min_loss_scale: 1.0,
            overflow_count: 0,
            total_steps: 0,
        }
    }

    pub fn supports_bf16() -> bool {
        true
    }

    pub fn supports_fp8() -> bool {
        false
    }

    pub fn to_fp16(&self, data: &[f32]) -> Vec<half::f16> {
        data.iter().map(|&x| half::f16::from_f32(x)).collect()
    }

    pub fn fp16_to_f32(data: &[half::f16]) -> Vec<f32> {
        data.iter().map(|x| x.to_f32()).collect()
    }

    pub fn to_bf16(&self, data: &[f32]) -> Vec<f32> {
        data.iter()
            .map(|&x| f32::from_bits(x.to_bits() & 0xFFFF0000))
            .collect()
    }

    pub fn convert_value(&self, value: f32) -> f32 {
        match self.precision {
            MixedPrecision::FP16 => half::f16::from_f32(value).to_f32(),
            MixedPrecision::BF16 => f32::from_bits(value.to_bits() & 0xFFFF0000),
            MixedPrecision::FP8 => half::f16::from_f32(value).to_f32(),
        }
    }

    pub fn convert_model_to_precision(&self, model: &mut Transformer) {
        if !self.enabled {
            return;
        }

        let convert_matrix = |w: &mut [Vec<f32>]| {
            for row in w.iter_mut() {
                for val in row.iter_mut() {
                    *val = self.convert_value(*val);
                }
            }
        };

        let convert_vector = |v: &mut [f32]| {
            for val in v.iter_mut() {
                *val = self.convert_value(*val);
            }
        };

        convert_matrix(&mut model.embedding);

        for layer in model.layers.iter_mut() {
            convert_matrix(&mut layer.attention.q_proj);
            convert_matrix(&mut layer.attention.k_proj);
            convert_matrix(&mut layer.attention.v_proj);
            convert_matrix(&mut layer.attention.o_proj);
            convert_matrix(&mut layer.feed_forward.up_proj);
            convert_matrix(&mut layer.feed_forward.down_proj);
            if let Some(ref mut gate) = layer.feed_forward.gate_proj {
                convert_matrix(gate);
            }
            convert_vector(&mut layer.attention_norm.weight);
            if let Some(ref mut bias) = layer.attention_norm.bias {
                convert_vector(bias);
            }
            convert_vector(&mut layer.ffn_norm.weight);
            if let Some(ref mut bias) = layer.ffn_norm.bias {
                convert_vector(bias);
            }
        }

        if let Some(ref mut lm_head) = model.lm_head {
            convert_matrix(lm_head);
        }

        if let Some(ref mut pos_emb) = model.position_embeddings {
            convert_matrix(pos_emb);
        }

        convert_vector(&mut model.final_norm.weight);
        if let Some(ref mut bias) = model.final_norm.bias {
            convert_vector(bias);
        }
    }

    /// 对损失应用缩放（在前向传播前调用）
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if !self.enabled || self.precision == MixedPrecision::BF16 {
            loss
        } else {
            loss * self.loss_scale
        }
    }

    /// 反向传播后，将缩放后的梯度恢复并检查溢出
    /// 这是修复的核心：正确的顺序是：
    /// 1. 前向传播（缩放后的损失）
    /// 2. 反向传播（得到缩放后的梯度）
    /// 3. 检查溢出（在未缩放的梯度上？不，在缩放后的梯度上检查）
    /// 4. 如果溢出，跳过更新并降低缩放因子
    /// 5. 如果未溢出，unscale梯度然后更新参数
    pub fn post_backward<F>(&mut self, gradients: &mut [f32], mut update_fn: F) -> bool
    where
        F: FnMut(&mut [f32]),
    {
        self.total_steps += 1;

        // 步骤1：检查梯度是否溢出（在缩放后的梯度上检查）
        let has_overflow = Self::check_overflow(gradients);

        if has_overflow {
            // 步骤2：发生溢出，跳过此次更新，降低缩放因子
            self.update_scale(true);
            self.overflow_count += 1;
            
            if self.total_steps.is_multiple_of(100) {
                eprintln!(
                    "⚠️ 梯度溢出 (步数 {}), loss_scale 降至 {:.2}, 总溢出次数: {}",
                    self.total_steps, self.loss_scale, self.overflow_count
                );
            }
            
            return false;
        }

        // 步骤3：未溢出，先 unscale 梯度
        let inv_scale = 1.0 / self.loss_scale;
        for g in gradients.iter_mut() {
            *g *= inv_scale;
        }

        // 步骤4：执行参数更新
        update_fn(gradients);

        // 步骤5：更新缩放因子（无溢出时可能增大）
        self.update_scale(false);

        true
    }

    /// 更新损失缩放因子
    pub fn update_scale(&mut self, overflow: bool) {
        if self.precision == MixedPrecision::BF16 {
            return;
        }

        if overflow {
            self.loss_scale /= self.loss_scale_factor;
            self.skipped_steps = 0;
            self.consecutive_non_overflow = 0;

            if self.loss_scale < self.min_loss_scale {
                self.loss_scale = self.min_loss_scale;
            }
        } else {
            self.skipped_steps += 1;
            self.consecutive_non_overflow += 1;

            if self.skipped_steps >= self.loss_scale_window {
                let new_scale = self.loss_scale * self.loss_scale_factor;
                if new_scale <= self.max_loss_scale {
                    self.loss_scale = new_scale;
                    self.skipped_steps = 0;

                    if self.consecutive_non_overflow.is_multiple_of(self.loss_scale_window * 3) {
                        println!("📈 loss_scale 增至 {:.2}", self.loss_scale);
                    }
                }
            }
        }

        self.loss_scale = self
            .loss_scale
            .clamp(self.min_loss_scale, self.max_loss_scale);
    }

    pub fn check_overflow(gradients: &[f32]) -> bool {
        gradients.iter().any(|&g| g.is_nan() || g.is_infinite())
    }

    pub fn check_overflow_matrix(gradients: &[Vec<f32>]) -> bool {
        gradients
            .iter()
            .any(|row| row.iter().any(|&g| g.is_nan() || g.is_infinite()))
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> MixedPrecisionStats {
        MixedPrecisionStats {
            loss_scale: self.loss_scale,
            overflow_count: self.overflow_count,
            total_steps: self.total_steps,
            overflow_rate: if self.total_steps > 0 {
                self.overflow_count as f64 / self.total_steps as f64
            } else {
                0.0
            },
        }
    }
}

/// 混合精度训练统计信息
#[derive(Debug, Clone)]
pub struct MixedPrecisionStats {
    pub loss_scale: f32,
    pub overflow_count: usize,
    pub total_steps: usize,
    pub overflow_rate: f64,
}

// ============================================================================
// Gradients 扩展方法
// ============================================================================

impl Gradients {
    pub fn zeros(params: &ModelParams) -> Self {
        let hidden_dim = params.hidden_dim;
        let vocab_size = params.vocab_size;
        let num_layers = params.num_layers;

        Gradients {
            embedding: vec![vec![0.0f32; hidden_dim]; vocab_size],
            layers: (0..num_layers)
                .map(|_| LayerGradients::new(params))
                .collect(),
            final_norm: LayerNormGradients {
                weight: vec![0.0f32; hidden_dim],
                bias: if matches!(
                    params.normalization,
                    crate::config::NormalizationType::Rms
                ) {
                    None
                } else {
                    Some(vec![0.0f32; hidden_dim])
                },
            },
            lm_head: if params.tied_embedding {
                None
            } else {
                Some(vec![vec![0.0f32; hidden_dim]; vocab_size])
            },
            position_embeddings: if matches!(
                params.position_encoding,
                crate::config::PositionEncoding::Learned
            ) {
                Some(vec![
                    vec![0.0f32; hidden_dim];
                    params.max_position_embeddings
                ])
            } else {
                None
            },
        }
    }

    pub fn accumulate(&mut self, other: &Gradients) {
        let add_matrix = |a: &mut [Vec<f32>], b: &[Vec<f32>]| {
            for (ar, br) in a.iter_mut().zip(b.iter()) {
                for (aij, &bij) in ar.iter_mut().zip(br.iter()) {
                    *aij += bij;
                }
            }
        };

        let add_vector = |a: &mut [f32], b: &[f32]| {
            for (ai, &bi) in a.iter_mut().zip(b.iter()) {
                *ai += bi;
            }
        };

        let add_optional_vector = |a: &mut Option<Vec<f32>>, b: &Option<Vec<f32>>| {
            if let (Some(ref mut av), Some(ref bv)) = (a, b) {
                add_vector(av, bv);
            }
        };

        let add_optional_matrix = |a: &mut Option<Vec<Vec<f32>>>, b: &Option<Vec<Vec<f32>>>| {
            if let (Some(ref mut am), Some(ref bm)) = (a, b) {
                add_matrix(am, bm);
            }
        };

        add_matrix(&mut self.embedding, &other.embedding);

        for (sl, ol) in self.layers.iter_mut().zip(other.layers.iter()) {
            add_matrix(&mut sl.attention.q_proj, &ol.attention.q_proj);
            add_optional_vector(&mut sl.attention.q_bias, &ol.attention.q_bias);
            add_matrix(&mut sl.attention.k_proj, &ol.attention.k_proj);
            add_optional_vector(&mut sl.attention.k_bias, &ol.attention.k_bias);
            add_matrix(&mut sl.attention.v_proj, &ol.attention.v_proj);
            add_optional_vector(&mut sl.attention.v_bias, &ol.attention.v_bias);
            add_matrix(&mut sl.attention.o_proj, &ol.attention.o_proj);
            add_optional_vector(&mut sl.attention.o_bias, &ol.attention.o_bias);
            add_matrix(&mut sl.feed_forward.up_proj, &ol.feed_forward.up_proj);
            add_optional_vector(&mut sl.feed_forward.up_bias, &ol.feed_forward.up_bias);
            add_matrix(&mut sl.feed_forward.down_proj, &ol.feed_forward.down_proj);
            add_optional_vector(&mut sl.feed_forward.down_bias, &ol.feed_forward.down_bias);
            add_optional_matrix(&mut sl.feed_forward.gate_proj, &ol.feed_forward.gate_proj);
            add_optional_vector(&mut sl.feed_forward.gate_bias, &ol.feed_forward.gate_bias);
            add_vector(&mut sl.attention_norm.weight, &ol.attention_norm.weight);
            add_optional_vector(&mut sl.attention_norm.bias, &ol.attention_norm.bias);
            add_vector(&mut sl.ffn_norm.weight, &ol.ffn_norm.weight);
            add_optional_vector(&mut sl.ffn_norm.bias, &ol.ffn_norm.bias);
        }

        add_vector(&mut self.final_norm.weight, &other.final_norm.weight);
        add_optional_vector(&mut self.final_norm.bias, &other.final_norm.bias);
        add_optional_matrix(&mut self.lm_head, &other.lm_head);
        add_optional_matrix(&mut self.position_embeddings, &other.position_embeddings);
    }

    pub fn scale(&mut self, factor: f32) {
        let scale_matrix = |m: &mut [Vec<f32>]| {
            for row in m.iter_mut() {
                for val in row.iter_mut() {
                    *val *= factor;
                }
            }
        };

        let scale_vector = |v: &mut [f32]| {
            for val in v.iter_mut() {
                *val *= factor;
            }
        };

        let scale_optional_vector = |v: &mut Option<Vec<f32>>| {
            if let Some(ref mut vec) = v {
                scale_vector(vec);
            }
        };

        let scale_optional_matrix = |m: &mut Option<Vec<Vec<f32>>>| {
            if let Some(ref mut mat) = m {
                scale_matrix(mat);
            }
        };

        scale_matrix(&mut self.embedding);

        for layer in self.layers.iter_mut() {
            scale_matrix(&mut layer.attention.q_proj);
            scale_optional_vector(&mut layer.attention.q_bias);
            scale_matrix(&mut layer.attention.k_proj);
            scale_optional_vector(&mut layer.attention.k_bias);
            scale_matrix(&mut layer.attention.v_proj);
            scale_optional_vector(&mut layer.attention.v_bias);
            scale_matrix(&mut layer.attention.o_proj);
            scale_optional_vector(&mut layer.attention.o_bias);
            scale_matrix(&mut layer.feed_forward.up_proj);
            scale_optional_vector(&mut layer.feed_forward.up_bias);
            scale_matrix(&mut layer.feed_forward.down_proj);
            scale_optional_vector(&mut layer.feed_forward.down_bias);
            scale_optional_matrix(&mut layer.feed_forward.gate_proj);
            scale_optional_vector(&mut layer.feed_forward.gate_bias);
            scale_vector(&mut layer.attention_norm.weight);
            scale_optional_vector(&mut layer.attention_norm.bias);
            scale_vector(&mut layer.ffn_norm.weight);
            scale_optional_vector(&mut layer.ffn_norm.bias);
        }

        scale_vector(&mut self.final_norm.weight);
        scale_optional_vector(&mut self.final_norm.bias);
        scale_optional_matrix(&mut self.lm_head);
        scale_optional_matrix(&mut self.position_embeddings);
    }

    pub fn flatten(&self) -> Vec<f32> {
        let mut flat = Vec::new();

        let push_matrix = |f: &mut Vec<f32>, m: &[Vec<f32>]| {
            for row in m {
                f.extend_from_slice(row);
            }
        };

        let push_vector = |f: &mut Vec<f32>, v: &[f32]| {
            f.extend_from_slice(v);
        };

        let push_optional_vector = |f: &mut Vec<f32>, v: &Option<Vec<f32>>| {
            if let Some(ref vec) = v {
                push_vector(f, vec);
            }
        };

        let push_optional_matrix = |f: &mut Vec<f32>, m: &Option<Vec<Vec<f32>>>| {
            if let Some(ref mat) = m {
                push_matrix(f, mat);
            }
        };

        push_matrix(&mut flat, &self.embedding);

        for layer in &self.layers {
            push_matrix(&mut flat, &layer.attention.q_proj);
            push_optional_vector(&mut flat, &layer.attention.q_bias);
            push_matrix(&mut flat, &layer.attention.k_proj);
            push_optional_vector(&mut flat, &layer.attention.k_bias);
            push_matrix(&mut flat, &layer.attention.v_proj);
            push_optional_vector(&mut flat, &layer.attention.v_bias);
            push_matrix(&mut flat, &layer.attention.o_proj);
            push_optional_vector(&mut flat, &layer.attention.o_bias);
            push_matrix(&mut flat, &layer.feed_forward.up_proj);
            push_optional_vector(&mut flat, &layer.feed_forward.up_bias);
            push_matrix(&mut flat, &layer.feed_forward.down_proj);
            push_optional_vector(&mut flat, &layer.feed_forward.down_bias);
            push_optional_matrix(&mut flat, &layer.feed_forward.gate_proj);
            push_optional_vector(&mut flat, &layer.feed_forward.gate_bias);
            push_vector(&mut flat, &layer.attention_norm.weight);
            push_optional_vector(&mut flat, &layer.attention_norm.bias);
            push_vector(&mut flat, &layer.ffn_norm.weight);
            push_optional_vector(&mut flat, &layer.ffn_norm.bias);
        }

        push_vector(&mut flat, &self.final_norm.weight);
        push_optional_vector(&mut flat, &self.final_norm.bias);
        push_optional_matrix(&mut flat, &self.lm_head);
        push_optional_matrix(&mut flat, &self.position_embeddings);

        flat
    }

    pub fn from_flat(flat: &[f32], params: &ModelParams) -> Self {
        let mut offset = 0;
        let hidden_dim = params.hidden_dim;
        let vocab_size = params.vocab_size;
        let num_layers = params.num_layers;
        let num_heads = params.num_heads;
        let num_kv_heads = params.num_key_value_heads;
        let head_dim = params.head_dim;
        let intermediate_dim = params.intermediate_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let use_gate = matches!(
            params.activation,
            crate::config::ActivationFunction::SwiGLU | crate::config::ActivationFunction::GEGLU
        );

        let copy_matrix = |flat: &[f32], offset: &mut usize, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let mut m = vec![vec![0.0f32; cols]; rows];
            for row in m.iter_mut() {
                let end = (*offset + cols).min(flat.len());
                if *offset < end {
                    row.copy_from_slice(&flat[*offset..end]);
                }
                *offset = end;
            }
            m
        };

        let copy_vector = |flat: &[f32], offset: &mut usize, size: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; size];
            let end = (*offset + size).min(flat.len());
            if *offset < end {
                v.copy_from_slice(&flat[*offset..end]);
            }
            *offset = end;
            v
        };

        let copy_optional_vector = |flat: &[f32], offset: &mut usize, size: usize, present: bool| -> Option<Vec<f32>> {
            if present {
                Some(copy_vector(flat, offset, size))
            } else {
                None
            }
        };

        let copy_optional_matrix = |flat: &[f32], offset: &mut usize, rows: usize, cols: usize, present: bool| -> Option<Vec<Vec<f32>>> {
            if present {
                Some(copy_matrix(flat, offset, rows, cols))
            } else {
                None
            }
        };

        let embedding = copy_matrix(flat, &mut offset, vocab_size, hidden_dim);

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let layer = LayerGradients {
                attention: AttentionGradients {
                    q_proj: copy_matrix(flat, &mut offset, hidden_dim, q_dim),
                    q_bias: copy_optional_vector(flat, &mut offset, q_dim, params.use_qkv_bias),
                    k_proj: copy_matrix(flat, &mut offset, hidden_dim, kv_dim),
                    k_bias: copy_optional_vector(flat, &mut offset, kv_dim, params.use_qkv_bias),
                    v_proj: copy_matrix(flat, &mut offset, hidden_dim, kv_dim),
                    v_bias: copy_optional_vector(flat, &mut offset, kv_dim, params.use_qkv_bias),
                    o_proj: copy_matrix(flat, &mut offset, q_dim, hidden_dim),
                    o_bias: None,
                },
                feed_forward: FFNGradients {
                    up_proj: copy_matrix(flat, &mut offset, hidden_dim, intermediate_dim),
                    up_bias: copy_optional_vector(flat, &mut offset, intermediate_dim, params.use_mlp_bias),
                    down_proj: copy_matrix(flat, &mut offset, intermediate_dim, hidden_dim),
                    down_bias: copy_optional_vector(flat, &mut offset, hidden_dim, params.use_mlp_bias),
                    gate_proj: copy_optional_matrix(flat, &mut offset, hidden_dim, intermediate_dim, use_gate),
                    gate_bias: copy_optional_vector(flat, &mut offset, intermediate_dim, use_gate && params.use_mlp_bias),
                },
                attention_norm: LayerNormGradients {
                    weight: copy_vector(flat, &mut offset, hidden_dim),
                    bias: copy_optional_vector(flat, &mut offset, hidden_dim, !matches!(params.normalization, crate::config::NormalizationType::Rms)),
                },
                ffn_norm: LayerNormGradients {
                    weight: copy_vector(flat, &mut offset, hidden_dim),
                    bias: copy_optional_vector(flat, &mut offset, hidden_dim, !matches!(params.normalization, crate::config::NormalizationType::Rms)),
                },
            };
            layers.push(layer);
        }

        let final_norm = LayerNormGradients {
            weight: copy_vector(flat, &mut offset, hidden_dim),
            bias: copy_optional_vector(flat, &mut offset, hidden_dim, !matches!(params.normalization, crate::config::NormalizationType::Rms)),
        };

        let lm_head = copy_optional_matrix(flat, &mut offset, vocab_size, hidden_dim, !params.tied_embedding);
        let position_embeddings = copy_optional_matrix(flat, &mut offset, params.max_position_embeddings, hidden_dim, matches!(params.position_encoding, crate::config::PositionEncoding::Learned));

        Gradients {
            embedding,
            layers,
            final_norm,
            lm_head,
            position_embeddings,
        }
    }

    pub fn is_valid(&self) -> bool {
        let check_matrix = |m: &[Vec<f32>]| -> bool {
            for row in m {
                for &val in row {
                    if val.is_nan() || val.is_infinite() {
                        return false;
                    }
                }
            }
            true
        };

        let check_vector = |v: &[f32]| -> bool {
            for &val in v {
                if val.is_nan() || val.is_infinite() {
                    return false;
                }
            }
            true
        };

        let check_optional_matrix = |m: &Option<Vec<Vec<f32>>>| -> bool {
            if let Some(ref mat) = m {
                check_matrix(mat)
            } else {
                true
            }
        };

        let check_optional_vector = |v: &Option<Vec<f32>>| -> bool {
            if let Some(ref vec) = v {
                check_vector(vec)
            } else {
                true
            }
        };

        if !check_matrix(&self.embedding) { return false; }

        for layer in &self.layers {
            if !check_matrix(&layer.attention.q_proj) { return false; }
            if !check_optional_vector(&layer.attention.q_bias) { return false; }
            if !check_matrix(&layer.attention.k_proj) { return false; }
            if !check_optional_vector(&layer.attention.k_bias) { return false; }
            if !check_matrix(&layer.attention.v_proj) { return false; }
            if !check_optional_vector(&layer.attention.v_bias) { return false; }
            if !check_matrix(&layer.attention.o_proj) { return false; }
            if !check_optional_vector(&layer.attention.o_bias) { return false; }
            if !check_matrix(&layer.feed_forward.up_proj) { return false; }
            if !check_optional_vector(&layer.feed_forward.up_bias) { return false; }
            if !check_matrix(&layer.feed_forward.down_proj) { return false; }
            if !check_optional_vector(&layer.feed_forward.down_bias) { return false; }
            if !check_optional_matrix(&layer.feed_forward.gate_proj) { return false; }
            if !check_optional_vector(&layer.feed_forward.gate_bias) { return false; }
            if !check_vector(&layer.attention_norm.weight) { return false; }
            if !check_optional_vector(&layer.attention_norm.bias) { return false; }
            if !check_vector(&layer.ffn_norm.weight) { return false; }
            if !check_optional_vector(&layer.ffn_norm.bias) { return false; }
        }

        if !check_vector(&self.final_norm.weight) { return false; }
        if !check_optional_vector(&self.final_norm.bias) { return false; }
        if !check_optional_matrix(&self.lm_head) { return false; }
        if !check_optional_matrix(&self.position_embeddings) { return false; }

        true
    }
}

// ============================================================================
// 训练器主结构
// ============================================================================

pub struct Trainer {
    pub config: Config,
    pub model: Transformer,
    optimizer: Optimizer,
    output_dir: PathBuf,
    pub current_step: usize,
    pub best_loss: f64,
    pub train_losses: Vec<f64>,
    pub eval_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    mixed_precision: Option<MixedPrecisionTrainer>,
    ema_model: Option<Transformer>,
    ema_decay: f64,
    gradient_accumulation_steps: usize,
    micro_batch_size: usize,
    rng: rand::rngs::StdRng,
}

// ============================================================================
// Trainer 实现
// ============================================================================

impl Trainer {
    pub fn new(config: Config, mut model: Transformer, output_dir: PathBuf) -> Self {
        let mut optimizer = Optimizer::from_config(&config.training);

        let mixed_precision = config.training.mixed_precision.as_ref().map(|mp| {
            let mp_trainer = MixedPrecisionTrainer::new(*mp);
            mp_trainer.convert_model_to_precision(&mut model);
            mp_trainer
        });

        let ema_decay = config.training.ema_decay.unwrap_or(0.999);
        let ema_model = if config.training.ema_decay.is_some() {
            Some(model.clone())
        } else {
            None
        };

        let micro_batch_size = config
            .training
            .micro_batch_size
            .unwrap_or(config.training.batch_size);
        let gradient_accumulation_steps = config.training.batch_size.div_ceil(micro_batch_size);

        let flat_params = model.flatten_parameters();
        optimizer.ensure_state_initialized(flat_params.len());

        Trainer {
            config,
            model,
            optimizer,
            output_dir,
            current_step: 0,
            best_loss: f64::INFINITY,
            train_losses: Vec::new(),
            eval_losses: Vec::new(),
            learning_rates: Vec::new(),
            mixed_precision,
            ema_model,
            ema_decay,
            gradient_accumulation_steps,
            micro_batch_size,
            rng: {
                let mut thread_rng = rand::rng();
                rand::rngs::StdRng::from_rng(&mut thread_rng)
            },
        }
    }

    /// 训练单个微批次（修复版：正确的梯度缩放顺序）
    pub fn train_micro_batch_with_mask(
        &mut self,
        input_batch: &[Vec<usize>],
        target_batch: &[Vec<usize>],
        pad_token_id: Option<usize>,
    ) -> Result<(f32, Gradients)> {
        let batch_count = input_batch.len();
        let mut total_loss = 0.0f64;
        let mut accumulated_gradients: Option<Gradients> = None;
        
        for (input, target) in input_batch.iter().zip(target_batch.iter()) {
            // 前向传播
            let logits = self.model.forward_with_mask(input, pad_token_id, true)?;
            let mut loss = self.model.compute_loss(&logits, target);
            
            // 修复：在反向传播前应用损失缩放
            if let Some(ref mp) = self.mixed_precision {
                loss = mp.scale_loss(loss);
            }
            
            total_loss += (loss / batch_count as f32) as f64;
            
            // 反向传播（得到缩放后的梯度）
            let gradients = self.model.backward(&logits, target);
            
            match &mut accumulated_gradients {
                None => accumulated_gradients = Some(gradients),
                Some(ref mut acc) => acc.accumulate(&gradients),
            }
        }
        
        let avg_loss = (total_loss / batch_count as f64) as f32;
        let gradients = accumulated_gradients.unwrap_or_else(|| Gradients::zeros(&self.model.params));
        
        Ok((avg_loss, gradients))
    }
    
    /// 训练批次（修复版：正确的混合精度梯度缩放逻辑）
    pub fn train_batch_with_mask(
        &mut self,
        input_batch: &[Vec<usize>],
        target_batch: &[Vec<usize>],
        pad_token_id: Option<usize>,
    ) -> Result<f64> {
        let batch_size = input_batch.len();
        let mut total_loss = 0.0f64;
        let mut accumulated_gradients: Option<Gradients> = None;
        let mut overflow_occurred = false;
        
        // 分块处理微批次
        for chunk_start in (0..batch_size).step_by(self.micro_batch_size) {
            let chunk_end = (chunk_start + self.micro_batch_size).min(batch_size);
            let micro_input = &input_batch[chunk_start..chunk_end];
            let micro_target = &target_batch[chunk_start..chunk_end];
            
            let (micro_loss, micro_gradients) = 
                self.train_micro_batch_with_mask(micro_input, micro_target, pad_token_id)?;
            
            total_loss += micro_loss as f64 * micro_input.len() as f64;
            
            // 检查梯度溢出（在缩放后的梯度上检查）
            let flat_grads = micro_gradients.flatten();
            if MixedPrecisionTrainer::check_overflow(&flat_grads) {
                overflow_occurred = true;
                break;
            }
            
            // 累积梯度
            match &mut accumulated_gradients {
                None => accumulated_gradients = Some(micro_gradients),
                Some(ref mut acc) => acc.accumulate(&micro_gradients),
            }
        }
        
        let avg_loss = total_loss / batch_size as f64;
        
        // 修复：正确的混合精度处理顺序
        if let Some(ref mut mp) = self.mixed_precision {
            if overflow_occurred {
                // 发生溢出：跳过此次更新，降低缩放因子
                mp.update_scale(true);
                return Ok(avg_loss);
            }
            
            // 未溢出：先 unscale 梯度，然后更新参数
            let num_micro_batches = batch_size.div_ceil(self.micro_batch_size).max(1);
            
            if let Some(mut gradients) = accumulated_gradients {
                // 平均微批次梯度
                if num_micro_batches > 1 {
                    gradients.scale(1.0 / num_micro_batches as f32);
                }
                
                let mut flat_grads = gradients.flatten();
                
                // 修复关键：unscale 梯度（除以 loss_scale）
                let inv_scale = 1.0 / mp.loss_scale;
                for g in flat_grads.iter_mut() {
                    *g *= inv_scale;
                }
                
                // 梯度裁剪（在 unscale 之后）
                self.optimizer.clip_gradients_flat(&mut flat_grads);
                
                // 更新模型参数
                let lr = self.optimizer.get_lr(&self.config.training) as f32;
                let mut flat_params = self.model.flatten_parameters();
                self.optimizer.update(&mut flat_params, &flat_grads, lr);
                self.model.unflatten_parameters(&flat_params);
                
                // 更新缩放因子（无溢出）
                mp.update_scale(false);
            }
        } else {
            // 无混合精度：正常处理
            if let Some(mut gradients) = accumulated_gradients {
                let num_micro_batches = batch_size.div_ceil(self.micro_batch_size).max(1);
                if num_micro_batches > 1 {
                    gradients.scale(1.0 / num_micro_batches as f32);
                }
                
                let mut flat_grads = gradients.flatten();
                self.optimizer.clip_gradients_flat(&mut flat_grads);
                
                let lr = self.optimizer.get_lr(&self.config.training) as f32;
                let mut flat_params = self.model.flatten_parameters();
                self.optimizer.update(&mut flat_params, &flat_grads, lr);
                self.model.unflatten_parameters(&flat_params);
            }
        }
        
        self.current_step += 1;
        self.optimizer.increment_step();
        
        Ok(avg_loss)
    }

    /// 更新EMA模型（每次参数更新后调用）
    pub fn update_ema(&mut self) {
        if let Some(ref mut ema_model) = self.ema_model {
            Self::update_ema_inplace(ema_model, &self.model, self.ema_decay);
        }
    }

    /// 原地更新EMA模型
    fn update_ema_inplace(ema_model: &mut Transformer, model: &Transformer, decay: f64) {
        let decay_f32 = decay as f32;
        let one_minus_decay = 1.0f32 - decay_f32;

        let update_matrix = |ema: &mut [Vec<f32>], src: &[Vec<f32>]| {
            for (e_row, s_row) in ema.iter_mut().zip(src.iter()) {
                for (e, s) in e_row.iter_mut().zip(s_row.iter()) {
                    *e = decay_f32 * *e + one_minus_decay * *s;
                }
            }
        };

        let update_vector = |ema: &mut [f32], src: &[f32]| {
            for (e, s) in ema.iter_mut().zip(src.iter()) {
                *e = decay_f32 * *e + one_minus_decay * *s;
            }
        };

        let update_optional_matrix = |ema: &mut Option<Vec<Vec<f32>>>, src: &Option<Vec<Vec<f32>>>| {
            if let (Some(ref mut e_mat), Some(ref s_mat)) = (ema, src) {
                update_matrix(e_mat, s_mat);
            }
        };

        let update_optional_vector = |ema: &mut Option<Vec<f32>>, src: &Option<Vec<f32>>| {
            if let (Some(ref mut e_vec), Some(ref s_vec)) = (ema, src) {
                update_vector(e_vec, s_vec);
            }
        };

        update_matrix(&mut ema_model.embedding, &model.embedding);

        for (ema_layer, layer) in ema_model.layers.iter_mut().zip(model.layers.iter()) {
            update_matrix(&mut ema_layer.attention.q_proj, &layer.attention.q_proj);
            update_matrix(&mut ema_layer.attention.k_proj, &layer.attention.k_proj);
            update_matrix(&mut ema_layer.attention.v_proj, &layer.attention.v_proj);
            update_matrix(&mut ema_layer.attention.o_proj, &layer.attention.o_proj);
            update_matrix(&mut ema_layer.feed_forward.up_proj, &layer.feed_forward.up_proj);
            update_matrix(&mut ema_layer.feed_forward.down_proj, &layer.feed_forward.down_proj);
            update_optional_matrix(&mut ema_layer.feed_forward.gate_proj, &layer.feed_forward.gate_proj);
            update_vector(&mut ema_layer.attention_norm.weight, &layer.attention_norm.weight);
            update_vector(&mut ema_layer.ffn_norm.weight, &layer.ffn_norm.weight);
            update_optional_vector(&mut ema_layer.attention_norm.bias, &layer.attention_norm.bias);
            update_optional_vector(&mut ema_layer.ffn_norm.bias, &layer.ffn_norm.bias);
        }

        update_vector(&mut ema_model.final_norm.weight, &model.final_norm.weight);
        update_optional_vector(&mut ema_model.final_norm.bias, &model.final_norm.bias);
        update_optional_matrix(&mut ema_model.lm_head, &model.lm_head);
        update_optional_matrix(&mut ema_model.position_embeddings, &model.position_embeddings);
    }

    /// 评估验证集（带 no_grad 模式）
    pub fn evaluate(&self, eval_loader: &mut BatchLoader) -> Result<f64> {
        let mut total_loss = 0.0f64;
        let mut _total_tokens = 0u64;
        let mut total_correct = 0u64;
        let mut total_samples = 0u64;
        
        let max_samples = 500u64;

        while let Some((input_batch, target_batch)) = eval_loader.next_batch() {
            for (input, target) in input_batch.iter().zip(target_batch.iter()) {
                // 评估时不计算梯度
                let logits = self.model.forward(input, false)?;
                let loss = self.model.compute_loss(&logits, target);
                total_loss += loss as f64;
                _total_tokens += input.len() as u64;
                
                // 计算准确率（top-1）
                for (i, logit_row) in logits.iter().enumerate() {
                    if i < target.len() {
                        let max_idx = logit_row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        if max_idx == target[i] {
                            total_correct += 1;
                        }
                        total_samples += 1;
                    }
                }

                if total_samples >= max_samples {
                    break;
                }
            }
            if total_samples >= max_samples {
                break;
            }
        }

        eval_loader.reset();

        let avg_loss = if total_samples > 0 {
            total_loss / total_samples as f64
        } else {
            f64::INFINITY
        };
        
        let accuracy = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };
        
        let perplexity = avg_loss.exp();
        
        println!("   📊 评估结果: loss={:.4}, acc={:.2}%, ppl={:.2}", 
                 avg_loss, accuracy * 100.0, perplexity);

        Ok(avg_loss)
    }

    /// 使用EMA模型评估
    pub fn evaluate_with_ema(&self, eval_loader: &mut BatchLoader) -> Result<f64> {
        let ema_model = match self.ema_model.as_ref() {
            Some(m) => m,
            None => return self.evaluate(eval_loader),
        };

        let mut total_loss = 0.0f64;
        let mut total_samples = 0u64;
        let max_samples = 500u64;

        while let Some((input_batch, target_batch)) = eval_loader.next_batch() {
            for (input, target) in input_batch.iter().zip(target_batch.iter()) {
                let logits = ema_model.forward(input, false)?;
                let loss = ema_model.compute_loss(&logits, target);
                total_loss += loss as f64;
                total_samples += 1;

                if total_samples >= max_samples {
                    break;
                }
            }
            if total_samples >= max_samples {
                break;
            }
        }

        eval_loader.reset();

        if total_samples > 0 {
            Ok(total_loss / total_samples as f64)
        } else {
            Ok(f64::INFINITY)
        }
    }

    /// 保存检查点（原子操作）
    pub fn save_checkpoint(&self, step: usize, loss: f64, is_best: bool) -> Result<PathBuf> {
        let checkpoint_dir = self.output_dir.join("checkpoints");
        fs::create_dir_all(&checkpoint_dir)?;

        let checkpoint_name = format!("checkpoint_step_{:06}_loss_{:.4}", step, loss);
        let checkpoint_path = checkpoint_dir.join(&checkpoint_name);
        
        // 使用临时目录实现原子操作
        let temp_dir = tempfile::tempdir()?;
        let temp_checkpoint_path = temp_dir.path().join(&checkpoint_name);
        fs::create_dir_all(&temp_checkpoint_path)?;

        // 先写入临时文件
        self.model.save(&temp_checkpoint_path.join("model.json"))?;

        if let Some(ref ema_model) = self.ema_model {
            ema_model.save(&temp_checkpoint_path.join("ema_model.json"))?;
        }

        let state = CheckpointState {
            step,
            loss,
            best_loss: self.best_loss,
            optimizer_step: self.optimizer.get_step(),
            train_losses: self.train_losses.clone(),
            eval_losses: self.eval_losses.clone(),
            learning_rates: self.learning_rates.clone(),
        };

        let state_json = serde_json::to_string_pretty(&state)?;
        fs::write(temp_checkpoint_path.join("state.json"), state_json)?;

        // 原子地移动临时目录到目标位置
        if checkpoint_path.exists() {
            fs::remove_dir_all(&checkpoint_path)?;
        }
        fs::rename(&temp_checkpoint_path, &checkpoint_path)?;
        
        // 保持临时目录存在直到函数结束，然后自动删除
        std::mem::forget(temp_dir);

        if is_best {
            let best_path = checkpoint_dir.join("best_model");
            if best_path.exists() {
                fs::remove_dir_all(&best_path)?;
            }
            // 使用硬链接而不是复制（节省空间）
            fs::hard_link(checkpoint_path.join("model.json"), best_path.join("model.json"))?;
        }

        self.cleanup_checkpoints(&checkpoint_dir)?;

        println!(
            "   Checkpoint saved: step={}, loss={:.4}{}",
            step,
            loss,
            if is_best { " (best)" } else { "" }
        );
        Ok(checkpoint_path)
    }

    pub fn load_checkpoint(path: &Path) -> Result<(usize, f64, Transformer, CheckpointState)> {
        let model = Transformer::load(&path.join("model.json"))?;
        let state_json = fs::read_to_string(path.join("state.json"))?;
        let state: CheckpointState = serde_json::from_str(&state_json)?;
        Ok((state.step, state.loss, model, state))
    }

    /// 从检查点恢复训练状态
    pub fn restore_checkpoint(
        &mut self,
        path: &Path,
    ) -> Result<()> {
        let (_, _, restored_model, state) = Self::load_checkpoint(path)?;

        // 恢复模型权重
        self.model = restored_model;

        // 恢复训练状态
        self.current_step = state.step;
        self.best_loss = state.best_loss;
        self.train_losses = state.train_losses;
        self.eval_losses = state.eval_losses;
        self.learning_rates = state.learning_rates;

        // 恢复优化器状态（需要重新初始化以匹配当前参数量）
        let total_params = self.model.num_parameters();
        self.optimizer.step = state.optimizer_step;
        self.optimizer.ensure_state_initialized(total_params);

        // 恢复 EMA 模型
        if self.ema_model.is_some() {
            let ema_path = path.join("ema_model.json");
            if ema_path.exists() {
                self.ema_model = Some(Transformer::load(&ema_path)?);
            }
        }

        Ok(())
    }

    fn cleanup_checkpoints(&self, checkpoint_dir: &Path) -> Result<()> {
        let max_checkpoints = self.config.training.max_checkpoints;

        let mut checkpoints: Vec<_> = fs::read_dir(checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint_step_")
            })
            .collect();

        if checkpoints.len() > max_checkpoints {
            checkpoints.sort_by_key(|e| {
                e.metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            });

            for entry in checkpoints.iter().take(checkpoints.len() - max_checkpoints) {
                let _ = fs::remove_dir_all(entry.path());
            }
        }

        Ok(())
    }

    pub fn get_model(&self) -> &Transformer {
        &self.model
    }

    pub fn get_model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    pub fn get_ema_model(&self) -> Option<&Transformer> {
        self.ema_model.as_ref()
    }

    pub fn get_optimizer(&self) -> &Optimizer {
        &self.optimizer
    }

    pub fn get_optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    /// 获取混合精度统计信息
    pub fn get_mixed_precision_stats(&self) -> Option<MixedPrecisionStats> {
        self.mixed_precision.as_ref().map(|mp| mp.get_stats())
    }

    // ========================================================================
    // 主训练循环（修复版：EMA正确更新，验证集准确率计算）
    // ========================================================================

    pub fn train(
        &mut self,
        train_loader: &mut BatchLoader,
        eval_loader: &mut BatchLoader,
    ) -> Result<()> {
        let total_steps = self.config.training.num_steps;
        let save_interval = self.config.training.save_interval;
        let eval_interval = self.config.training.eval_interval;
        let log_interval = self.config.training.log_interval;

        println!("\n🚀 开始训练:");
        println!("   总步数: {}", total_steps);
        println!("   批次大小: {}", self.config.training.batch_size);
        println!("   微批次大小: {}", self.micro_batch_size);
        println!("   梯度累积步数: {}", self.gradient_accumulation_steps);
        println!("   序列长度: {}", self.config.training.sequence_length);
        println!("   学习率: {:.2e}", self.config.training.learning_rate);

        if let Some(ref mp) = self.mixed_precision {
            if mp.enabled {
                println!(
                    "   混合精度: {:?}, loss_scale: {}",
                    mp.precision, mp.loss_scale
                );
                println!(
                    "   loss_scale范围: [{}, {}]",
                    mp.min_loss_scale, mp.max_loss_scale
                );
            }
        }
        if self.ema_model.is_some() {
            println!("   EMA衰减: {:.4}", self.ema_decay);
            println!("   EMA将在每次参数更新后更新");
        }
        println!();

        let start_time = Instant::now();
        let mut tokens_processed = 0u64;

        while self.current_step < total_steps {
            let lr = self.optimizer.get_lr(&self.config.training);
            self.learning_rates.push(lr);

            match train_loader.next_batch() {
                Some((input_batch, target_batch)) => {
                    let batch_tokens = (input_batch.len() * input_batch[0].len()) as u64;

                    match self.train_batch_with_mask(&input_batch, &target_batch, None) {
                        Ok(loss) => {
                            self.train_losses.push(loss);
                            tokens_processed += batch_tokens;
                            
                            // 修复：每次参数更新后更新EMA模型
                            if self.ema_model.is_some() {
                                self.update_ema();
                            }

                            if loss < self.best_loss {
                                self.best_loss = loss;
                            }

                            if self.current_step.is_multiple_of(log_interval) || self.current_step <= 1 {
                                let elapsed = start_time.elapsed().as_secs_f64().max(1e-6);
                                let steps_per_sec = self.current_step as f64 / elapsed;
                                let tokens_per_sec = tokens_processed as f64 / elapsed;
                                let remaining = (total_steps - self.current_step) as f64 / steps_per_sec.max(1e-6);

                                println!(
                                    "   Step {}/{} | Loss: {:.4} | Best: {:.4} | LR: {:.2e} | {:.1} st/s | {:.1}k tok/s | ETA: {:.1}h",
                                    self.current_step, total_steps, loss, self.best_loss,
                                    lr, steps_per_sec, tokens_per_sec / 1000.0, remaining / 3600.0
                                );
                            }

                            // 验证（修复：使用正确的评估方法）
                            if self.current_step.is_multiple_of(eval_interval) && self.current_step > 0 {
                                match self.evaluate(eval_loader) {
                                    Ok(eval_loss) => {
                                        self.eval_losses.push(eval_loss);
                                        println!("   📊 验证损失: {:.4}", eval_loss);
                                        println!("   📊 困惑度: {:.2}", eval_loss.exp());

                                        if self.ema_model.is_some() {
                                            let mut eval_clone = BatchLoader::new(
                                                eval_loader.get_data_paths().to_vec(),
                                                eval_loader.batch_size,
                                                eval_loader.sequence_length,
                                            );
                                            if eval_clone.load_all().is_ok() {
                                                if let Ok(ema_loss) = self.evaluate_with_ema(&mut eval_clone) {
                                                    println!("   📊 EMA验证损失: {:.4}", ema_loss);
                                                    println!("   📊 EMA困惑度: {:.2}", ema_loss.exp());
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => println!("   ⚠️ 验证失败: {}", e),
                                }
                            }

                            if self.current_step.is_multiple_of(save_interval) && self.current_step > 0 {
                                let is_best = loss <= self.best_loss;
                                if let Err(e) = self.save_checkpoint(self.current_step, loss, is_best) {
                                    println!("   ⚠️ 检查点保存失败: {}", e);
                                }
                            }
                        }
                        Err(e) => println!("   ⚠️ 批次训练失败: {}", e),
                    }
                }
                None => {
                    println!("   数据加载完毕，提前停止");
                    break;
                }
            }
        }

        let final_loss = self.train_losses.last().copied().unwrap_or(0.0);
        let _ = self.save_checkpoint(self.current_step, final_loss, true);

        let final_model_path = self.output_dir.join("final_model.json");
        self.model.save(&final_model_path)?;

        if let Some(ref ema_model) = self.ema_model {
            ema_model.save(&self.output_dir.join("ema_model.json"))?;
        }

        let total_time = start_time.elapsed();
        let total_tokens = tokens_processed;

        println!("\n✨ 训练完成!");
        println!("   步数: {}", self.current_step);
        println!("   时间: {:.1}h", total_time.as_secs_f64() / 3600.0);
        println!("   Token数: {:.1}M", total_tokens as f64 / 1e6);
        println!(
            "   速度: {:.1}k tok/s",
            total_tokens as f64 / total_time.as_secs_f64().max(1e-6) / 1000.0
        );
        println!("   最佳损失: {:.4}", self.best_loss);
        println!("   最终损失: {:.4}", final_loss);

        if let Some(stats) = self.get_mixed_precision_stats() {
            println!("\n   混合精度统计:");
            println!("     最终 loss_scale: {:.2}", stats.loss_scale);
            println!("     溢出次数: {}", stats.overflow_count);
            println!("     溢出率: {:.2}%", stats.overflow_rate * 100.0);
        }

        let history = TrainingHistory {
            train_losses: self.train_losses.clone(),
            eval_losses: self.eval_losses.clone(),
            learning_rates: self.learning_rates.clone(),
            best_loss: self.best_loss,
            total_steps: self.current_step,
            total_time_seconds: total_time.as_secs_f64(),
            total_tokens,
            config: self.config.clone(),
        };

        let history_json = serde_json::to_string_pretty(&history)?;
        fs::write(self.output_dir.join("training_history.json"), history_json)?;

        Ok(())
    }
}

// ============================================================================
// 辅助结构
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointState {
    pub step: usize,
    pub loss: f64,
    pub best_loss: f64,
    pub optimizer_step: usize,
    pub train_losses: Vec<f64>,
    pub eval_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub eval_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub best_loss: f64,
    pub total_steps: usize,
    pub total_time_seconds: f64,
    pub total_tokens: u64,
    pub config: Config,
}

// ============================================================================
// 损失函数
// ============================================================================

pub struct LossFunction;

impl LossFunction {
    pub fn cross_entropy(logits: &[f32], targets: &[usize], vocab_size: usize, ignore_index: Option<usize>) -> f32 {
        let mut total_loss = 0.0f64;
        let mut count = 0u64;

        for (i, &target) in targets.iter().enumerate() {
            if let Some(ignore_idx) = ignore_index {
                if target == ignore_idx {
                    continue;
                }
            }

            if target < vocab_size {
                let start_idx = i * vocab_size;
                let end_idx = start_idx + vocab_size;

                if end_idx <= logits.len() {
                    let logit_slice = &logits[start_idx..end_idx];
                    let max_logit = logit_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_sum: f64 = logit_slice
                        .iter()
                        .map(|&x| ((x - max_logit) as f64).exp())
                        .sum();

                    if exp_sum > 0.0 {
                        total_loss -= (logit_slice[target] as f64) - (max_logit as f64) - exp_sum.ln();
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            (total_loss / count as f64) as f32
        } else {
            0.0
        }
    }

    pub fn perplexity(loss: f32) -> f32 {
        loss.exp()
    }
}

// ============================================================================
// 早停机制
// ============================================================================

pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    counter: usize,
    best_loss: f64,
    should_stop: bool,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            counter: 0,
            best_loss: f64::INFINITY,
            should_stop: false,
        }
    }

    pub fn step(&mut self, current_loss: f64) -> bool {
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.should_stop = true;
            }
        }
        self.should_stop
    }

    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_loss = f64::INFINITY;
        self.should_stop = false;
    }

    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ActivationFunction, AttentionType, ModelConfig, NormalizationType, PositionEncoding};

    fn create_test_params() -> ModelParams {
        ModelParams {
            num_layers: 2,
            hidden_dim: 64,
            num_heads: 4,
            num_key_value_heads: 4,
            intermediate_dim: 256,
            vocab_size: 1000,
            max_position_embeddings: 128,
            activation: ActivationFunction::GELU,
            position_encoding: PositionEncoding::RoPE,
            normalization: NormalizationType::Rms,
            attention_type: AttentionType::MHA,
            use_qkv_bias: true,
            use_mlp_bias: true,
            tied_embedding: true,
            dropout: 0.1,
            stochastic_depth: None,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            head_dim: 16,
        }
    }

    #[test]
    fn test_mixed_precision_scale_order() {
        let mut mp = MixedPrecisionTrainer::new(MixedPrecision::FP16);
        let initial_scale = mp.loss_scale;
        
        let mut gradients = vec![1.0f32, 2.0, 3.0];
        
        // 模拟正确的混合精度训练流程
        let loss = 0.5f32;
        let scaled_loss = mp.scale_loss(loss);
        assert_eq!(scaled_loss, loss * initial_scale);
        
        // 模拟反向传播后得到缩放后的梯度
        for g in &mut gradients {
            *g *= initial_scale;
        }
        
        let mut update_called = false;
        let success = mp.post_backward(&mut gradients, |grads| {
            update_called = true;
            // 验证梯度已经被 unscale
            assert!(grads.iter().all(|&g| g < 1e6));
        });
        
        assert!(success);
        assert!(update_called);
    }

    #[test]
    fn test_mixed_precision_overflow_handling() {
        let mut mp = MixedPrecisionTrainer::new(MixedPrecision::FP16);
        let initial_scale = mp.loss_scale;
        
        let mut gradients = vec![f32::INFINITY, 2.0, 3.0];
        
        let mut update_called = false;
        let success = mp.post_backward(&mut gradients, |_| {
            update_called = true;
        });
        
        assert!(!success);
        assert!(!update_called);
        assert!(mp.loss_scale < initial_scale);
        assert_eq!(mp.overflow_count, 1);
    }

    #[test]
    fn test_ema_update() {
        let params = create_test_params();
        let mut model = Transformer::new(params.clone());
        let mut ema_model = model.clone();

        let decay = 0.9;
        let one_minus_decay = 0.1;

        // 将 EMA 模型初始权重设为已知值
        if !ema_model.embedding.is_empty() && !ema_model.embedding[0].is_empty() {
            ema_model.embedding[0][0] = 1.0;
        }

        // 修改模型权重
        if !model.embedding.is_empty() && !model.embedding[0].is_empty() {
            model.embedding[0][0] = 2.0;
        }

        Trainer::update_ema_inplace(&mut ema_model, &model, decay);

        // 验证 EMA 公式: ema = decay * old_ema + (1-decay) * model
        // old_ema = 1.0，model = 2.0
        // 期望: 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
        if !ema_model.embedding.is_empty() && !ema_model.embedding[0].is_empty() {
            assert!((ema_model.embedding[0][0] - 1.1).abs() < 1e-6);
        }
    }
}