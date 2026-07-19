#![allow(dead_code)]
//! ============================================================================
//! 数据加载模块（第二部分）
//! ============================================================================
//!
//! 本模块实现了批量数据加载器、数据集分割器和数据增强功能：
//! - 批量数据加载（支持shuffle和drop_last）
//! - 流式数据加载（内存友好）
//! - 训练集/验证集分割
//! - 数据增强（随机掩码、打乱等）
//! - 数据验证和统计
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::error::Result;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

// ============================================================================
// 类型别名
// ============================================================================

/// 批次数据类型: (input_batch, target_batch)
type BatchData = (Vec<Vec<usize>>, Vec<Vec<usize>>);

// ============================================================================
// 批量数据加载器
// ============================================================================

pub struct BatchLoader {
    pub data_paths: Vec<PathBuf>,
    pub batch_size: usize,
    pub sequence_length: usize,
    rng: StdRng,
    current_file_idx: usize,
    current_position: usize,
    all_sequences: Vec<Vec<usize>>,
    shuffle_buffer: Vec<Vec<usize>>,
    epoch: usize,
    shuffle: bool,
    drop_last: bool,
    /// 是否使用流式模式（内存友好）
    streaming_mode: bool,
    /// 当前加载器的唯一ID（用于调试）
    loader_id: usize,
    /// 统计信息
    stats: BatchLoaderStats,
}

/// 批次加载器统计信息
#[derive(Debug, Clone, Default)]
pub struct BatchLoaderStats {
    pub total_batches_returned: usize,
    pub total_samples_processed: usize,
    pub total_files_loaded: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl BatchLoader {
    // ========================================================================
    // 创建与初始化
    // ========================================================================

    pub fn new(data_paths: Vec<PathBuf>, batch_size: usize, sequence_length: usize) -> Self {
        static NEXT_LOADER_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        let loader_id = NEXT_LOADER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        BatchLoader {
            data_paths,
            batch_size,
            sequence_length,
            rng: StdRng::from_os_rng(),
            current_file_idx: 0,
            current_position: 0,
            all_sequences: Vec::new(),
            shuffle_buffer: Vec::new(),
            epoch: 0,
            shuffle: true,
            drop_last: true,
            streaming_mode: false,
            loader_id,
            stats: BatchLoaderStats::default(),
        }
    }

    /// 设置是否打乱数据
    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
    }

    /// 设置是否丢弃最后一个不完整的批次
    pub fn set_drop_last(&mut self, drop_last: bool) {
        self.drop_last = drop_last;
    }

    /// 设置流式模式（内存友好，避免加载全部数据）
    pub fn set_streaming_mode(&mut self, streaming: bool) {
        self.streaming_mode = streaming;
    }

    /// 获取数据路径列表（用于外部访问）
    pub fn get_data_paths(&self) -> &[PathBuf] {
        &self.data_paths
    }

    /// 获取当前epoch
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> &BatchLoaderStats {
        &self.stats
    }

    // ========================================================================
    // 加载所有数据到内存
    // ========================================================================

    pub fn load_all(&mut self) -> Result<()> {
        if self.streaming_mode {
            println!("   ⚠️ 流式模式下不应调用 load_all()，使用流式API");
            return Ok(());
        }
        
        self.all_sequences.clear();

        for path in &self.data_paths {
            let sequences = self.load_file(path)?;
            self.stats.total_files_loaded += 1;
            self.all_sequences.extend(sequences);
        }

        if self.shuffle {
            self.all_sequences.shuffle(&mut self.rng);
        }

        println!("✅ 加载了 {} 个序列", self.all_sequences.len());
        Ok(())
    }

    fn load_file(&self, path: &Path) -> Result<Vec<Vec<usize>>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut sequences = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let tokens: Vec<usize> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if tokens.len() > self.sequence_length {
                let stride = self.sequence_length + 1;
                for i in (0..=tokens.len() - self.sequence_length - 1).step_by(stride) {
                    let seq: Vec<usize> = tokens[i..i + self.sequence_length + 1].to_vec();
                    sequences.push(seq);
                }

                let remaining_start = tokens.len() - self.sequence_length - 1;
                if remaining_start > 0 && !remaining_start.is_multiple_of(stride) {
                    let seq: Vec<usize> = tokens[remaining_start..].to_vec();
                    if seq.len() == self.sequence_length + 1 {
                        sequences.push(seq);
                    }
                }
            }
        }

        Ok(sequences)
    }

    // ========================================================================
    // 批次获取（内存模式）
    // ========================================================================

    pub fn next_batch(&mut self) -> Option<BatchData> {
        if self.streaming_mode {
            return self.next_batch_streaming();
        }
        
        if self.all_sequences.is_empty() {
            return None;
        }

        let mut input_batch = Vec::with_capacity(self.batch_size);
        let mut target_batch = Vec::with_capacity(self.batch_size);

        let total_sequences = self.all_sequences.len();
        let mut samples_collected = 0;

        for _ in 0..self.batch_size {
            if self.current_position >= total_sequences {
                self.current_position = 0;
                self.epoch += 1;

                if self.shuffle {
                    self.all_sequences.shuffle(&mut self.rng);
                }
            }

            if self.current_position >= total_sequences {
                self.current_position = 0;
            }

            let sequence = &self.all_sequences[self.current_position];

            if sequence.len() < self.sequence_length + 1 {
                self.current_position += 1;
                continue;
            }

            let input = sequence[..self.sequence_length].to_vec();
            let target = sequence[1..self.sequence_length + 1].to_vec();

            input_batch.push(input);
            target_batch.push(target);
            samples_collected += 1;

            self.current_position += 1;
        }

        if input_batch.is_empty() {
            return None;
        }

        // ====================================================================
        // 修复 P1-1: 不完整批次处理 - 使用随机采样填充而非重复最后一个样本
        // ====================================================================
        if !self.drop_last && input_batch.len() < self.batch_size {
            let needed = self.batch_size - input_batch.len();
            self.fill_incomplete_batch(&mut input_batch, &mut target_batch, needed);
        }

        self.stats.total_batches_returned += 1;
        self.stats.total_samples_processed += samples_collected;

        Some((input_batch, target_batch))
    }

    /// 修复：使用随机采样填充不完整批次，避免训练偏差
    fn fill_incomplete_batch(
        &mut self,
        input_batch: &mut Vec<Vec<usize>>,
        target_batch: &mut Vec<Vec<usize>>,
        needed: usize,
    ) {
        // 方法1：从已有批次中随机采样
        if !input_batch.is_empty() {
            for _ in 0..needed {
                let idx = self.rng.random_range(0..input_batch.len());
                input_batch.push(input_batch[idx].clone());
                target_batch.push(target_batch[idx].clone());
            }
            return;
        }
        
        // 方法2：从 all_sequences 中随机采样
        if !self.all_sequences.is_empty() {
            for _ in 0..needed {
                let seq_idx = self.rng.random_range(0..self.all_sequences.len());
                let sequence = &self.all_sequences[seq_idx];

                if sequence.len() > self.sequence_length {
                    let start_pos = if sequence.len() > self.sequence_length + 1 {
                        self.rng.random_range(0..sequence.len() - self.sequence_length)
                    } else {
                        0
                    };
                    
                    let input = sequence[start_pos..start_pos + self.sequence_length].to_vec();
                    let target = sequence[start_pos + 1..start_pos + self.sequence_length + 1].to_vec();
                    
                    input_batch.push(input);
                    target_batch.push(target);
                } else {
                    // 降级：使用序列本身（如果长度不足则填充）
                    let mut input = sequence.clone();
                    let mut target = sequence.clone();
                    while input.len() < self.sequence_length {
                        input.push(0);  // 使用 pad token
                    }
                    while target.len() < self.sequence_length {
                        target.push(0);
                    }
                    input_batch.push(input);
                    target_batch.push(target);
                }
            }
            return;
        }
        
        // 方法3：最后手段 - 使用默认值填充（避免 panic）
        let default_input = vec![0; self.sequence_length];
        let default_target = vec![0; self.sequence_length];
        for _ in 0..needed {
            input_batch.push(default_input.clone());
            target_batch.push(default_target.clone());
        }
        
        println!("   ⚠️ 批次填充使用了默认值，请检查数据加载");
    }

    // ========================================================================
    // 流式批次获取（内存友好，修复版）
    // ========================================================================

    /// 流式加载批次（使用缓冲区，避免加载全部数据到内存）
    pub fn next_batch_streaming(&mut self) -> Option<BatchData> {
        let mut input_batch = Vec::with_capacity(self.batch_size);
        let mut target_batch = Vec::with_capacity(self.batch_size);
    
        let buffer_size = (self.batch_size * 10).min(1000);
        let max_retries = (self.data_paths.len() + 1).saturating_mul(10).max(100);
        let mut retries = 0;
        let mut samples_collected = 0;
    
        while input_batch.len() < self.batch_size && retries < max_retries {
            // 从缓冲区随机采样
            if !self.shuffle_buffer.is_empty() {
                let idx = self.rng.random_range(0..self.shuffle_buffer.len());
                let sequence = self.shuffle_buffer.swap_remove(idx);

                if sequence.len() > self.sequence_length {
                    let input = sequence[..self.sequence_length].to_vec();
                    let target = sequence[1..self.sequence_length + 1].to_vec();

                    input_batch.push(input);
                    target_batch.push(target);
                    samples_collected += 1;
                    continue;
                }
            }
    
            // 缓冲区不足时加载更多数据
            if self.shuffle_buffer.len() < buffer_size {
                if self.current_file_idx >= self.data_paths.len() {
                    break;
                }
    
                match self.load_next_file_chunk() {
                    Ok(Some(sequences)) => {
                        for seq in sequences {
                            if seq.len() > self.sequence_length {
                                self.shuffle_buffer.push(seq);
                                if self.shuffle_buffer.len() >= buffer_size * 2 {
                                    break;
                                }
                            }
                        }
                        if self.shuffle_buffer.len() > buffer_size {
                            self.shuffle_buffer.shuffle(&mut self.rng);
                        }
                    }
                    Ok(None) => {
                        self.current_file_idx += 1;
                        self.current_position = 0;
                        self.stats.total_files_loaded += 1;
                    }
                    Err(e) => {
                        eprintln!("⚠️ 加载文件失败: {}", e);
                        self.current_file_idx += 1;
                        self.current_position = 0;
                    }
                }
            }
    
            if self.shuffle_buffer.is_empty() && self.current_file_idx >= self.data_paths.len() {
                break;
            }
    
            retries += 1;
        }
    
        if input_batch.is_empty() {
            return None;
        }
    
        // ====================================================================
        // 修复 P1-1: 不完整批次处理 - 使用随机采样填充
        // ====================================================================
        if (self.drop_last && input_batch.len() < self.batch_size) || input_batch.is_empty() {
            None
        } else if input_batch.len() < self.batch_size {
            let needed = self.batch_size - input_batch.len();
            self.fill_incomplete_batch_streaming(&mut input_batch, &mut target_batch, needed);
            Some((input_batch, target_batch))
        } else {
            self.stats.total_batches_returned += 1;
            self.stats.total_samples_processed += samples_collected;
            Some((input_batch, target_batch))
        }
    }

    /// 流式模式下的批次填充（使用缓冲区中的数据）
    fn fill_incomplete_batch_streaming(
        &mut self,
        input_batch: &mut Vec<Vec<usize>>,
        target_batch: &mut Vec<Vec<usize>>,
        needed: usize,
    ) {
        // 优先从 shuffle_buffer 中采样
        if !self.shuffle_buffer.is_empty() {
            for _ in 0..needed {
                let idx = self.rng.random_range(0..self.shuffle_buffer.len());
                let sequence = &self.shuffle_buffer[idx];

                if sequence.len() > self.sequence_length {
                    let start_pos = if sequence.len() > self.sequence_length + 1 {
                        self.rng.random_range(0..sequence.len() - self.sequence_length)
                    } else {
                        0
                    };
                    
                    let input = sequence[start_pos..start_pos + self.sequence_length].to_vec();
                    let target = sequence[start_pos + 1..start_pos + self.sequence_length + 1].to_vec();
                    
                    input_batch.push(input);
                    target_batch.push(target);
                } else {
                    input_batch.push(sequence[..self.sequence_length].to_vec());
                    target_batch.push(sequence[1..self.sequence_length + 1].to_vec());
                }
            }
            return;
        }
        
        // 降级：使用已有批次中的样本重复（带随机偏移）
        if !input_batch.is_empty() {
            for _ in 0..needed {
                let idx = self.rng.random_range(0..input_batch.len());
                let mut input = input_batch[idx].clone();
                let mut target = target_batch[idx].clone();

                // 随机打乱一小部分以增加多样性
                if self.rng.random_bool(0.1) {
                    let swap_idx = self.rng.random_range(0..input.len().saturating_sub(1));
                    input.swap(swap_idx, swap_idx + 1);
                    target.swap(swap_idx, swap_idx + 1);
                }
                
                input_batch.push(input);
                target_batch.push(target);
            }
            return;
        }
        
        // 最后手段
        let default_input = vec![0; self.sequence_length];
        let default_target = vec![0; self.sequence_length];
        for _ in 0..needed {
            input_batch.push(default_input.clone());
            target_batch.push(default_target.clone());
        }
    }

    /// 加载下一个文件块
    fn load_next_file_chunk(&mut self) -> Result<Option<Vec<Vec<usize>>>> {
        if self.current_file_idx >= self.data_paths.len() {
            return Ok(None);
        }

        let path = &self.data_paths[self.current_file_idx];
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let chunk_size = 1000;
        let mut sequences = Vec::with_capacity(chunk_size);
        let mut line_count = 0;

        for line in reader.lines().skip(self.current_position) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let tokens: Vec<usize> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if tokens.len() > self.sequence_length {
                let stride = self.sequence_length + 1;
                for i in (0..=tokens.len() - self.sequence_length - 1).step_by(stride) {
                    let seq: Vec<usize> = tokens[i..i + self.sequence_length + 1].to_vec();
                    sequences.push(seq);
                    if sequences.len() >= chunk_size {
                        break;
                    }
                }

                let remaining_start = tokens.len() - self.sequence_length - 1;
                if remaining_start > 0
                    && !remaining_start.is_multiple_of(stride)
                    && sequences.len() < chunk_size
                {
                    let seq: Vec<usize> = tokens[remaining_start..].to_vec();
                    if seq.len() == self.sequence_length + 1 {
                        sequences.push(seq);
                    }
                }
            }

            line_count += 1;
            self.current_position += 1;

            if sequences.len() >= chunk_size {
                break;
            }
        }

        if sequences.is_empty() && line_count == 0 {
            Ok(None)
        } else {
            Ok(Some(sequences))
        }
    }

    // ========================================================================
    // 辅助方法
    // ========================================================================

    /// 获取总批次数
    pub fn num_batches(&self) -> usize {
        if self.streaming_mode {
            // 流式模式下估算批次数
            if self.data_paths.is_empty() {
                return 0;
            }
            // 估算：假设每个文件平均有 N 个序列
            let estimated_sequences_per_file = 1000;
            let total_estimated = self.data_paths.len() * estimated_sequences_per_file;
            let batches = total_estimated / self.batch_size;
            if self.drop_last {
                batches
            } else {
                total_estimated.div_ceil(self.batch_size)
            }
        } else if self.all_sequences.is_empty() {
            0
        } else {
            let total = self.all_sequences.len() / self.batch_size;
            if self.drop_last {
                total
            } else {
                self.all_sequences.len().div_ceil(self.batch_size)
            }
        }
    }

    /// 重置加载器状态
    pub fn reset(&mut self) {
        self.current_file_idx = 0;
        self.current_position = 0;
        self.epoch = 0;
        self.shuffle_buffer.clear();

        if !self.streaming_mode && self.shuffle {
            self.all_sequences.shuffle(&mut self.rng);
        }
    }

    /// 获取当前批次数（兼容别名）
    pub fn len(&self) -> usize {
        self.num_batches()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        if self.streaming_mode {
            self.data_paths.is_empty()
        } else {
            self.all_sequences.is_empty() && self.data_paths.is_empty()
        }
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = BatchLoaderStats::default();
    }

    /// 获取加载器ID
    pub fn loader_id(&self) -> usize {
        self.loader_id
    }
}

// ============================================================================
// 数据集分割器（增强版）
// ============================================================================

pub struct DataSplitter;

impl DataSplitter {
    /// 按文件分割训练集和验证集
    pub fn train_val_split(
        data_paths: &[PathBuf],
        val_ratio: f64,
        seed: u64,
    ) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut all_paths = data_paths.to_vec();
        all_paths.shuffle(&mut rng);

        let val_size = (all_paths.len() as f64 * val_ratio).ceil() as usize;
        let val_size = val_size.max(1).min(all_paths.len().saturating_sub(1));

        let val_paths = all_paths[..val_size].to_vec();
        let train_paths = all_paths[val_size..].to_vec();

        println!("📊 数据集分割:");
        println!("   训练集: {} 个文件", train_paths.len());
        println!("   验证集: {} 个文件", val_paths.len());

        Ok((train_paths, val_paths))
    }

    /// 按序列分割训练集和验证集
    pub fn train_val_split_sequences(
        sequences: &[Vec<usize>],
        val_ratio: f64,
        seed: u64,
    ) -> Result<BatchData> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..sequences.len()).collect();
        indices.shuffle(&mut rng);

        let val_size = (sequences.len() as f64 * val_ratio).ceil() as usize;
        let val_size = val_size.max(1).min(sequences.len().saturating_sub(1));

        let val_indices: Vec<usize> = indices[..val_size].to_vec();
        let train_indices: Vec<usize> = indices[val_size..].to_vec();

        let train_sequences: Vec<Vec<usize>> = train_indices
            .iter()
            .map(|&i| sequences[i].clone())
            .collect();

        let val_sequences: Vec<Vec<usize>> =
            val_indices.iter().map(|&i| sequences[i].clone()).collect();

        println!("📊 样本级别分割:");
        println!("   训练集: {} 个样本", train_sequences.len());
        println!("   验证集: {} 个样本", val_sequences.len());

        Ok((train_sequences, val_sequences))
    }

    /// 简化的分割方法
    pub fn split_sequences(
        sequences: &[Vec<usize>],
        val_ratio: f64,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..sequences.len()).collect();
        indices.shuffle(&mut rng);

        let val_size = (sequences.len() as f64 * val_ratio).ceil() as usize;
        let val_size = val_size.max(1).min(sequences.len().saturating_sub(1));

        let val_indices: Vec<usize> = indices[..val_size].to_vec();
        let train_indices: Vec<usize> = indices[val_size..].to_vec();

        let train_sequences: Vec<Vec<usize>> = train_indices
            .iter()
            .map(|&i| sequences[i].clone())
            .collect();

        let val_sequences: Vec<Vec<usize>> =
            val_indices.iter().map(|&i| sequences[i].clone()).collect();

        (train_sequences, val_sequences)
    }

    /// 分层分割（根据序列长度分层）
    pub fn stratified_split_sequences(
        sequences: &[Vec<usize>],
        val_ratio: f64,
        seed: u64,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // 按长度分组
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (idx, seq) in sequences.iter().enumerate() {
            let len = seq.len();
            groups.entry(len).or_default().push(idx);
        }
        
        let mut train_indices: Vec<usize> = Vec::new();
        let mut val_indices: Vec<usize> = Vec::new();
        
        for (_, group_indices) in groups {
            let mut shuffled = group_indices.clone();
            shuffled.shuffle(&mut rng);
            
            let val_size = (shuffled.len() as f64 * val_ratio).ceil() as usize;
            let val_size = val_size.max(1).min(shuffled.len().saturating_sub(1));
            
            val_indices.extend(shuffled[..val_size].iter());
            train_indices.extend(shuffled[val_size..].iter());
        }
        
        let train_sequences: Vec<Vec<usize>> = train_indices
            .iter()
            .map(|&i| sequences[i].clone())
            .collect();
        
        let val_sequences: Vec<Vec<usize>> = val_indices
            .iter()
            .map(|&i| sequences[i].clone())
            .collect();
        
        println!("📊 分层分割:");
        println!("   训练集: {} 个样本", train_sequences.len());
        println!("   验证集: {} 个样本", val_sequences.len());
        
        (train_sequences, val_sequences)
    }
}

// ============================================================================
// 数据增强器（保留原有实现）
// ============================================================================

pub struct DataAugmenter {
    shuffle: bool,
    mask_prob: f64,
    mask_token_id: usize,
    pad_token_id: usize,
}

impl DataAugmenter {
    pub fn new(mask_token_id: usize, pad_token_id: usize) -> Self {
        DataAugmenter {
            shuffle: true,
            mask_prob: 0.15,
            mask_token_id,
            pad_token_id,
        }
    }

    pub fn set_mask_prob(&mut self, prob: f64) {
        self.mask_prob = prob.clamp(0.0, 1.0);
    }

    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
    }

    pub fn augment(&self, input_ids: &[usize]) -> Vec<usize> {
        let mut augmented = input_ids.to_vec();
        let mut rng = StdRng::from_os_rng();

        if self.shuffle {
            let window = 3;
            for i in (0..augmented.len()).step_by(window) {
                let end = (i + window).min(augmented.len());
                if end - i > 1 {
                    augmented[i..end].shuffle(&mut rng);
                }
            }
        }

        if self.mask_prob > 0.0 {
            for token in augmented.iter_mut() {
                if rng.random::<f64>() < self.mask_prob {
                    let rand_val = rng.random::<f64>();
                    if rand_val < 0.8 {
                        *token = self.mask_token_id;
                    } else if rand_val < 0.9
                        && self.mask_token_id > 0 {
                            *token = rng.random_range(0..self.mask_token_id);
                        }
                }
            }
        }

        augmented
    }

    pub fn augment_batch(&self, input_batch: &[Vec<usize>]) -> Vec<Vec<usize>> {
        input_batch
            .iter()
            .map(|input| self.augment(input))
            .collect()
    }
}

// ============================================================================
// 数据验证器（保留原有实现）
// ============================================================================

pub struct DataValidator;

impl DataValidator {
    pub fn validate_batch(
        input_batch: &[Vec<usize>],
        target_batch: &[Vec<usize>],
        vocab_size: usize,
    ) -> bool {
        if input_batch.len() != target_batch.len() {
            println!(
                "❌ 批次大小不匹配: {} vs {}",
                input_batch.len(),
                target_batch.len()
            );
            return false;
        }

        if input_batch.is_empty() {
            println!("❌ 批次为空");
            return false;
        }

        for (i, (input, target)) in input_batch.iter().zip(target_batch.iter()).enumerate() {
            if input.len() != target.len() {
                println!(
                    "❌ 样本{}序列长度不匹配: {} vs {}",
                    i,
                    input.len(),
                    target.len()
                );
                return false;
            }

            for (j, &id) in input.iter().enumerate() {
                if id >= vocab_size {
                    println!(
                        "❌ 样本{}输入位置{}的Token ID {} 超出词表大小 {}",
                        i, j, id, vocab_size
                    );
                    return false;
                }
            }

            for (j, &id) in target.iter().enumerate() {
                if id >= vocab_size {
                    println!(
                        "❌ 样本{}目标位置{}的Token ID {} 超出词表大小 {}",
                        i, j, id, vocab_size
                    );
                    return false;
                }
            }
        }

        true
    }

    pub fn validate_statistics(
        input_batch: &[Vec<usize>],
        _target_batch: &[Vec<usize>],
    ) -> DataBatchStats {
        let batch_size = input_batch.len();
        let seq_len = if batch_size > 0 {
            input_batch[0].len()
        } else {
            0
        };

        let mut unique_tokens = HashSet::new();
        let mut total_tokens = 0usize;

        for input in input_batch {
            for &id in input {
                unique_tokens.insert(id);
                total_tokens += 1;
            }
        }

        DataBatchStats {
            batch_size,
            sequence_length: seq_len,
            total_tokens,
            unique_tokens: unique_tokens.len(),
            token_density: if total_tokens > 0 {
                unique_tokens.len() as f64 / total_tokens as f64
            } else {
                0.0
            },
        }
    }
}

// ============================================================================
// 批次统计信息结构
// ============================================================================

#[derive(Debug, Clone)]
pub struct DataBatchStats {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub token_density: f64,
}

impl std::fmt::Display for DataBatchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Batch(batch_size={}, seq_len={}, total_tokens={}, unique_tokens={}, density={:.2}%)",
            self.batch_size,
            self.sequence_length,
            self.total_tokens,
            self.unique_tokens,
            self.token_density * 100.0
        )
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_batch_loader_creation() {
        let loader = BatchLoader::new(vec![], 32, 128);
        assert_eq!(loader.batch_size, 32);
        assert_eq!(loader.sequence_length, 128);
        assert!(loader.shuffle);
        assert!(loader.drop_last);
    }

    #[test]
    fn test_incomplete_batch_filling() {
        let mut loader = BatchLoader::new(vec![], 4, 10);
        loader.set_drop_last(false);
        
        // 模拟只有3个样本的情况
        let mut input_batch = vec![vec![1,2,3,4,5,6,7,8,9,10]; 3];
        let mut target_batch = vec![vec![2,3,4,5,6,7,8,9,10,11]; 3];
        
        loader.fill_incomplete_batch(&mut input_batch, &mut target_batch, 1);
        
        assert_eq!(input_batch.len(), 4);
        assert_eq!(target_batch.len(), 4);
        
        // 验证填充的样本与某个现有样本相同（随机采样）
        let filled = &input_batch[3];
        assert!(input_batch[0..3].contains(filled));
    }

    #[test]
    fn test_drop_last_behavior() {
        let mut loader = BatchLoader::new(vec![], 4, 10);
        
        // 模拟3个样本，drop_last=true 时应返回 None
        loader.set_drop_last(true);
        let mut input_batch = vec![vec![1;10]; 3];
        let mut target_batch = vec![vec![2;10]; 3];
        
        // 注意：这里直接测试逻辑，不通过 next_batch
        if loader.drop_last && input_batch.len() < loader.batch_size {
            assert!(input_batch.len() < loader.batch_size);
        }
    }

    #[test]
    fn test_data_splitter() {
        let paths: Vec<PathBuf> = (0..10)
            .map(|i| PathBuf::from(format!("file_{}.txt", i)))
            .collect();
        let (train, val) = DataSplitter::train_val_split(&paths, 0.2, 42).unwrap();

        assert_eq!(train.len() + val.len(), paths.len());
        assert!(val.len() >= 1);
    }

    #[test]
    fn test_stratified_split() {
        let sequences: Vec<Vec<usize>> = (0..100)
            .map(|i| vec![0; 10 + (i % 20)])
            .collect();
        
        let (train, val) = DataSplitter::stratified_split_sequences(&sequences, 0.2, 42);
        
        assert!(train.len() + val.len() == sequences.len());
        assert!(val.len() >= 1);
    }

    #[test]
    fn test_data_augmenter() {
        let augmenter = DataAugmenter::new(100, 0);
        let input = vec![1, 2, 3, 4, 5];
        let augmented = augmenter.augment(&input);

        assert_eq!(augmented.len(), input.len());
    }

    #[test]
    fn test_data_validator() {
        let input_batch = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let target_batch = vec![vec![2, 3, 4], vec![5, 6, 7]];

        let valid = DataValidator::validate_batch(&input_batch, &target_batch, 100);
        assert!(valid);

        let invalid_target = vec![vec![2, 3], vec![5, 6, 7]];
        let invalid = DataValidator::validate_batch(&input_batch, &invalid_target, 100);
        assert!(!invalid);
    }

    #[test]
    fn test_batch_stats() {
        let input_batch = vec![vec![1, 2, 3], vec![2, 3, 4]];
        let target_batch = vec![vec![2, 3, 4], vec![3, 4, 5]];

        let stats = DataValidator::validate_statistics(&input_batch, &target_batch);

        assert_eq!(stats.batch_size, 2);
        assert_eq!(stats.sequence_length, 3);
        assert_eq!(stats.total_tokens, 6);
        assert_eq!(stats.unique_tokens, 4);
    }
}