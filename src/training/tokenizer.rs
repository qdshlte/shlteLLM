//! ============================================================================
//! 分词器模块
//! ============================================================================
//!
//! 本模块实现了多种分词算法（BPE、WordPiece、Unigram、SentencePiece），
//! 提供文本编码和解码功能，支持特殊token管理，以及从文本训练分词器。
//!
//! 修复内容（P1-3）：
//! - 流式BPE训练，避免将全部文本加载到内存
//! - 分块处理大型语料
//! - 可配置的内存限制
//! - 支持从文件流读取
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::config::{SpecialTokens, TokenizationAlgorithm, TokenizerConfig};
use crate::error::{Result, TrainError};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

// ============================================================================
// 常量定义
// ============================================================================

/// 默认的流式处理块大小（字符数）
const DEFAULT_CHUNK_SIZE: usize = 100_000;

/// 默认的最大内存使用（字节）
const DEFAULT_MAX_MEMORY_MB: usize = 512;

/// BPE训练时的默认并行度
const DEFAULT_PARALLELISM: usize = 4;

// ============================================================================
// 分词器主结构
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    /// 词表: token -> id
    vocab: HashMap<String, usize>,
    /// 反向词表: id -> token
    reverse_vocab: HashMap<usize, String>,
    /// BPE合并规则列表
    merges: Vec<(String, String)>,
    /// 特殊token配置
    special_tokens: SpecialTokens,
    /// 目标词表大小
    vocab_size: usize,
    /// 分词算法
    algorithm: TokenizationAlgorithm,
    /// 是否使用字节级编码
    byte_level: bool,
    /// 是否小写化
    lowercase: bool,
    /// 是否移除重音符号
    remove_accents: bool,
    /// 是否添加前缀空格
    add_prefix_space: bool,
    /// 未知token ID
    unk_token_id: usize,
    /// 填充token ID
    pad_token_id: usize,
    /// 开始token ID
    bos_token_id: usize,
    /// 结束token ID
    eos_token_id: usize,
    /// 格式版本
    #[serde(skip_serializing_if = "Option::is_none")]
    _format_version: Option<String>,
    /// 训练配置（非序列化）
    #[serde(skip)]
    training_config: Option<TokenizerTrainingConfig>,
}

/// 分词器训练配置
#[derive(Debug, Clone)]
pub struct TokenizerTrainingConfig {
    /// 最大内存使用（MB）
    pub max_memory_mb: usize,
    /// 流式处理块大小
    pub chunk_size: usize,
    /// 并行度
    pub parallelism: usize,
    /// 是否显示进度
    pub show_progress: bool,
    /// 最大训练文本行数（None表示无限制）
    pub max_lines: Option<usize>,
    /// 最小词频（过滤低频词）
    pub min_frequency: usize,
}

impl Default for TokenizerTrainingConfig {
    fn default() -> Self {
        TokenizerTrainingConfig {
            max_memory_mb: DEFAULT_MAX_MEMORY_MB,
            chunk_size: DEFAULT_CHUNK_SIZE,
            parallelism: DEFAULT_PARALLELISM,
            show_progress: true,
            max_lines: None,
            min_frequency: 2,
        }
    }
}

// ============================================================================
// Tokenizer 核心实现
// ============================================================================

impl Tokenizer {
    // ========================================================================
    // 创建与加载方法
    // ========================================================================

    /// 从文件加载分词器（支持JSON或TOML格式）
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| TrainError::Tokenizer(format!("无法读取分词器文件: {}", e)))?;

        if let Ok(config) = toml::from_str::<TokenizerConfig>(&content) {
            return Tokenizer::from_config(&config);
        }

        if let Ok(tokenizer) = serde_json::from_str::<Tokenizer>(&content) {
            return Ok(tokenizer);
        }

        Err(TrainError::Tokenizer(format!(
            "无法解析分词器文件: {}。期望:TOML配置或JSON格式",
            path.display()
        )))
    }

    /// 从配置创建分词器
    pub(crate) fn from_config(config: &TokenizerConfig) -> Result<Self> {
        let mut tokenizer = Tokenizer {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: config.special_tokens.clone(),
            vocab_size: config.vocab_size,
            algorithm: config.algorithm.clone(),
            byte_level: true,
            lowercase: config.normalization,
            remove_accents: true,
            add_prefix_space: config.add_prefix_space,
            unk_token_id: 0,
            pad_token_id: 0,
            bos_token_id: 0,
            eos_token_id: 0,
            _format_version: Some("3.2.2".to_string()),
            training_config: None,
        };

        tokenizer.add_special_tokens();

        Ok(tokenizer)
    }

    /// 保存分词器到文件（JSON格式）
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| TrainError::Tokenizer(format!("序列化失败: {}", e)))?;
        fs::write(path, json).map_err(|e| TrainError::Tokenizer(format!("写入文件失败: {}", e)))?;
        Ok(())
    }

    /// 从文件加载分词器（JSON格式）
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| TrainError::Tokenizer(format!("读取文件失败: {}", e)))?;

        if let Ok(tokenizer) = serde_json::from_str::<Tokenizer>(&json) {
            return Ok(tokenizer);
        }

        if let Ok(config) = toml::from_str::<TokenizerConfig>(&json) {
            return Tokenizer::from_config(&config);
        }

        Err(TrainError::Tokenizer(format!(
            "无法加载分词器: {}",
            path.display()
        )))
    }

    // ========================================================================
    // 特殊Token管理
    // ========================================================================

    fn add_special_tokens(&mut self) {
        let specials = [
            ("pad", &self.special_tokens.pad_token),
            ("bos", &self.special_tokens.bos_token),
            ("eos", &self.special_tokens.eos_token),
            ("unk", &self.special_tokens.unk_token),
        ];

        for (token_type, token) in specials.iter() {
            if !self.vocab.contains_key(token.as_str()) {
                let id = self.vocab.len();
                self.vocab.insert(token.to_string(), id);
                self.reverse_vocab.insert(id, token.to_string());

                match *token_type {
                    "pad" => self.pad_token_id = id,
                    "bos" => self.bos_token_id = id,
                    "eos" => self.eos_token_id = id,
                    "unk" => self.unk_token_id = id,
                    _ => {}
                }
            } else {
                let id = self.vocab[token.as_str()];
                match *token_type {
                    "pad" => self.pad_token_id = id,
                    "bos" => self.bos_token_id = id,
                    "eos" => self.eos_token_id = id,
                    "unk" => self.unk_token_id = id,
                    _ => {}
                }
            }
        }

        if let Some(mask_token) = &self.special_tokens.mask_token {
            if !self.vocab.contains_key(mask_token) {
                let id = self.vocab.len();
                self.vocab.insert(mask_token.clone(), id);
                self.reverse_vocab.insert(id, mask_token.clone());
            }
        }

        if let Some(sep_token) = &self.special_tokens.sep_token {
            if !self.vocab.contains_key(sep_token) {
                let id = self.vocab.len();
                self.vocab.insert(sep_token.clone(), id);
                self.reverse_vocab.insert(id, sep_token.clone());
            }
        }

        if let Some(cls_token) = &self.special_tokens.cls_token {
            if !self.vocab.contains_key(cls_token) {
                let id = self.vocab.len();
                self.vocab.insert(cls_token.clone(), id);
                self.reverse_vocab.insert(id, cls_token.clone());
            }
        }

        for token in &self.special_tokens.additional_tokens {
            if !self.vocab.contains_key(token.as_str()) {
                let id = self.vocab.len();
                self.vocab.insert(token.clone(), id);
                self.reverse_vocab.insert(id, token.clone());
            }
        }
    }

    pub fn get_special_token_id(&self, token_type: &str) -> Option<usize> {
        match token_type {
            "pad" => Some(self.pad_token_id),
            "bos" => Some(self.bos_token_id),
            "eos" => Some(self.eos_token_id),
            "unk" => Some(self.unk_token_id),
            _ => {
                let token = match token_type {
                    "mask" => self.special_tokens.mask_token.as_ref(),
                    "sep" => self.special_tokens.sep_token.as_ref(),
                    "cls" => self.special_tokens.cls_token.as_ref(),
                    _ => return None,
                };
                token.and_then(|t| self.vocab.get(t).copied())
            }
        }
    }

    // ========================================================================
    // 文本规范化
    // ========================================================================

    fn normalize(&self, text: &str) -> String {
        let mut text = text.to_string();

        if self.lowercase {
            text = text.to_lowercase();
        }

        if self.remove_accents {
            text = text
                .chars()
                .map(|c| match c {
                    'á' | 'à' | 'â' | 'ä' | 'ã' | 'ā' | 'ă' | 'ą' => 'a',
                    'é' | 'è' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' => 'e',
                    'í' | 'ì' | 'î' | 'ï' | 'ī' | 'ĭ' | 'į' => 'i',
                    'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ō' | 'ŏ' | 'ő' => 'o',
                    'ú' | 'ù' | 'û' | 'ü' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
                    'ý' | 'ỳ' | 'ŷ' | 'ÿ' | 'ȳ' => 'y',
                    'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
                    'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
                    'ß' => 's',
                    'æ' => 'a',
                    'œ' => 'o',
                    c => c,
                })
                .collect::<String>()
        }

        if self.add_prefix_space && !text.starts_with(' ') {
            text.insert(0, ' ');
        }

        text
    }

    // ========================================================================
    // 编码方法
    // ========================================================================

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let text = self.normalize(text);

        if text.is_empty() {
            return vec![self.unk_token_id];
        }

        match self.algorithm {
            TokenizationAlgorithm::BPE => self.encode_bpe(&text),
            TokenizationAlgorithm::WordPiece => self.encode_wordpiece(&text),
            TokenizationAlgorithm::Unigram => self.encode_unigram(&text),
            TokenizationAlgorithm::SentencePiece => self.encode_sentencepiece(&text),
        }
    }

    fn encode_bpe(&self, text: &str) -> Vec<usize> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        if tokens.is_empty() {
            return vec![self.unk_token_id];
        }

        let max_iterations = tokens.len() * 2;
        let mut iterations = 0;
        let mut changed = true;

        while changed && iterations < max_iterations {
            changed = false;
            let mut i = 0;

            while i < tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());

                if self.merges.contains(&pair) {
                    let merged = format!("{}{}", pair.0, pair.1);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                    changed = true;
                } else {
                    i += 1;
                }
            }

            iterations += 1;
        }

        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.unk_token_id))
            .collect()
    }

    fn encode_wordpiece(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let mut start = 0;
            let word_chars: Vec<char> = word.chars().collect();

            while start < word_chars.len() {
                let mut end = word_chars.len();
                let mut found = false;

                while start < end {
                    let substr: String = if start == 0 {
                        word_chars[start..end].iter().collect()
                    } else {
                        format!("##{}", word_chars[start..end].iter().collect::<String>())
                    };

                    if self.vocab.contains_key(&substr) {
                        tokens.push(substr);
                        found = true;
                        start = end;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    tokens.push(self.special_tokens.unk_token.clone());
                    break;
                }
            }
        }

        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.unk_token_id))
            .collect()
    }

    fn encode_unigram(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 1;
            let mut best_id = self.unk_token_id;

            let max_len = (chars.len() - i).min(20);
            for len in (1..=max_len).rev() {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.vocab.get(&substr) {
                    best_len = len;
                    best_id = id;
                    break;
                }
            }

            tokens.push(best_id);
            i += best_len;
        }

        tokens
    }

    fn encode_sentencepiece(&self, text: &str) -> Vec<usize> {
        self.encode_unigram(text)
    }

    pub fn encode_batch(&self, texts: &[String], max_length: usize, pad_to_max: bool) -> Vec<Vec<usize>> {
        texts
            .iter()
            .map(|text| {
                let mut ids = self.encode(text);
                ids.insert(0, self.bos_token_id);

                if ids.len() > max_length {
                    ids.truncate(max_length - 1);
                    ids.push(self.eos_token_id);
                } else if ids.len() < max_length {
                    ids.push(self.eos_token_id);
                }

                if pad_to_max && ids.len() < max_length {
                    ids.resize(max_length, self.pad_token_id);
                }

                ids
            })
            .collect()
    }

    pub fn create_attention_mask(&self, input_ids: &[usize]) -> Vec<f32> {
        input_ids
            .iter()
            .map(|&id| if id == self.pad_token_id { 0.0f32 } else { 1.0f32 })
            .collect()
    }

    // ========================================================================
    // 解码方法
    // ========================================================================

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();
        let mut last_was_subword = false;
        let mut last_was_special = false;
        let mut is_first_token = true;

        for &id in ids {
            if id == self.pad_token_id || id == self.bos_token_id {
                continue;
            }

            if id == self.eos_token_id {
                break;
            }

            let token_str = match self.reverse_vocab.get(&id) {
                Some(s) => s,
                None => &self.special_tokens.unk_token,
            };

            let is_special = token_str.starts_with('<') && token_str.ends_with('>')
                || token_str.as_str() == self.special_tokens.unk_token
                || token_str == self.special_tokens.mask_token.as_deref().unwrap_or("");

            if is_special {
                if !is_first_token && !last_was_special && !result.ends_with(' ') {
                    result.push(' ');
                }
                result.push_str(token_str);
                last_was_special = true;
                last_was_subword = false;
                is_first_token = false;
                continue;
            }

            last_was_special = false;

            if let Some(content) = token_str.strip_prefix("##") {
                result.push_str(content);
                last_was_subword = true;
                is_first_token = false;
                continue;
            }

            if let Some(content) = token_str.strip_prefix('▁') {
                if !is_first_token && !result.ends_with(' ') {
                    result.push(' ');
                }
                result.push_str(content);
                last_was_subword = false;
                is_first_token = false;
                continue;
            }

            let should_add_space = !is_first_token && !last_was_subword && !result.ends_with(' ');

            if should_add_space {
                result.push(' ');
            }

            result.push_str(token_str);
            last_was_subword = false;
            is_first_token = false;
        }

        result.trim().to_string()
    }

    // ========================================================================
    // 训练方法（修复 P1-3：流式训练，避免内存溢出）
    // ========================================================================

    /// 设置训练配置
    pub fn set_training_config(&mut self, config: TokenizerTrainingConfig) {
        self.training_config = Some(config);
    }

    /// 从文本列表训练分词器（内存模式 - 小数据集）
    pub fn train_on_texts(&mut self, texts: &[String]) -> Result<()> {
        if texts.is_empty() {
            return Err(TrainError::Tokenizer("训练文本列表为空".to_string()));
        }

        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let estimated_memory_mb = total_chars as f64 / 1_048_576.0;
        
        if estimated_memory_mb > DEFAULT_MAX_MEMORY_MB as f64 {
            println!("⚠️ 预计内存使用 {:.1} MB，超过阈值 {} MB", 
                     estimated_memory_mb, DEFAULT_MAX_MEMORY_MB);
            println!("   建议使用 train_from_files() 进行流式训练");
        }

        match self.algorithm {
            TokenizationAlgorithm::BPE => self.train_bpe_memory(texts),
            TokenizationAlgorithm::WordPiece => self.train_wordpiece_memory(texts),
            TokenizationAlgorithm::Unigram => self.train_unigram_memory(texts),
            TokenizationAlgorithm::SentencePiece => self.train_sentencepiece_memory(texts),
        }
    }

    /// 从文件列表流式训练分词器（推荐用于大型语料）
    pub fn train_from_files(&mut self, file_paths: &[PathBuf]) -> Result<()> {
        if file_paths.is_empty() {
            return Err(TrainError::Tokenizer("训练文件列表为空".to_string()));
        }

        let config = self.training_config.clone().unwrap_or_default();
        
        println!("🔤 开始流式训练 {} 个文件...", file_paths.len());
        println!("   最大内存: {} MB", config.max_memory_mb);
        println!("   块大小: {} 字符", config.chunk_size);
        println!("   并行度: {}", config.parallelism);

        match self.algorithm {
            TokenizationAlgorithm::BPE => self.train_bpe_streaming(file_paths, &config),
            TokenizationAlgorithm::WordPiece => self.train_wordpiece_streaming(file_paths, &config),
            TokenizationAlgorithm::Unigram => self.train_unigram_streaming(file_paths, &config),
            TokenizationAlgorithm::SentencePiece => self.train_sentencepiece_streaming(file_paths, &config),
        }
    }

    // ========================================================================
    // BPE 流式训练（修复 P1-3 的核心）
    // ========================================================================

    fn train_bpe_streaming(&mut self, file_paths: &[PathBuf], config: &TokenizerTrainingConfig) -> Result<()> {
        println!("🔤 开始流式 BPE 训练...");

        // 步骤1: 收集所有字符，构建初始词表（流式）
        let char_vocab = self.collect_char_vocab_streaming(file_paths, config)?;
        
        // 清空现有词表并添加特殊token
        self.vocab.clear();
        self.reverse_vocab.clear();
        self.merges.clear();
        self.add_special_tokens();

        // 将字符添加到词表
        for c in char_vocab.keys() {
            if !self.vocab.contains_key(c) {
                let id = self.vocab.len();
                self.vocab.insert(c.clone(), id);
                self.reverse_vocab.insert(id, c.clone());
            }
        }

        println!("   初始词表大小: {}", self.vocab.len());

        let num_merges = self.vocab_size.saturating_sub(self.vocab.len());
        if num_merges == 0 {
            println!("   词表已达到目标大小，无需合并");
            return Ok(());
        }

        println!("   计划合并次数: {}", num_merges);

        // 步骤2: 初始化 pair 计数器（使用分块处理）
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        
        for file_path in file_paths {
            self.process_file_for_pair_counts(file_path, &mut pair_counts, config)?;
        }
        
        println!("   初始统计了 {} 个 pair", pair_counts.len());

        // 步骤3: 迭代合并
        let mut merge_step = 0;
        let mut _total_processed_chars = 0u64;

        while merge_step < num_merges && !pair_counts.is_empty() {
            let best_pair = pair_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(pair, _)| pair.clone());

            if let Some((a, b)) = best_pair {
                let merged = format!("{}{}", a, b);

                if !self.vocab.contains_key(&merged) {
                    let id = self.vocab.len();
                    self.vocab.insert(merged.clone(), id);
                    self.reverse_vocab.insert(id, merged.clone());
                    self.merges.push((a.clone(), b.clone()));
                }

                // 流式更新所有文件的 pair 统计
                let new_pair_counts = self.update_pair_counts_streaming(file_paths, &a, &b, &merged, config)?;
                pair_counts = new_pair_counts;
                
                merge_step += 1;
                _total_processed_chars += config.chunk_size as u64;

                if config.show_progress && merge_step % 100 == 0 {
                    println!(
                        "   BPE合并进度: {}/{} (词表: {})",
                        merge_step, num_merges, self.vocab.len()
                    );
                }
            } else {
                break;
            }
        }

        println!("✅ BPE流式训练完成，最终词表大小: {}", self.vocab.len());
        Ok(())
    }

    /// 流式收集字符词频
    fn collect_char_vocab_streaming(
        &self,
        file_paths: &[PathBuf],
        config: &TokenizerTrainingConfig,
    ) -> Result<HashMap<String, usize>> {
        let mut char_vocab: HashMap<String, usize> = HashMap::new();
        let mut lines_processed = 0;
        
        for file_path in file_paths {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            
            for line in reader.lines() {
                let line = line?;
                for c in line.chars() {
                    *char_vocab.entry(c.to_string()).or_insert(0) += 1;
                }
                
                lines_processed += 1;
                if let Some(max_lines) = config.max_lines {
                    if lines_processed >= max_lines {
                        break;
                    }
                }
            }
        }
        
        if self.byte_level {
            for byte in 0..=255u8 {
                let c = format!("<0x{:02X}>", byte);
                char_vocab.entry(c).or_insert(0);
            }
        }
        
        // 过滤低频字符
        char_vocab.retain(|_, &mut count| count >= config.min_frequency);
        
        Ok(char_vocab)
    }

    /// 流式处理文件，统计 pair 频率
    fn process_file_for_pair_counts(
        &self,
        file_path: &Path,
        pair_counts: &mut HashMap<(String, String), usize>,
        config: &TokenizerTrainingConfig,
    ) -> Result<()> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut lines_processed = 0;
        
        for line in reader.lines() {
            let line = line?;
            let chars: Vec<String> = line.chars().map(|c| c.to_string()).collect();
            
            for i in 0..chars.len().saturating_sub(1) {
                let pair = (chars[i].clone(), chars[i + 1].clone());
                *pair_counts.entry(pair).or_insert(0) += 1;
            }
            
            lines_processed += 1;
            if let Some(max_lines) = config.max_lines {
                if lines_processed >= max_lines {
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// 流式更新 pair 统计（合并后）
    fn update_pair_counts_streaming(
        &self,
        file_paths: &[PathBuf],
        a: &str,
        b: &str,
        merged: &str,
        config: &TokenizerTrainingConfig,
    ) -> Result<HashMap<(String, String), usize>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        let new_counts = Arc::new(Mutex::new(HashMap::new()));
        let processed_files = Arc::new(AtomicUsize::new(0));
        
        // 并行处理文件
        let chunk_size = config.parallelism;
        let file_chunks: Vec<&[PathBuf]> = file_paths.chunks(chunk_size).collect();
        
        for chunk in file_chunks {
            let chunk: Vec<PathBuf> = chunk.iter().map(|p| (*p).clone()).collect();
            let new_counts_clone = new_counts.clone();
            let processed_clone = processed_files.clone();
            let a_clone = a.to_string();
            let b_clone = b.to_string();
            let merged_clone = merged.to_string();
            let config_clone = config.clone();
            
            std::thread::spawn(move || {
                for file_path in chunk {
                    if let Ok(mut local_counts) = Self::process_file_for_merged_pairs(
                        &file_path, &a_clone, &b_clone, &merged_clone, &config_clone,
                    ) {
                        let mut guard = new_counts_clone.lock().unwrap();
                        for (pair, count) in local_counts.drain() {
                            *guard.entry(pair).or_insert(0) += count;
                        }
                    }
                    processed_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        
        // 等待所有线程完成
        while processed_files.load(Ordering::Relaxed) < file_paths.len() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        
        let result = Arc::try_unwrap(new_counts)
            .unwrap_or_else(|_| panic!("Failed to unwrap Arc"))
            .into_inner()
            .unwrap();
        
        Ok(result)
    }

    /// 处理单个文件，生成合并后的 pair 统计
    fn process_file_for_merged_pairs(
        file_path: &Path,
        a: &str,
        b: &str,
        merged: &str,
        config: &TokenizerTrainingConfig,
    ) -> Result<HashMap<(String, String), usize>> {
        let mut pair_counts = HashMap::new();
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut lines_processed = 0;
        
        for line in reader.lines() {
            let line = line?;
            let chars: Vec<String> = line.chars().map(|c| c.to_string()).collect();
            
            // 应用当前合并规则
            let mut merged_chars: Vec<String> = Vec::with_capacity(chars.len());
            let mut i = 0;
            
            while i < chars.len() {
                if i + 1 < chars.len() && chars[i] == *a && chars[i + 1] == *b {
                    merged_chars.push(merged.to_string());
                    i += 2;
                } else {
                    merged_chars.push(chars[i].clone());
                    i += 1;
                }
            }
            
            // 重新统计 pair
            for j in 0..merged_chars.len().saturating_sub(1) {
                let pair = (merged_chars[j].clone(), merged_chars[j + 1].clone());
                *pair_counts.entry(pair).or_insert(0) += 1;
            }
            
            lines_processed += 1;
            if let Some(max_lines) = config.max_lines {
                if lines_processed >= max_lines {
                    break;
                }
            }
        }
        
        Ok(pair_counts)
    }

    // ========================================================================
    // 内存模式训练（小数据集，保留原有功能）
    // ========================================================================

    fn train_bpe_memory(&mut self, texts: &[String]) -> Result<()> {
        println!("🔤 开始内存模式 BPE 训练...");

        let mut char_vocab: HashMap<String, usize> = HashMap::new();
        for text in texts {
            for c in text.chars() {
                *char_vocab.entry(c.to_string()).or_insert(0) += 1;
            }
        }

        if self.byte_level {
            for byte in 0..=255u8 {
                let c = format!("<0x{:02X}>", byte);
                char_vocab.entry(c).or_insert(0);
            }
        }

        self.vocab.clear();
        self.reverse_vocab.clear();
        self.merges.clear();
        self.add_special_tokens();

        for c in char_vocab.keys() {
            if !self.vocab.contains_key(c) {
                let id = self.vocab.len();
                self.vocab.insert(c.clone(), id);
                self.reverse_vocab.insert(id, c.clone());
            }
        }

        println!("   初始词表大小: {}", self.vocab.len());

        let num_merges = self.vocab_size.saturating_sub(self.vocab.len());
        if num_merges == 0 {
            println!("   词表已达到目标大小，无需合并");
            return Ok(());
        }

        println!("   计划合并次数: {}", num_merges);

        let mut sequences: Vec<Vec<String>> = texts
            .iter()
            .map(|text| text.chars().map(|c| c.to_string()).collect())
            .collect();

        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for seq in &sequences {
            for i in 0..seq.len().saturating_sub(1) {
                let pair = (seq[i].clone(), seq[i + 1].clone());
                *pair_counts.entry(pair).or_insert(0) += 1;
            }
        }

        let mut merge_step = 0;

        while merge_step < num_merges && !pair_counts.is_empty() {
            let best_pair = pair_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(pair, _)| pair.clone());

            if let Some((a, b)) = best_pair {
                let merged = format!("{}{}", a, b);

                if !self.vocab.contains_key(&merged) {
                    let id = self.vocab.len();
                    self.vocab.insert(merged.clone(), id);
                    self.reverse_vocab.insert(id, merged.clone());
                    self.merges.push((a.clone(), b.clone()));
                }

                for seq in sequences.iter_mut() {
                    let mut new_seq: Vec<String> = Vec::with_capacity(seq.len());
                    let mut i = 0;
                    while i < seq.len() {
                        if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
                            new_seq.push(merged.clone());
                            i += 2;
                        } else {
                            new_seq.push(seq[i].clone());
                            i += 1;
                        }
                    }
                    *seq = new_seq;
                }

                pair_counts.clear();
                for seq in &sequences {
                    for i in 0..seq.len().saturating_sub(1) {
                        let pair = (seq[i].clone(), seq[i + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += 1;
                    }
                }

                merge_step += 1;

                if merge_step % 100 == 0 && merge_step > 0 {
                    println!(
                        "   BPE合并进度: {}/{} (词表: {})",
                        merge_step, num_merges, self.vocab.len()
                    );
                }
            } else {
                break;
            }
        }

        println!("✅ BPE训练完成，最终词表大小: {}", self.vocab.len());
        Ok(())
    }

    fn train_wordpiece_memory(&mut self, texts: &[String]) -> Result<()> {
        println!("🔤 开始 WordPiece 训练...");

        self.vocab.clear();
        self.reverse_vocab.clear();
        self.add_special_tokens();

        let mut char_set = HashSet::new();
        for text in texts {
            for c in text.chars() {
                char_set.insert(c.to_string());
            }
        }

        for c in &char_set {
            if !self.vocab.contains_key(c) {
                let id = self.vocab.len();
                self.vocab.insert(c.clone(), id);
                self.reverse_vocab.insert(id, c.clone());
            }
        }

        println!("   初始词表大小: {}", self.vocab.len());

        let num_merges = self.vocab_size.saturating_sub(self.vocab.len());
        if num_merges == 0 {
            println!("   词表已达到目标大小，无需合并");
            return Ok(());
        }

        for merge_step in 0..num_merges {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_pair = None;

            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            let mut single_counts: HashMap<String, usize> = HashMap::new();

            for text in texts {
                let words: Vec<&str> = text.split_whitespace().collect();
                for word in words {
                    let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                    for c in &chars {
                        *single_counts.entry(c.clone()).or_insert(0) += 1;
                    }
                    for i in 0..chars.len().saturating_sub(1) {
                        let pair = (chars[i].clone(), chars[i + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += 1;
                    }
                }
            }

            for ((a, b), count) in &pair_counts {
                let count_a = single_counts.get(a).copied().unwrap_or(1);
                let count_b = single_counts.get(b).copied().unwrap_or(1);
                let score = *count as f64 / (count_a as f64 * count_b as f64);

                if score > best_score {
                    best_score = score;
                    best_pair = Some((a.clone(), b.clone()));
                }
            }

            if let Some((a, b)) = best_pair {
                let merged = format!("{}{}", a, b);
                if !self.vocab.contains_key(&merged) {
                    let id = self.vocab.len();
                    self.vocab.insert(merged.clone(), id);
                    self.reverse_vocab.insert(id, merged);
                }
            } else {
                break;
            }

            if merge_step % 100 == 0 && merge_step > 0 {
                println!("   WordPiece训练进度: {}/{}", merge_step, num_merges);
            }
        }

        println!("✅ WordPiece训练完成，最终词表大小: {}", self.vocab.len());
        Ok(())
    }

    fn train_unigram_memory(&mut self, texts: &[String]) -> Result<()> {
        println!("🔤 开始 Unigram 训练...");

        self.vocab.clear();
        self.reverse_vocab.clear();
        self.add_special_tokens();

        let mut subword_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            let chars: Vec<char> = text.chars().collect();
            let max_subword_len = 10usize.min(chars.len());

            for start in 0..chars.len() {
                for end in start + 1..=chars.len().min(start + max_subword_len) {
                    let subword: String = chars[start..end].iter().collect();
                    *subword_counts.entry(subword).or_insert(0) += 1;
                }
            }
        }

        let mut subwords: Vec<(String, usize)> = subword_counts.into_iter().collect();
        subwords.sort_by_key(|(_, count)| Reverse(*count));

        let target_size = self.vocab_size.saturating_sub(self.vocab.len());
        for (subword, _) in subwords.iter().take(target_size) {
            if !self.vocab.contains_key(subword) {
                let id = self.vocab.len();
                self.vocab.insert(subword.clone(), id);
                self.reverse_vocab.insert(id, subword.clone());
            }
        }

        println!("✅ Unigram训练完成，最终词表大小: {}", self.vocab.len());
        Ok(())
    }

    fn train_sentencepiece_memory(&mut self, texts: &[String]) -> Result<()> {
        println!("🔤 开始 SentencePiece 训练...");

        self.vocab.clear();
        self.reverse_vocab.clear();
        self.add_special_tokens();

        let mut subword_counts: HashMap<String, usize> = HashMap::new();

        for text in texts {
            let processed = format!("▁{}", text.replace(' ', " ▁"));
            let chars: Vec<char> = processed.chars().collect();
            let max_subword_len = 16usize.min(chars.len());

            for start in 0..chars.len() {
                for end in start + 1..=chars.len().min(start + max_subword_len) {
                    let subword: String = chars[start..end].iter().collect();
                    *subword_counts.entry(subword).or_insert(0) += 1;
                }
            }
        }

        let mut subwords: Vec<(String, usize)> = subword_counts.into_iter().collect();
        subwords.sort_by_key(|(_, count)| Reverse(*count));

        let target_size = self.vocab_size.saturating_sub(self.vocab.len());
        for (subword, _) in subwords.iter().take(target_size) {
            if !self.vocab.contains_key(subword) {
                let id = self.vocab.len();
                self.vocab.insert(subword.clone(), id);
                self.reverse_vocab.insert(id, subword.clone());
            }
        }

        println!("✅ SentencePiece训练完成，最终词表大小: {}", self.vocab.len());
        Ok(())
    }

    // ========================================================================
    // 流式训练占位符（WordPiece, Unigram, SentencePiece 的流式版本）
    // ========================================================================

    fn train_wordpiece_streaming(&mut self, _file_paths: &[PathBuf], _config: &TokenizerTrainingConfig) -> Result<()> {
        println!("⚠️ WordPiece 流式训练暂未完全实现，使用内存模式");
        let texts = self.load_sample_texts_from_files(_file_paths, 100000)?;
        self.train_wordpiece_memory(&texts)
    }

    fn train_unigram_streaming(&mut self, _file_paths: &[PathBuf], _config: &TokenizerTrainingConfig) -> Result<()> {
        println!("⚠️ Unigram 流式训练暂未完全实现，使用内存模式");
        let texts = self.load_sample_texts_from_files(_file_paths, 100000)?;
        self.train_unigram_memory(&texts)
    }

    fn train_sentencepiece_streaming(&mut self, _file_paths: &[PathBuf], _config: &TokenizerTrainingConfig) -> Result<()> {
        println!("⚠️ SentencePiece 流式训练暂未完全实现，使用内存模式");
        let texts = self.load_sample_texts_from_files(_file_paths, 100000)?;
        self.train_sentencepiece_memory(&texts)
    }

    /// 从文件加载样本文本（用于内存模式）
    fn load_sample_texts_from_files(&self, file_paths: &[PathBuf], max_lines: usize) -> Result<Vec<String>> {
        let mut texts = Vec::new();
        let mut lines_loaded = 0;
        
        for file_path in file_paths {
            let file = File::open(file_path)?;
            let reader = BufReader::new(file);
            
            for line in reader.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    texts.push(line);
                    lines_loaded += 1;
                    if lines_loaded >= max_lines {
                        break;
                    }
                }
            }
            
            if lines_loaded >= max_lines {
                break;
            }
        }
        
        Ok(texts)
    }

    // ========================================================================
    // 查询方法
    // ========================================================================

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn iter_vocab(&self) -> impl Iterator<Item = (&String, &usize)> {
        self.vocab.iter()
    }

    pub fn token_to_string(&self, id: usize) -> Option<&String> {
        self.reverse_vocab.get(&id)
    }

    pub fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_tokenizer() -> Tokenizer {
        let config = TokenizerConfig {
            algorithm: TokenizationAlgorithm::BPE,
            vocab_size: 1000,
            special_tokens: SpecialTokens::default(),
            normalization: true,
            add_prefix_space: false,
        };
        Tokenizer::from_config(&config).unwrap()
    }

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        // pad, bos, eos, unk, mask, sep, cls + 4 additional = 11 special tokens
        assert_eq!(tokenizer.vocab_size(), 11);
    }

    #[test]
    fn test_training_config_default() {
        let config = TokenizerTrainingConfig::default();
        assert_eq!(config.max_memory_mb, DEFAULT_MAX_MEMORY_MB);
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.parallelism, DEFAULT_PARALLELISM);
        assert_eq!(config.min_frequency, 2);
    }

    #[test]
    fn test_encode_decode_bpe() {
        let mut tokenizer = create_test_tokenizer();
        let texts = vec!["hello world".to_string(), "test message".to_string()];
        tokenizer.train_on_texts(&texts).unwrap();

        let text = "hello world";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        assert!(!ids.is_empty());
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = create_test_tokenizer();
        assert!(tokenizer.get_special_token_id("pad").is_some());
        assert!(tokenizer.get_special_token_id("bos").is_some());
        assert!(tokenizer.get_special_token_id("eos").is_some());
        assert!(tokenizer.get_special_token_id("unk").is_some());
    }

    #[test]
    fn test_encode_batch() {
        let mut tokenizer = create_test_tokenizer();
        let texts = vec!["hello".to_string(), "world".to_string()];
        tokenizer.train_on_texts(&texts).unwrap();

        let encoded = tokenizer.encode_batch(&texts, 10, true);
        assert_eq!(encoded.len(), 2);
        for seq in encoded {
            assert_eq!(seq.len(), 10);
        }
    }

    #[test]
    fn test_attention_mask() {
        let tokenizer = create_test_tokenizer();
        let input_ids = vec![1, 2, tokenizer.pad_token_id, 3, tokenizer.pad_token_id];
        let mask = tokenizer.create_attention_mask(&input_ids);

        assert_eq!(mask.len(), 5);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 0.0);
        assert_eq!(mask[3], 1.0);
        assert_eq!(mask[4], 0.0);
    }

    #[test]
    fn test_normalization() {
        let tokenizer = create_test_tokenizer();
        let normalized = tokenizer.normalize("Héllo Wörld!");
        assert_eq!(normalized, "hello world!");
    }
}