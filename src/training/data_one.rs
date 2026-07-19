//! ============================================================================
//! 数据加载与预处理模块（第一部分）
//! ============================================================================
//!
//! 本模块实现了数据集下载、加载和预处理功能：
//! - 支持HuggingFace、镜像站、自定义URL等多种下载源
//! - 支持断点续传和流式下载
//! - 支持Parquet、JSONL、CSV、TXT等多种数据格式
//! - 数据预处理和序列化
//! - 修复：正确处理Parquet中的所有列类型（嵌套结构、列表、结构体等）
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::config::{DatasetConfig, DownloadSource};
use crate::error::{Result, TrainError};
use crate::tokenizer::Tokenizer;
use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeStringArray, ListArray, StringArray, StructArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// 数据集加载器
// ============================================================================

pub struct DataLoader {
    config: DatasetConfig,
    cache_dir: PathBuf,
}

impl DataLoader {
    // ========================================================================
    // 创建与初始化
    // ========================================================================

    pub fn new(config: DatasetConfig) -> Self {
        let cache_dir = PathBuf::from(&config.cache_dir);
        let _ = fs::create_dir_all(&cache_dir);

        DataLoader { config, cache_dir }
    }

    // ========================================================================
    // 下载方法（保留原有实现）
    // ========================================================================

    /// 下载数据集（基础版本）
    pub async fn download(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for source in &self.config.mix {
            let source_path = self.download_source(source).await?;
            paths.push(source_path);
        }

        Ok(paths)
    }

    /// 流式下载数据集（支持断点续传）
    pub async fn download_streaming(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for source in &self.config.mix {
            println!("📥 下载数据集源: {}", source.name);
            let source_path = self.download_source_streaming(source).await?;
            paths.push(source_path);
        }

        Ok(paths)
    }

    // ========================================================================
    // 下载源处理（保留原有实现）
    // ========================================================================

    async fn download_source(&self, source: &crate::config::DatasetSource) -> Result<PathBuf> {
        let dataset_name = &source.name;
        let dataset_dir = self.cache_dir.join(dataset_name);
        fs::create_dir_all(&dataset_dir)?;

        match &self.config.download_source {
            DownloadSource::HuggingFace => {
                self.download_from_huggingface(dataset_name, &dataset_dir)
                    .await
            }
            DownloadSource::Mirror { url } => {
                self.download_from_mirror(dataset_name, url, &dataset_dir)
                    .await
            }
            DownloadSource::CustomUrl { url } => {
                self.download_from_custom_url(url, &dataset_dir).await
            }
            DownloadSource::Local => {
                if let Some(local_path) = &self.config.local_path {
                    Ok(PathBuf::from(local_path))
                } else {
                    Err(TrainError::Download("本地路径未指定".to_string()))
                }
            }
        }
    }

    async fn download_source_streaming(
        &self,
        source: &crate::config::DatasetSource,
    ) -> Result<PathBuf> {
        let dataset_name = &source.name;
        let dataset_dir = self.cache_dir.join(dataset_name);
        fs::create_dir_all(&dataset_dir)?;

        match &self.config.download_source {
            DownloadSource::HuggingFace => {
                self.stream_download_from_huggingface(dataset_name, &dataset_dir)
                    .await
            }
            DownloadSource::Mirror { url } => {
                self.stream_download_from_mirror(dataset_name, url, &dataset_dir)
                    .await
            }
            DownloadSource::CustomUrl { url } => {
                self.stream_download_from_url(url, &dataset_dir).await
            }
            DownloadSource::Local => {
                if let Some(local_path) = &self.config.local_path {
                    Ok(PathBuf::from(local_path))
                } else {
                    Err(TrainError::Download("本地路径未指定".to_string()))
                }
            }
        }
    }

    // ========================================================================
    // HuggingFace 下载（保留原有实现）
    // ========================================================================

    async fn download_from_huggingface(&self, dataset: &str, output_dir: &Path) -> Result<PathBuf> {
        let base_url = "https://huggingface.co/datasets";
        self.download_dataset_files_smart(base_url, dataset, output_dir)
            .await
    }

    async fn download_from_mirror(
        &self,
        dataset: &str,
        mirror: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let base_url = format!("{}/datasets", mirror);
        self.download_dataset_files_smart(&base_url, dataset, output_dir)
            .await
    }

    async fn download_from_custom_url(&self, url: &str, output_dir: &Path) -> Result<PathBuf> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .user_agent("shlteLLM/3.0.0")
            .build()?;

        let response = client.get(url).send().await?;
        let content = response.bytes().await?;
        let filename = url.split('/').next_back().unwrap_or("dataset.txt");
        let filepath = output_dir.join(filename);

        let mut file = File::create(&filepath)?;
        file.write_all(&content)?;

        println!("✅ 下载完成: {}", filepath.display());
        Ok(filepath)
    }

    // ========================================================================
    // 智能下载（保留原有实现）
    // ========================================================================

    async fn download_dataset_files_smart(
        &self,
        base_url: &str,
        dataset: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let mut downloaded_paths = Vec::new();

        let api_url = format!("https://huggingface.co/api/datasets/{}", dataset);

        if let Ok(response) = reqwest::Client::new()
            .get(&api_url)
            .header("User-Agent", "shlteLLM/3.0.0")
            .send()
            .await
        {
            if response.status().is_success() {
                if let Ok(json) = response.json::<serde_json::Value>().await {
                    if let Some(configs) = json.get("configs").and_then(|c| c.as_array()) {
                        for config in configs {
                            if let Some(data_files) = config.get("data_files") {
                                let files = self.extract_filenames_from_data_files(data_files);
                                for filename in files {
                                    if let Ok(path) = self
                                        .try_download_file_with_branches(
                                            base_url, dataset, &filename, output_dir,
                                        )
                                        .await
                                    {
                                        downloaded_paths.push(path);
                                    }
                                }
                            }

                            if let Some(splits) = config.get("splits") {
                                let files = self.extract_filenames_from_splits(splits);
                                for filename in files {
                                    if let Ok(path) = self
                                        .try_download_file_with_branches(
                                            base_url, dataset, &filename, output_dir,
                                        )
                                        .await
                                    {
                                        downloaded_paths.push(path);
                                    }
                                }
                            }

                            if !downloaded_paths.is_empty() {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if downloaded_paths.is_empty() {
            downloaded_paths = self
                .download_by_common_patterns(base_url, dataset, output_dir)
                .await?;
        }

        if downloaded_paths.is_empty() {
            downloaded_paths = self
                .download_by_recursive_search(base_url, dataset, output_dir)
                .await?;
        }

        if downloaded_paths.is_empty() {
            Err(TrainError::Download(format!("无法下载数据集: {}", dataset)))
        } else {
            Ok(downloaded_paths[0].clone())
        }
    }

    // ========================================================================
    // 文件列表提取（保留原有实现）
    // ========================================================================

    fn extract_filenames_from_data_files(&self, data_files: &serde_json::Value) -> Vec<String> {
        let mut filenames = Vec::new();

        match data_files {
            serde_json::Value::String(s) => {
                filenames.push(s.clone());
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        filenames.push(s.to_string());
                    } else if let Some(obj) = item.as_object() {
                        if let Some(filename) = obj.get("filename").and_then(|f| f.as_str()) {
                            filenames.push(filename.to_string());
                        }
                    }
                }
            }
            serde_json::Value::Object(obj) => {
                for (_split, file) in obj {
                    if let Some(s) = file.as_str() {
                        filenames.push(s.to_string());
                    }
                }
            }
            _ => {}
        }

        filenames
    }

    fn extract_filenames_from_splits(&self, splits: &serde_json::Value) -> Vec<String> {
        let mut filenames = Vec::new();

        if let Some(arr) = splits.as_array() {
            for split in arr {
                if let Some(name) = split.get("name").and_then(|n| n.as_str()) {
                    let patterns = vec![
                        format!("{}.parquet", name),
                        format!("{}-00000-of-00001.parquet", name),
                        format!("{}/{}", name, name),
                        format!("{}/{}.parquet", name, name),
                    ];
                    filenames.extend(patterns);
                }
            }
        }

        filenames
    }

    // ========================================================================
    // 分支下载（保留原有实现）
    // ========================================================================

    async fn try_download_file_with_branches(
        &self,
        base_url: &str,
        dataset: &str,
        filepath: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let branches = ["main", "master", "data", "refs/convert/parquet"];

        for branch in &branches {
            let url = if branch == &"refs/convert/parquet" {
                format!("{}/{}/{}/{}", base_url, dataset, branch, filepath)
            } else {
                format!("{}/{}/resolve/{}/{}", base_url, dataset, branch, filepath)
            };

            let filename = filepath.split('/').next_back().unwrap_or(filepath);

            match self.try_download_file(&url, output_dir, filename).await {
                Ok(path) => return Ok(path),
                Err(_) => continue,
            }
        }

        Err(TrainError::Download(format!("无法下载文件: {}", filepath)))
    }

    // ========================================================================
    // 常见模式下载（保留原有实现）
    // ========================================================================

    async fn download_by_common_patterns(
        &self,
        base_url: &str,
        dataset: &str,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        let patterns = vec![
            "train.parquet",
            "validation.parquet",
            "test.parquet",
            "data.parquet",
            "dataset.parquet",
            "train-00000-of-00001.parquet",
            "data-00000-of-00001.parquet",
            "data/train.parquet",
            "data/validation.parquet",
            "data/data.parquet",
            "train.jsonl",
            "data.jsonl",
            "dataset.jsonl",
            "train.csv",
            "data.csv",
        ];

        for pattern in &patterns {
            let url = format!("{}/{}/resolve/main/{}", base_url, dataset, pattern);
            let filename = pattern.split('/').next_back().unwrap_or(pattern);

            if let Ok(path) = self.try_download_file(&url, output_dir, filename).await {
                paths.push(path);
            }
        }

        if paths.is_empty() {
            paths = self
                .download_all_shards(dataset, base_url, output_dir)
                .await?;
        }

        Ok(paths)
    }

    // ========================================================================
    // 递归搜索下载（保留原有实现）
    // ========================================================================

    async fn download_by_recursive_search(
        &self,
        base_url: &str,
        dataset: &str,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let mut stack = vec![base_url.to_string()];

        while let Some(current_url) = stack.pop() {
            let api_url = format!("{}/resolve/main/", current_url);

            if let Ok(response) = reqwest::Client::new()
                .get(&api_url)
                .header("User-Agent", "shlteLLM/3.0.0")
                .send()
                .await
            {
                if response.status().is_success() {
                    if let Ok(files) = response.json::<Vec<serde_json::Value>>().await {
                        for file in files {
                            let file_type = file.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            let path_str = file.get("path").and_then(|p| p.as_str()).unwrap_or("");

                            if file_type == "file" || file_type == "blob" {
                                if path_str.ends_with(".parquet")
                                    || path_str.ends_with(".jsonl")
                                    || path_str.ends_with(".csv")
                                {
                                    if let Ok(file_path) = self
                                        .try_download_file_with_branches(
                                            base_url, dataset, path_str, output_dir,
                                        )
                                        .await
                                    {
                                        paths.push(file_path);
                                    }
                                }
                            } else if file_type == "directory" {
                                let sub_base =
                                    format!("{}/{}/resolve/main/{}", base_url, dataset, path_str);
                                stack.push(sub_base);
                            }
                        }
                    }
                }
            }
        }

        Ok(paths)
    }

    // ========================================================================
    // 分片下载（保留原有实现）
    // ========================================================================

    pub async fn download_all_shards(
        &self,
        dataset: &str,
        base_url: &str,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut downloaded_paths = Vec::new();

        let shard_patterns = [
            ("train-{:05}-of-{:05}.parquet", 0..32, 32),
            ("data-{:05}-of-{:05}.parquet", 0..32, 32),
            ("train-{:05}-of-{:05}.parquet", 0..16, 16),
            ("data-{:05}-of-{:05}.parquet", 0..16, 16),
            ("part-{:05}.parquet", 0..10, 0),
            ("train-{:05}.parquet", 0..10, 0),
            ("data-{:05}.parquet", 0..10, 0),
        ];

        for (pattern, range, total) in &shard_patterns {
            for shard in range.clone() {
                let filename = if *total > 0 {
                    pattern
                        .replace("{:05}", &format!("{:05}", shard))
                        .replace("{:05}", &format!("{:05}", total))
                } else {
                    pattern.replace("{:05}", &format!("{:05}", shard))
                };

                let url = format!("{}/{}/resolve/main/{}", base_url, dataset, filename);

                match self.try_download_file(&url, output_dir, &filename).await {
                    Ok(filepath) => {
                        downloaded_paths.push(filepath);
                        println!("   📦 下载分片: {}", filename);
                    }
                    Err(_) => {
                        if shard == 0 {
                            break;
                        }
                    }
                }
            }

            if !downloaded_paths.is_empty() {
                break;
            }
        }

        if downloaded_paths.is_empty() {
            Err(TrainError::Download(format!(
                "无法下载任何分片: {}",
                dataset
            )))
        } else {
            Ok(downloaded_paths)
        }
    }

    // ========================================================================
    // 单文件下载（保留原有实现）
    // ========================================================================

    async fn try_download_file(
        &self,
        url: &str,
        output_dir: &Path,
        filename: &str,
    ) -> Result<PathBuf> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .user_agent("shlteLLM/3.0.0")
            .build()?;

        let response = client.get(url).send().await?;

        if !response.status().is_success() {
            return Err(TrainError::Download(format!(
                "HTTP {}: {}",
                response.status(),
                url
            )));
        }

        let filepath = output_dir.join(filename);
        let content = response.bytes().await?;

        let mut file = File::create(&filepath)?;
        file.write_all(&content)?;

        println!("✅ 下载: {} -> {}", url, filepath.display());
        Ok(filepath)
    }

    // ========================================================================
    // 流式下载（保留原有实现）
    // ========================================================================

    async fn stream_download_from_huggingface(
        &self,
        dataset: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let base_url = "https://huggingface.co/datasets";
        self.stream_download_files_smart(base_url, dataset, output_dir)
            .await
    }

    async fn stream_download_from_mirror(
        &self,
        dataset: &str,
        mirror: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let base_url = format!("{}/datasets", mirror);
        self.stream_download_files_smart(&base_url, dataset, output_dir)
            .await
    }

    async fn stream_download_files_smart(
        &self,
        base_url: &str,
        dataset: &str,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        match self
            .download_dataset_files_smart(base_url, dataset, output_dir)
            .await
        {
            Ok(path) => Ok(path),
            Err(_) => {
                let url = format!("{}/{}/resolve/main/train.parquet", base_url, dataset);
                self.stream_download_from_url(&url, output_dir).await
            }
        }
    }

    async fn stream_download_from_url(&self, url: &str, output_dir: &Path) -> Result<PathBuf> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .user_agent("shlteLLM/3.0.0")
            .build()?;

        let filename = url.split('/').next_back().unwrap_or("dataset.bin");
        let filepath = output_dir.join(filename);

        let mut downloaded_bytes = 0u64;
        if filepath.exists() {
            downloaded_bytes = fs::metadata(&filepath)?.len();
        }

        let mut request = client.get(url);

        if downloaded_bytes > 0 {
            request = request.header("Range", format!("bytes={}-", downloaded_bytes));
        }

        let response = request.send().await?;

        if response.status().is_success()
            || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
        {
            let total_size = response.content_length().unwrap_or(0) + downloaded_bytes;

            let pb = ProgressBar::new(total_size);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"));

            let mut file = if downloaded_bytes > 0 {
                fs::OpenOptions::new().append(true).open(&filepath)?
            } else {
                File::create(&filepath)?
            };

            let mut stream = response.bytes_stream();
            use futures_util::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk)?;
                pb.inc(chunk.len() as u64);
            }

            pb.finish_with_message("下载完成");
            Ok(filepath)
        } else {
            Err(TrainError::Download(format!(
                "下载失败: HTTP {}",
                response.status()
            )))
        }
    }
}

// ============================================================================
// Parquet 文本提取器（修复 P1-2）
// ============================================================================

/// Parquet 列类型（用于识别可提取文本的列）
#[derive(Debug, Clone, PartialEq)]
enum ParquetColumnType {
    String,
    LargeString,
    Binary,
    ListOfString,
    StructWithText,
    Numeric,
    Boolean,
    Null,
    Unknown,
}

/// 从 Parquet 中提取文本的配置
#[derive(Debug, Clone)]
pub struct ParquetTextExtractor {
    /// 优先搜索的列名模式
    pub text_column_patterns: Vec<String>,
    /// 是否递归搜索嵌套结构
    pub recurse_nested: bool,
    /// 最大嵌套深度
    pub max_depth: usize,
    /// 是否将数字转换为文本
    pub convert_numbers: bool,
    /// 是否将布尔值转换为文本
    pub convert_booleans: bool,
    /// 列表项分隔符
    pub list_separator: String,
    /// 结构体字段分隔符
    pub struct_separator: String,
}

impl Default for ParquetTextExtractor {
    fn default() -> Self {
        ParquetTextExtractor {
            text_column_patterns: vec![
                "text".to_string(),
                "content".to_string(),
                "data".to_string(),
                "sentence".to_string(),
                "document".to_string(),
                "input".to_string(),
                "output".to_string(),
                "message".to_string(),
                "description".to_string(),
                "title".to_string(),
                "body".to_string(),
                "paragraph".to_string(),
                "passage".to_string(),
                "article".to_string(),
                "dialogue".to_string(),
                "utterance".to_string(),
                "response".to_string(),
                "prompt".to_string(),
                "completion".to_string(),
            ],
            recurse_nested: true,
            max_depth: 5,
            convert_numbers: true,
            convert_booleans: true,
            list_separator: " ".to_string(),
            struct_separator: " ".to_string(),
        }
    }
}

impl ParquetTextExtractor {
    /// 创建新的提取器
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加列名模式
    pub fn add_pattern(&mut self, pattern: &str) {
        self.text_column_patterns.push(pattern.to_string());
    }

    /// 检查列名是否匹配文本列模式
    fn is_text_column(&self, column_name: &str) -> bool {
        let lower_name = column_name.to_lowercase();
        self.text_column_patterns
            .iter()
            .any(|pattern| lower_name.contains(pattern))
    }

    /// 从 Arrow RecordBatch 中提取文本
    pub fn extract_text_from_batch(
        &self,
        batch: &RecordBatch,
        row_idx: usize,
    ) -> Result<Option<String>> {
        for (col_idx, column) in batch.columns().iter().enumerate() {
            let schema = batch.schema();
            let field = schema.field(col_idx);
            let col_name = field.name();
            
            if self.is_text_column(col_name) {
                if let Some(text) = self.extract_from_column(column, row_idx, 0)? {
                    return Ok(Some(text));
                }
            }
        }
        Ok(None)
    }

    /// 从单个列中提取文本
    fn extract_from_column(
        &self,
        column: &ArrayRef,
        row_idx: usize,
        depth: usize,
    ) -> Result<Option<String>> {
        if depth > self.max_depth {
            return Ok(None);
        }

        match column.data_type() {
            DataType::Utf8 => {
                let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::LargeUtf8 => {
                let array = column.as_any().downcast_ref::<LargeStringArray>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Binary => {
                let array = column.as_any().downcast_ref::<BinaryArray>().unwrap();
                if !array.is_null(row_idx) {
                    if let Ok(text) = String::from_utf8(array.value(row_idx).to_vec()) {
                        return Ok(Some(text));
                    }
                }
            }
            DataType::Boolean if self.convert_booleans => {
                let array = column.as_any().downcast_ref::<BooleanArray>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Int8 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Int8Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Int16 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Int16Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Int32 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Int32Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Int64 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::UInt8 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<UInt8Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::UInt16 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<UInt16Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::UInt32 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<UInt32Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::UInt64 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Float32 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Float32Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::Float64 if self.convert_numbers => {
                let array = column.as_any().downcast_ref::<Float64Array>().unwrap();
                if !array.is_null(row_idx) {
                    return Ok(Some(array.value(row_idx).to_string()));
                }
            }
            DataType::List(field) if self.recurse_nested => {
                let list_array = column.as_any().downcast_ref::<ListArray>().unwrap();
                if !list_array.is_null(row_idx) {
                    let offsets = list_array.value_offsets();
                    let list_start = offsets[row_idx] as usize;
                    let list_end = offsets[row_idx + 1] as usize;
                    let values = list_array.values();

                    let mut items = Vec::new();
                    for i in list_start..list_end {
                        if let Some(item_text) = self.extract_from_column(values, i, depth + 1)? {
                            items.push(item_text);
                        }
                    }
                    
                    if !items.is_empty() {
                        return Ok(Some(items.join(&self.list_separator)));
                    }
                }
            }
            DataType::Struct(fields) if self.recurse_nested => {
                let struct_array = column.as_any().downcast_ref::<StructArray>().unwrap();
                let mut struct_texts = Vec::new();
                
                for (field_idx, _field) in fields.iter().enumerate() {
                    let child_col = struct_array.column(field_idx);
                    if let Some(child_text) = self.extract_from_column(child_col, row_idx, depth + 1)? {
                        struct_texts.push(child_text);
                    }
                }
                
                if !struct_texts.is_empty() {
                    return Ok(Some(struct_texts.join(&self.struct_separator)));
                }
            }
            DataType::Map(field, _) if self.recurse_nested => {
                // Map 类型视为 List of Struct
                let list_array = column.as_any().downcast_ref::<ListArray>().unwrap();
                if !list_array.is_null(row_idx) {
                    let offsets = list_array.value_offsets();
                    let list_start = offsets[row_idx] as usize;
                    let list_end = offsets[row_idx + 1] as usize;
                    let values = list_array.values();
                    
                    let mut entries = Vec::new();
                    for i in list_start..list_end {
                        if let Some(entry_text) = self.extract_from_column(values, i, depth + 1)? {
                            entries.push(entry_text);
                        }
                    }
                    
                    if !entries.is_empty() {
                        return Ok(Some(entries.join(&self.list_separator)));
                    }
                }
            }
            _ => {}
        }

        Ok(None)
    }

    /// 获取所有文本列的索引
    pub fn get_text_column_indices(&self, schema: &arrow::datatypes::Schema) -> Vec<usize> {
        let mut indices = Vec::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            if self.is_text_column(field.name()) || self.can_extract_from_type(field.data_type()) {
                indices.push(idx);
            }
        }
        indices
    }

    /// 检查是否可以从该类型提取文本
    fn can_extract_from_type(&self, data_type: &DataType) -> bool {
        match data_type {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary => true,
            DataType::Boolean => self.convert_booleans,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => self.convert_numbers,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => self.convert_numbers,
            DataType::Float32 | DataType::Float64 => self.convert_numbers,
            DataType::List(field) if self.recurse_nested => self.can_extract_from_type(field.data_type()),
            DataType::Struct(fields) if self.recurse_nested => {
                fields.iter().any(|f| self.can_extract_from_type(f.data_type()))
            }
            DataType::Map(field, _) if self.recurse_nested => {
                self.can_extract_from_type(field.data_type())
            }
            _ => false,
        }
    }
}

// ============================================================================
// 数据预处理器
// ============================================================================

pub struct DataPreprocessor {
    tokenizer: Tokenizer,
    max_sequence_length: usize,
    min_sequence_length: usize,
    parquet_extractor: ParquetTextExtractor,
}

impl DataPreprocessor {
    // ========================================================================
    // 创建
    // ========================================================================

    pub fn new(tokenizer: Tokenizer, max_sequence_length: usize) -> Self {
        DataPreprocessor {
            tokenizer,
            max_sequence_length,
            min_sequence_length: 4,
            parquet_extractor: ParquetTextExtractor::new(),
        }
    }

    /// 创建带自定义 Parquet 提取器的预处理器
    pub fn with_extractor(
        tokenizer: Tokenizer,
        max_sequence_length: usize,
        extractor: ParquetTextExtractor,
    ) -> Self {
        DataPreprocessor {
            tokenizer,
            max_sequence_length,
            min_sequence_length: 4,
            parquet_extractor: extractor,
        }
    }

    // ========================================================================
    // 文件预处理入口
    // ========================================================================

    pub fn preprocess_file(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        let extension = input_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("txt");

        match extension {
            "parquet" => self.preprocess_parquet_enhanced(input_path, output_path),
            "jsonl" | "json" => self.preprocess_jsonl(input_path, output_path),
            "csv" => self.preprocess_csv(input_path, output_path),
            _ => self.preprocess_text(input_path, output_path),
        }
    }

    // ========================================================================
    // Parquet文件预处理（修复版 - 支持所有列类型）
    // ========================================================================

    fn preprocess_parquet_enhanced(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        println!("   📊 处理 Parquet 文件: {}", input_path.display());
        
        // 使用 Arrow Reader 读取 Parquet（支持所有类型）
        let file = File::open(input_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        
        let schema = builder.schema();
        let text_column_indices = self.parquet_extractor.get_text_column_indices(schema);
        
        println!("   📋 检测到 {} 个可提取文本的列", text_column_indices.len());
        
        // 打印列信息
        for &idx in &text_column_indices {
            let field = schema.field(idx);
            println!("      - {} ({:?})", field.name(), field.data_type());
        }
        
        let reader = builder.build()?;
        let mut all_tokens = Vec::new();
        let mut total_chars = 0;
        let mut rows_processed = 0;
        let mut rows_skipped = 0;
        
        for batch_result in reader {
            let batch = batch_result?;
            let num_rows = batch.num_rows();
            
            for row_idx in 0..num_rows {
                let mut row_text = None;
                
                // 优先使用检测到的文本列
                for &col_idx in &text_column_indices {
                    let column = batch.column(col_idx);
                    if let Some(text) = self.parquet_extractor.extract_from_column(
                        column, row_idx, 0,
                    )? {
                        if !text.trim().is_empty() {
                            row_text = Some(text);
                            break;
                        }
                    }
                }
                
                // 如果没有找到，尝试所有列
                if row_text.is_none() && self.parquet_extractor.recurse_nested {
                    for col_idx in 0..batch.num_columns() {
                        let column = batch.column(col_idx);
                        if let Some(text) = self.parquet_extractor.extract_from_column(
                            column, row_idx, 0,
                        )? {
                            if !text.trim().is_empty() {
                                row_text = Some(text);
                                break;
                            }
                        }
                    }
                }
                
                if let Some(text) = row_text {
                    if text.trim().is_empty() {
                        rows_skipped += 1;
                        continue;
                    }
                    
                    total_chars += text.len();
                    let tokens = self.tokenizer.encode(&text);
                    all_tokens.extend(tokens);
                    rows_processed += 1;
                } else {
                    rows_skipped += 1;
                }
            }
        }
        
        if rows_skipped > 0 {
            println!("   ⚠️ 跳过 {} 行（无有效文本）", rows_skipped);
        }
        
        if rows_processed == 0 {
            return Err(TrainError::Preprocessing(format!(
                "Parquet 文件 {} 中没有找到有效的文本数据",
                input_path.display()
            )));
        }
        
        self.save_preprocessed_data(&all_tokens, output_path, total_chars, rows_processed)
    }

    /// 原始 Parquet 预处理（保留作为备用）
    #[allow(dead_code)]
    fn preprocess_parquet(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        self.preprocess_parquet_enhanced(input_path, output_path)
    }

    // ========================================================================
    // JSONL文件预处理（增强版 - 处理嵌套JSON）
    // ========================================================================

    fn preprocess_jsonl(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);

        let mut all_tokens = Vec::new();
        let mut total_chars = 0;
        let mut skipped_lines = 0;
        let mut processed_lines = 0;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let text = self.extract_text_from_json_line(&line);
            
            if text.trim().is_empty() {
                skipped_lines += 1;
                if skipped_lines % 1000 == 0 && skipped_lines > 0 {
                    println!("   ⚠️ 已跳过 {} 行无效数据", skipped_lines);
                }
                continue;
            }

            total_chars += text.len();
            let tokens = self.tokenizer.encode(&text);
            all_tokens.extend(tokens);
            processed_lines += 1;
        }

        if skipped_lines > 0 {
            println!("   ⚠️ 共跳过 {} 行无法解析的数据", skipped_lines);
        }
        
        if processed_lines == 0 {
            return Err(TrainError::Preprocessing(format!(
                "JSONL 文件 {} 中没有找到有效的文本数据",
                input_path.display()
            )));
        }

        self.save_preprocessed_data(&all_tokens, output_path, total_chars, processed_lines)
    }

    /// 从 JSON 行中提取文本（递归处理嵌套结构）
    fn extract_text_from_json_line(&self, line: &str) -> String {
        // 尝试解析为 JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            return self.extract_text_from_json_value(&json);
        }
        
        // 如果不是 JSON，当作纯文本处理
        let trimmed = line.trim();
        if trimmed.chars().all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace()) {
            trimmed.to_string()
        } else {
            String::new()
        }
    }

    /// 从 JSON Value 中递归提取文本
    fn extract_text_from_json_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(obj) => {
                // 按优先级查找文本字段
                let text_fields = [
                    "text", "content", "data", "sentence", "document", 
                    "input", "output", "message", "description", "title",
                    "body", "paragraph", "passage", "article", "dialogue",
                    "utterance", "response", "prompt", "completion"
                ];
                
                for &field in &text_fields {
                    if let Some(v) = obj.get(field) {
                        let extracted = self.extract_text_from_json_value(v);
                        if !extracted.trim().is_empty() {
                            return extracted;
                        }
                    }
                }
                
                // 如果对象字段很少，尝试合并所有字符串值
                if obj.len() <= 5 {
                    let mut texts = Vec::new();
                    for v in obj.values() {
                        let extracted = self.extract_text_from_json_value(v);
                        if !extracted.trim().is_empty() {
                            texts.push(extracted);
                        }
                    }
                    return texts.join(" ");
                }
                
                String::new()
            }
            serde_json::Value::Array(arr) => {
                let texts: Vec<String> = arr
                    .iter()
                    .filter_map(|v| {
                        let extracted = self.extract_text_from_json_value(v);
                        if extracted.trim().is_empty() {
                            None
                        } else {
                            Some(extracted)
                        }
                    })
                    .collect();
                texts.join(" ")
            }
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => String::new(),
        }
    }

    // ========================================================================
    // CSV文件预处理
    // ========================================================================

    fn preprocess_csv(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        let mut reader = csv::Reader::from_path(input_path)?;

        let headers = reader.headers()?.clone();
        let text_column_names = [
            "text", "content", "data", "sentence", "document", "input", "output",
            "message", "description", "title", "body",
        ];

        let text_col_idx = headers
            .iter()
            .position(|h| {
                let h = h.to_lowercase();
                text_column_names.iter().any(|&name| h.contains(name))
            })
            .unwrap_or(0);

        println!(
            "   📊 CSV检测到文本列: {} (索引: {})",
            headers.get(text_col_idx).unwrap_or("未知"),
            text_col_idx
        );

        let mut all_tokens = Vec::new();
        let mut total_chars = 0;
        let mut processed_rows = 0;

        for result in reader.records() {
            let record = result?;
            if let Some(text) = record.get(text_col_idx) {
                if text.trim().is_empty() {
                    continue;
                }

                total_chars += text.len();
                let tokens = self.tokenizer.encode(text);
                all_tokens.extend(tokens);
                processed_rows += 1;
            }
        }

        if processed_rows == 0 {
            return Err(TrainError::Preprocessing(format!(
                "CSV 文件 {} 中没有找到有效的文本数据",
                input_path.display()
            )));
        }

        self.save_preprocessed_data(&all_tokens, output_path, total_chars, processed_rows)
    }

    // ========================================================================
    // 纯文本文件预处理
    // ========================================================================

    fn preprocess_text(&self, input_path: &Path, output_path: &Path) -> Result<DataStats> {
        // 检查是否为Parquet文件（通过magic number）
        let mut file = File::open(input_path)?;
        let mut header = [0u8; 4];

        if file.read_exact(&mut header).is_ok() && &header == b"PAR1" {
            drop(file);
            return self.preprocess_parquet_enhanced(input_path, output_path);
        }

        let file = File::open(input_path)?;
        let reader = BufReader::new(file);

        let mut all_tokens = Vec::new();
        let mut total_chars = 0;
        let mut processed_lines = 0;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            total_chars += line.len();
            let tokens = self.tokenizer.encode(&line);
            all_tokens.extend(tokens);
            processed_lines += 1;
        }

        if processed_lines == 0 {
            return Err(TrainError::Preprocessing(format!(
                "文本文件 {} 中没有找到有效的文本数据",
                input_path.display()
            )));
        }

        self.save_preprocessed_data(&all_tokens, output_path, total_chars, processed_lines)
    }

    // ========================================================================
    // 保存预处理数据（增强版）
    // ========================================================================

    fn save_preprocessed_data(
        &self,
        all_tokens: &[usize],
        output_path: &Path,
        total_chars: usize,
        num_samples: usize,
    ) -> Result<DataStats> {
        let sequence_len = self.max_sequence_length + 1;

        // 将token序列分割成固定长度的块
        let sequences: Vec<&[usize]> = all_tokens
            .chunks(sequence_len)
            .filter(|chunk| chunk.len() == sequence_len)
            .collect();

        let total_sequences = sequences.len();

        if total_sequences == 0 {
            return Err(TrainError::Preprocessing(format!(
                "没有足够的token来形成序列（需要至少{}个token）",
                sequence_len
            )));
        }

        // 写入文件
        let mut output = File::create(output_path)?;
        for sequence in &sequences {
            let line = sequence
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(output, "{}", line)?;
        }

        let avg_seq_length = all_tokens.len() as f64 / total_sequences as f64;

        println!("   ✅ 预处理完成:");
        println!("      样本数: {}", num_samples);
        println!("      总字符数: {}", total_chars);
        println!("      总token数: {}", all_tokens.len());
        println!("      序列数: {}", total_sequences);
        println!("      平均序列长度: {:.1}", avg_seq_length);
        println!("      输出文件: {}", output_path.display());

        Ok(DataStats {
            num_samples,
            num_tokens: all_tokens.len(),
            num_sequences: total_sequences,
            avg_sequence_length: avg_seq_length,
            vocab_size: self.tokenizer.vocab_size(),
        })
    }

    // ========================================================================
    // 目录预处理
    // ========================================================================

    pub fn preprocess_directory(
        &self,
        input_dir: &Path,
        output_dir: &Path,
    ) -> Result<Vec<DataStats>> {
        fs::create_dir_all(output_dir)?;

        let mut stats = Vec::new();
        let entries: Vec<_> = fs::read_dir(input_dir)?.filter_map(|e| e.ok()).collect();

        let pb = ProgressBar::new(entries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for entry in entries {
            let input_path = entry.path();

            if input_path.is_file() {
                let filename = input_path.file_stem().unwrap_or_default();
                let output_path =
                    output_dir.join(format!("{}.preprocessed.txt", filename.to_string_lossy()));

                pb.set_message(format!("处理: {}", filename.to_string_lossy()));

                match self.preprocess_file(&input_path, &output_path) {
                    Ok(file_stats) => {
                        stats.push(file_stats);
                    }
                    Err(e) => {
                        eprintln!("⚠️  处理文件 {} 失败: {}", input_path.display(), e);
                    }
                }

                pb.inc(1);
            }
        }

        pb.finish_with_message("预处理完成");
        Ok(stats)
    }
}

// ============================================================================
// 数据统计结构（扩展版）
// ============================================================================

#[derive(Debug, Clone)]
pub struct DataStats {
    pub num_samples: usize,
    pub num_tokens: usize,
    pub num_sequences: usize,
    pub avg_sequence_length: f64,
    pub vocab_size: usize,
}

impl DataStats {
    pub fn print_summary(&self) {
        println!("📊 数据统计:");
        println!("   样本数: {}", self.num_samples);
        println!("   Token数: {}", self.num_tokens);
        println!("   序列数: {}", self.num_sequences);
        println!("   平均序列长度: {:.1}", self.avg_sequence_length);
        println!("   词表大小: {}", self.vocab_size);
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
    fn test_parquet_extractor_default() {
        let extractor = ParquetTextExtractor::default();
        assert!(extractor.text_column_patterns.contains(&"text".to_string()));
        assert!(extractor.recurse_nested);
        assert_eq!(extractor.max_depth, 5);
    }

    #[test]
    fn test_parquet_extractor_add_pattern() {
        let mut extractor = ParquetTextExtractor::default();
        extractor.add_pattern("custom_field");
        assert!(extractor.text_column_patterns.contains(&"custom_field".to_string()));
    }

    #[test]
    fn test_is_text_column() {
        let extractor = ParquetTextExtractor::default();
        assert!(extractor.is_text_column("text_content"));
        assert!(extractor.is_text_column("my_sentence"));
        assert!(!extractor.is_text_column("id"));
        assert!(!extractor.is_text_column("timestamp"));
    }

    #[test]
    fn test_json_text_extraction() {
        let preprocessor = DataPreprocessor::new(
            Tokenizer::from_config(&crate::config::TokenizerConfig {
                algorithm: crate::config::TokenizationAlgorithm::BPE,
                vocab_size: 1000,
                special_tokens: crate::config::SpecialTokens::default(),
                normalization: true,
                add_prefix_space: false,
            }).unwrap(),
            128,
        );
        
        let json_line = r#"{"text": "Hello world", "id": 1}"#;
        let text = preprocessor.extract_text_from_json_line(json_line);
        assert_eq!(text, "Hello world");
        
        let nested_json = r#"{"data": {"content": "Nested text", "meta": "info"}}"#;
        let text = preprocessor.extract_text_from_json_line(nested_json);
        assert_eq!(text, "Nested text");
        
        let array_json = r#"{"messages": ["Hello", "World"]}"#;
        let text = preprocessor.extract_text_from_json_line(array_json);
        assert_eq!(text, "Hello World");
    }
}