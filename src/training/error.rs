#![allow(dead_code)]
//! ============================================================================
//! 错误处理模块
//! ============================================================================
//!
//! 本模块定义了训练框架中使用的所有错误类型，提供统一的错误处理接口。
//! 包含错误类型枚举、错误严重程度分类、错误建议系统以及结果扩展trait。
//!
//! 修复内容（P2-1）：
//! - 统一错误处理风格
//! - 添加错误上下文追踪
//! - 添加错误链支持
//! - 添加结构化日志记录
//!
//! ============================================================================

use std::io;
use thiserror::Error;
use std::backtrace::Backtrace;
use std::fmt;

// ============================================================================
// 主要错误类型枚举
// ============================================================================

#[derive(Error, Debug)]
pub enum TrainError {
    #[error("IO错误: {0}")]
    Io(#[from] io::Error),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("数据错误: {0}")]
    Data(String),

    #[error("模型错误: {0}")]
    Model(String),

    #[error("训练错误: {0}")]
    Training(String),

    #[error("分词器错误: {0}")]
    Tokenizer(String),

    #[error("数据库错误: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("序列化错误: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("网络错误: {0}")]
    Network(String),

    #[error("下载错误: {0}")]
    Download(String),

    #[error("预处理错误: {0}")]
    Preprocessing(String),

    #[error("验证错误: {0}")]
    Validation(String),

    #[error("设备错误: {0}")]
    Device(String),

    #[error("解析错误: {0}")]
    Parse(String),

    #[error("检查点错误: {0}")]
    Checkpoint(String),

    #[error("分布式错误: {0}")]
    Distributed(String),

    #[error("内存不足: {0}")]
    OutOfMemory(String),

    #[error("梯度溢出: {0}")]
    GradientOverflow(String),

    #[error("GPU错误: {0}")]
    Gpu(String),

    #[error("超时错误: {0}")]
    Timeout(String),

    #[error("未知错误: {0}")]
    Unknown(String),

    #[error("错误链: {0}")]
    #[cfg(feature = "onnx")]
    WithChain {
        #[source]
        source: Box<TrainError>,
        context: String,
    },
}

// ============================================================================
// 错误上下文（增强版）
// ============================================================================

/// 带上下文的错误
#[derive(Debug)]
pub struct ContextError {
    pub error: TrainError,
    pub context: String,
    pub file: &'static str,
    pub line: u32,
    pub backtrace: Backtrace,
}

impl ContextError {
    pub fn new(error: TrainError, context: &str, file: &'static str, line: u32) -> Self {
        ContextError {
            error,
            context: context.to_string(),
            file,
            line,
            backtrace: Backtrace::capture(),
        }
    }
    
    pub fn into_error(self) -> TrainError {
        self.error
    }
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}:{}] {}: {}",
            self.file, self.line, self.context, self.error
        )
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// ============================================================================
// 类型别名
// ============================================================================

pub type Result<T> = std::result::Result<T, TrainError>;

// ============================================================================
// 标准类型转换实现
// ============================================================================

impl From<&str> for TrainError {
    fn from(s: &str) -> Self {
        TrainError::Unknown(s.to_string())
    }
}

impl From<String> for TrainError {
    fn from(s: String) -> Self {
        TrainError::Unknown(s)
    }
}

impl From<parquet::errors::ParquetError> for TrainError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        TrainError::Parse(format!("Parquet错误: {}", e))
    }
}

impl From<arrow::error::ArrowError> for TrainError {
    fn from(e: arrow::error::ArrowError) -> Self {
        TrainError::Parse(format!("Arrow错误: {}", e))
    }
}

impl From<csv::Error> for TrainError {
    fn from(e: csv::Error) -> Self {
        TrainError::Parse(format!("CSV错误: {}", e))
    }
}

impl From<toml::ser::Error> for TrainError {
    fn from(e: toml::ser::Error) -> Self {
        TrainError::Config(format!("TOML序列化错误: {}", e))
    }
}

impl From<toml::de::Error> for TrainError {
    fn from(e: toml::de::Error) -> Self {
        TrainError::Config(format!("TOML反序列化错误: {}", e))
    }
}

impl From<std::time::SystemTimeError> for TrainError {
    fn from(e: std::time::SystemTimeError) -> Self {
        TrainError::Unknown(format!("系统时间错误: {}", e))
    }
}

impl From<std::string::FromUtf8Error> for TrainError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        TrainError::Parse(format!("UTF8解析错误: {}", e))
    }
}

impl From<std::num::ParseIntError> for TrainError {
    fn from(e: std::num::ParseIntError) -> Self {
        TrainError::Parse(format!("整数解析错误: {}", e))
    }
}

impl From<std::num::ParseFloatError> for TrainError {
    fn from(e: std::num::ParseFloatError) -> Self {
        TrainError::Parse(format!("浮点数解析错误: {}", e))
    }
}

impl From<std::array::TryFromSliceError> for TrainError {
    fn from(e: std::array::TryFromSliceError) -> Self {
        TrainError::Parse(format!("切片转换错误: {}", e))
    }
}

impl From<tokio::task::JoinError> for TrainError {
    fn from(e: tokio::task::JoinError) -> Self {
        TrainError::Unknown(format!("Tokio任务错误: {}", e))
    }
}

impl From<reqwest::Error> for TrainError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            TrainError::Timeout(format!("请求超时: {}", e))
        } else if e.is_connect() {
            TrainError::Network(format!("连接错误: {}", e))
        } else if e.is_request() {
            TrainError::Network(format!("请求错误: {}", e))
        } else if e.is_status() {
            if let Some(status) = e.status() {
                TrainError::Network(format!("HTTP {}: {}", status, e))
            } else {
                TrainError::Network(format!("状态码错误: {}", e))
            }
        } else {
            TrainError::Network(format!("网络错误: {}", e))
        }
    }
}

impl From<prost::EncodeError> for TrainError {
    fn from(e: prost::EncodeError) -> Self {
        TrainError::Serialization(serde_json::Error::io(std::io::Error::other(
            format!("Protobuf编码错误: {}", e),
        )))
    }
}

// ============================================================================
// 错误严重程度枚举
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Debug => write!(f, "DEBUG"),
            ErrorSeverity::Info => write!(f, "INFO"),
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ============================================================================
// TrainError 附加方法实现
// ============================================================================

impl TrainError {
    /// 获取错误类型名称
    pub fn error_type(&self) -> &str {
        match self {
            TrainError::Io(_) => "IO",
            TrainError::Config(_) => "Config",
            TrainError::Data(_) => "Data",
            TrainError::Model(_) => "Model",
            TrainError::Training(_) => "Training",
            TrainError::Tokenizer(_) => "Tokenizer",
            TrainError::Database(_) => "Database",
            TrainError::Serialization(_) => "Serialization",
            TrainError::Network(_) => "Network",
            TrainError::Download(_) => "Download",
            TrainError::Preprocessing(_) => "Preprocessing",
            TrainError::Validation(_) => "Validation",
            TrainError::Device(_) => "Device",
            TrainError::Parse(_) => "Parse",
            TrainError::Checkpoint(_) => "Checkpoint",
            TrainError::Distributed(_) => "Distributed",
            TrainError::OutOfMemory(_) => "OutOfMemory",
            TrainError::GradientOverflow(_) => "GradientOverflow",
            TrainError::Gpu(_) => "GPU",
            TrainError::Timeout(_) => "Timeout",
            TrainError::Unknown(_) => "Unknown",
        }
    }

    /// 获取错误严重程度
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            TrainError::Io(_) => ErrorSeverity::Error,
            TrainError::Config(_) => ErrorSeverity::Error,
            TrainError::Data(_) => ErrorSeverity::Error,
            TrainError::Model(_) => ErrorSeverity::Error,
            TrainError::Training(_) => ErrorSeverity::Error,
            TrainError::Tokenizer(_) => ErrorSeverity::Error,
            TrainError::Database(_) => ErrorSeverity::Error,
            TrainError::Serialization(_) => ErrorSeverity::Error,
            TrainError::Network(_) => ErrorSeverity::Warning,
            TrainError::Download(_) => ErrorSeverity::Warning,
            TrainError::Preprocessing(_) => ErrorSeverity::Warning,
            TrainError::Validation(_) => ErrorSeverity::Error,
            TrainError::Device(_) => ErrorSeverity::Error,
            TrainError::Parse(_) => ErrorSeverity::Error,
            TrainError::Checkpoint(_) => ErrorSeverity::Warning,
            TrainError::Distributed(_) => ErrorSeverity::Error,
            TrainError::OutOfMemory(_) => ErrorSeverity::Critical,
            TrainError::GradientOverflow(_) => ErrorSeverity::Warning,
            TrainError::Gpu(_) => ErrorSeverity::Critical,
            TrainError::Timeout(_) => ErrorSeverity::Warning,
            TrainError::Unknown(_) => ErrorSeverity::Error,
        }
    }

    /// 判断错误是否可恢复
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            TrainError::Network(_)
                | TrainError::Download(_)
                | TrainError::GradientOverflow(_)
                | TrainError::Timeout(_)
                | TrainError::Checkpoint(_)
                | TrainError::Preprocessing(_)
        )
    }

    /// 判断是否应该重试
    pub fn should_retry(&self) -> bool {
        matches!(
            self,
            TrainError::Network(_) | TrainError::Download(_) | TrainError::Timeout(_)
        )
    }

    /// 获取修复建议
    pub fn suggestion(&self) -> Option<String> {
        match self {
            TrainError::Config(msg) => {
                if msg.contains("文件不存在") {
                    Some("运行 'shlteLLM generate' 生成默认配置文件".to_string())
                } else if msg.contains("解析") {
                    Some("检查配置文件语法，确保符合TOML格式".to_string())
                } else {
                    Some("检查配置参数是否合法".to_string())
                }
            }
            TrainError::Network(_) => Some("检查网络连接，或尝试使用镜像源".to_string()),
            TrainError::Download(msg) => {
                if msg.contains("404") || msg.contains("无法下载") {
                    Some("检查数据集名称是否正确，或尝试手动下载".to_string())
                } else {
                    Some("检查网络连接和下载URL".to_string())
                }
            }
            TrainError::OutOfMemory(_) => {
                Some("尝试减小batch_size或micro_batch_size，或使用梯度累积".to_string())
            }
            TrainError::GradientOverflow(_) => {
                Some("尝试减小学习率，增加梯度裁剪阈值，或使用混合精度训练".to_string())
            }
            TrainError::Device(_) => Some("检查设备可用性，或切换到CPU训练".to_string()),
            TrainError::Validation(msg) => {
                if msg.contains("不能被") {
                    Some("调整模型参数确保维度匹配".to_string())
                } else {
                    Some("检查配置文件中的参数约束".to_string())
                }
            }
            TrainError::Tokenizer(msg) => {
                if msg.contains("内存") {
                    Some("使用流式训练模式 train_from_files() 替代 train_on_texts()".to_string())
                } else {
                    Some("检查分词器配置文件".to_string())
                }
            }
            TrainError::Database(msg) => {
                if msg.to_string().contains("locked") {
                    Some("等待其他进程释放数据库锁，或增加 busy_timeout".to_string())
                } else {
                    Some("检查数据库文件权限和磁盘空间".to_string())
                }
            }
            _ => None,
        }
    }
    
    /// 记录错误到日志
    pub fn log(&self, context: &str) {
        let severity = self.severity();
        let emoji = match severity {
            ErrorSeverity::Debug => "🔍",
            ErrorSeverity::Info => "ℹ️",
            ErrorSeverity::Warning => "⚠️",
            ErrorSeverity::Error => "❌",
            ErrorSeverity::Critical => "💀",
        };
        
        eprintln!("{} {}: {}", emoji, severity, context);
        eprintln!("   原因: {}", self);
        
        if let Some(suggestion) = self.suggestion() {
            eprintln!("   💡 建议: {}", suggestion);
        }
        
        // 在 debug 模式下打印更多信息
        if log::log_enabled!(log::Level::Debug) {
            eprintln!("   类型: {}", self.error_type());
            eprintln!("   可恢复: {}", self.is_recoverable());
        }
    }
}

// ============================================================================
// ResultExt Trait - 结果处理扩展
// ============================================================================

pub trait ResultExt<T> {
    /// 记录错误并返回Option，使用自定义上下文
    fn log_error(self, context: &str) -> Option<T>;
    
    /// 记录警告并返回Option
    fn log_warning(self, context: &str) -> Option<T>;
    
    /// 记录错误信息（不消费Result）
    fn log_error_ref(&self, context: &str);
    
    /// 出错时返回默认值
    fn or_default(self) -> T
    where
        T: Default;
    
    /// 出错时使用回调函数处理
    fn on_error<F>(self, f: F) -> Option<T>
    where
        F: FnOnce(&TrainError);
    
    /// 链式错误处理：连续尝试多个操作
    fn or_else<F>(self, fallback: F) -> Self
    where
        F: FnOnce() -> Self;
    
    /// 添加上下文
    fn context(self, context: &str) -> Self;
    
    /// 带回调的上下文
    fn with_context<F>(self, f: F) -> Self
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T> {
    fn log_error(self, context: &str) -> Option<T> {
        match self {
            Ok(val) => Some(val),
            Err(e) => {
                e.log(context);
                None
            }
        }
    }

    fn log_warning(self, context: &str) -> Option<T> {
        match self {
            Ok(val) => Some(val),
            Err(e) => {
                eprintln!("⚠️ {}: {}", context, e);
                None
            }
        }
    }
    
    fn log_error_ref(&self, context: &str) {
        if let Err(e) = self {
            e.log(context);
        }
    }

    fn or_default(self) -> T
    where
        T: Default,
    {
        self.unwrap_or_default()
    }

    fn on_error<F>(self, f: F) -> Option<T>
    where
        F: FnOnce(&TrainError),
    {
        match self {
            Ok(val) => Some(val),
            Err(e) => {
                f(&e);
                None
            }
        }
    }

    fn or_else<F>(self, fallback: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Ok(val) => Ok(val),
            Err(e) => {
                if e.is_recoverable() {
                    fallback()
                } else {
                    Err(e)
                }
            }
        }
    }
    
    fn context(self, context: &str) -> Self {
        self.map_err(|e| {
            if let TrainError::Unknown(msg) = &e {
                TrainError::Unknown(format!("{}: {}", context, msg))
            } else {
                TrainError::Unknown(format!("{}: {}", context, e))
            }
        })
    }
    
    fn with_context<F>(self, f: F) -> Self
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| TrainError::Unknown(format!("{}: {}", f(), e)))
    }
}

// ============================================================================
// OptionExt Trait - Option处理扩展
// ============================================================================

pub trait OptionExt<T> {
    /// 将Option转换为Result，提供错误信息
    fn ok_or_train_error(self, context: &str) -> Result<T>;
    
    /// 将Option转换为Result，使用懒加载的错误信息
    fn ok_or_else_train_error<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
    
    /// 获取值或使用默认值并记录警告
    fn or_log_warning(self, context: &str) -> T
    where
        T: Default;
    
    /// 获取值或执行回调
    fn or_else_log<F>(self, context: &str, f: F) -> T
    where
        F: FnOnce() -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_train_error(self, context: &str) -> Result<T> {
        self.ok_or_else(|| TrainError::Unknown(context.to_string()))
    }
    
    fn ok_or_else_train_error<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.ok_or_else(|| TrainError::Unknown(f()))
    }
    
    fn or_log_warning(self, context: &str) -> T
    where
        T: Default,
    {
        match self {
            Some(val) => val,
            None => {
                eprintln!("⚠️ {}: 使用默认值", context);
                T::default()
            }
        }
    }
    
    fn or_else_log<F>(self, context: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        match self {
            Some(val) => val,
            None => {
                eprintln!("⚠️ {}: 使用备选值", context);
                f()
            }
        }
    }
}

// ============================================================================
// 错误上下文包装器（已废弃 — 使用 ResultExt::context/with_context 替代）
// ============================================================================

/// 为Result添加上下文信息
#[deprecated(
    since = "3.2.2",
    note = "请直接使用 ResultExt::context 或 ResultExt::with_context"
)]
pub trait Context<T> {
    fn context(self, context: &str) -> Result<T>;
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

#[allow(deprecated)]
impl<T> Context<T> for Result<T> {
    fn context(self, context: &str) -> Result<T> {
        self.map_err(|e| {
            if let TrainError::Unknown(msg) = &e {
                TrainError::Unknown(format!("{}: {}", context, msg))
            } else {
                TrainError::Unknown(format!("{}: {}", context, e))
            }
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| TrainError::Unknown(format!("{}: {}", f(), e)))
    }
}

// ============================================================================
// 错误宏（简化错误处理）
// ============================================================================

/// 快速创建配置错误
#[macro_export]
macro_rules! config_error {
    ($msg:expr) => {
        $crate::error::TrainError::Config($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Config(format!($fmt, $($arg)*))
    };
}

/// 快速创建数据错误
#[macro_export]
macro_rules! data_error {
    ($msg:expr) => {
        $crate::error::TrainError::Data($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Data(format!($fmt, $($arg)*))
    };
}

/// 快速创建模型错误
#[macro_export]
macro_rules! model_error {
    ($msg:expr) => {
        $crate::error::TrainError::Model($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Model(format!($fmt, $($arg)*))
    };
}

/// 快速创建训练错误
#[macro_export]
macro_rules! training_error {
    ($msg:expr) => {
        $crate::error::TrainError::Training($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Training(format!($fmt, $($arg)*))
    };
}

/// 快速创建网络错误
#[macro_export]
macro_rules! network_error {
    ($msg:expr) => {
        $crate::error::TrainError::Network($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Network(format!($fmt, $($arg)*))
    };
}

/// 快速创建验证错误
#[macro_export]
macro_rules! validation_error {
    ($msg:expr) => {
        $crate::error::TrainError::Validation($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::TrainError::Validation(format!($fmt, $($arg)*))
    };
}

// ============================================================================
// 错误日志记录器
// ============================================================================

/// 错误日志记录器（支持结构化日志）
pub struct ErrorLogger {
    pub log_errors: bool,
    pub log_stack_trace: bool,
    pub output_format: ErrorOutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorOutputFormat {
    HumanReadable,
    Json,
    Compact,
}

impl Default for ErrorLogger {
    fn default() -> Self {
        ErrorLogger {
            log_errors: true,
            log_stack_trace: cfg!(debug_assertions),
            output_format: ErrorOutputFormat::HumanReadable,
        }
    }
}

impl ErrorLogger {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_json_format() -> Self {
        ErrorLogger {
            output_format: ErrorOutputFormat::Json,
            ..Default::default()
        }
    }
    
    pub fn with_compact_format() -> Self {
        ErrorLogger {
            output_format: ErrorOutputFormat::Compact,
            ..Default::default()
        }
    }
    
    pub fn log(&self, error: &TrainError, context: Option<&str>) {
        if !self.log_errors {
            return;
        }
        
        match self.output_format {
            ErrorOutputFormat::HumanReadable => {
                self.log_human_readable(error, context);
            }
            ErrorOutputFormat::Json => {
                self.log_json(error, context);
            }
            ErrorOutputFormat::Compact => {
                self.log_compact(error, context);
            }
        }
    }
    
    fn log_human_readable(&self, error: &TrainError, context: Option<&str>) {
        let severity = error.severity();
        let emoji = match severity {
            ErrorSeverity::Debug => "🔍",
            ErrorSeverity::Info => "ℹ️",
            ErrorSeverity::Warning => "⚠️",
            ErrorSeverity::Error => "❌",
            ErrorSeverity::Critical => "💀",
        };
        
        if let Some(ctx) = context {
            eprintln!("{} [{}] {}: {}", emoji, severity, ctx, error);
        } else {
            eprintln!("{} [{}] {}", emoji, severity, error);
        }
        
        if let Some(suggestion) = error.suggestion() {
            eprintln!("   💡 建议: {}", suggestion);
        }
        
        if self.log_stack_trace {
            eprintln!("   📍 错误类型: {}", error.error_type());
        }
    }
    
    fn log_json(&self, error: &TrainError, context: Option<&str>) {
        let entry = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "severity": error.severity().to_string(),
            "error_type": error.error_type(),
            "message": error.to_string(),
            "context": context,
            "recoverable": error.is_recoverable(),
            "suggestion": error.suggestion(),
        });
        
        eprintln!("{}", serde_json::to_string(&entry).unwrap_or_default());
    }
    
    fn log_compact(&self, error: &TrainError, context: Option<&str>) {
        let severity_char = match error.severity() {
            ErrorSeverity::Debug => 'D',
            ErrorSeverity::Info => 'I',
            ErrorSeverity::Warning => 'W',
            ErrorSeverity::Error => 'E',
            ErrorSeverity::Critical => 'C',
        };
        
        if let Some(ctx) = context {
            eprintln!("{}|{}|{}", severity_char, ctx, error);
        } else {
            eprintln!("{}||{}", severity_char, error);
        }
    }
}

// ============================================================================
// 全局错误处理器
// ============================================================================

use std::sync::OnceLock;

static ERROR_LOGGER: OnceLock<ErrorLogger> = OnceLock::new();

/// 设置全局错误日志器
pub fn set_error_logger(logger: ErrorLogger) {
    let _ = ERROR_LOGGER.set(logger);
}

/// 获取全局错误日志器
pub fn get_error_logger() -> &'static ErrorLogger {
    ERROR_LOGGER.get_or_init(ErrorLogger::default)
}

/// 记录全局错误
pub fn log_error(error: &TrainError, context: Option<&str>) {
    get_error_logger().log(error, context);
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_types() {
        let err = TrainError::Config("测试错误".to_string());
        assert_eq!(err.error_type(), "Config");
        assert_eq!(err.severity(), ErrorSeverity::Error);
        assert!(!err.is_recoverable());
    }
    
    #[test]
    fn test_network_error_recoverable() {
        let err = TrainError::Network("连接失败".to_string());
        assert!(err.is_recoverable());
        assert!(err.should_retry());
    }
    
    #[test]
    fn test_oom_unrecoverable() {
        let err = TrainError::OutOfMemory("GPU内存不足".to_string());
        assert_eq!(err.severity(), ErrorSeverity::Critical);
        assert!(!err.is_recoverable());
        assert!(err.suggestion().is_some());
    }
    
    #[test]
    fn test_result_ext() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.log_error("test"), Some(42));
        
        let result: Result<i32> = Err(TrainError::Unknown("test".to_string()));
        assert_eq!(result.log_warning("test"), None);
    }
    
    #[test]
    fn test_option_ext() {
        let opt: Option<i32> = Some(42);
        assert_eq!(opt.ok_or_train_error("test").unwrap(), 42);
        
        let opt: Option<i32> = None;
        assert!(opt.ok_or_train_error("test").is_err());
        assert_eq!(opt.or_log_warning("test"), 0);
    }
    
    #[test]
    fn test_context() {
        let result: Result<i32> = Err(TrainError::Unknown("原始错误".to_string()));
        let with_context = Context::context(result, "额外上下文");
        assert!(with_context.is_err());
    }
    
    #[test]
    fn test_macros() {
        let err = config_error!("配置错误");
        assert!(matches!(err, TrainError::Config(_)));
        
        let err = data_error!("数据错误");
        assert!(matches!(err, TrainError::Data(_)));
        
        let err = model_error!("模型错误");
        assert!(matches!(err, TrainError::Model(_)));
        
        let err = training_error!("训练错误");
        assert!(matches!(err, TrainError::Training(_)));
        
        let err = network_error!("网络错误");
        assert!(matches!(err, TrainError::Network(_)));
        
        let err = validation_error!("验证错误");
        assert!(matches!(err, TrainError::Validation(_)));
    }
    
    #[test]
    fn test_error_logger() {
        let logger = ErrorLogger::new();
        let err = TrainError::Config("测试配置错误".to_string());
        
        // 不应 panic
        logger.log(&err, Some("测试上下文"));
    }
    
    #[test]
    fn test_json_error_logger() {
        let logger = ErrorLogger::with_json_format();
        let err = TrainError::Network("连接超时".to_string());
        
        logger.log(&err, None);
        // 不应 panic
    }
}