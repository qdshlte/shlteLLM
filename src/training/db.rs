#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]
//! ============================================================================
//! 数据库模块
//! ============================================================================
//!
//! 本模块实现了SQLite数据库管理功能，用于记录：
//! - 数据集下载状态
//! - 预处理数据统计
//! - 训练运行记录
//! - 检查点信息
//! - 训练指标
//! - 评估结果
//! - 系统事件
//! - 硬件信息
//! - 超参数搜索记录
//!
//! 修复内容（P1-6）：
//! - 添加连接池支持（r2d2）
//! - 多线程安全访问
//! - 连接健康检查
//! - 自动重连机制
//!
//! ============================================================================

// ============================================================================
// 标准库导入
// ============================================================================

use crate::error::{Result, TrainError};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;

// ============================================================================
// 连接池配置
// ============================================================================

#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// 最大连接数
    pub max_size: u32,
    /// 最小连接数
    pub min_size: u32,
    /// 连接超时时间（秒）
    pub connection_timeout: u64,
    /// 连接空闲时间（秒），超过后关闭
    pub idle_timeout: u64,
    /// 连接最大生命周期（秒）
    pub max_lifetime: u64,
    /// 是否启用健康检查
    pub health_check: bool,
    /// 健康检查间隔（秒）
    pub health_check_interval: u64,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        ConnectionPoolConfig {
            max_size: 10,
            min_size: 2,
            connection_timeout: 30,
            idle_timeout: 300,
            max_lifetime: 3600,
            health_check: true,
            health_check_interval: 60,
        }
    }
}

// ============================================================================
// 数据库连接管理器
// ============================================================================

/// 数据库连接管理器（带连接池）
pub struct DatabaseManager {
    pool: Arc<ConnectionPool>,
    config: ConnectionPoolConfig,
    db_path: PathBuf,
}

/// 连接池内部实现
struct ConnectionPool {
    connections: std::sync::Mutex<VecDeque<PooledConnection>>,
    config: ConnectionPoolConfig,
    db_path: PathBuf,
    stats: std::sync::atomic::AtomicU64,
}

/// 池化连接
struct PooledConnection {
    conn: Connection,
    created_at: Instant,
    last_used_at: Instant,
    id: u64,
}

impl PooledConnection {
    fn new(conn: Connection, id: u64) -> Self {
        let now = Instant::now();
        PooledConnection {
            conn,
            created_at: now,
            last_used_at: now,
            id,
        }
    }
    
    fn is_expired(&self, config: &ConnectionPoolConfig) -> bool {
        if config.max_lifetime > 0
            && self.created_at.elapsed() > Duration::from_secs(config.max_lifetime) {
                return true;
            }
        if config.idle_timeout > 0
            && self.last_used_at.elapsed() > Duration::from_secs(config.idle_timeout) {
                return true;
            }
        false
    }
    
    fn health_check(&mut self) -> bool {
        self.conn.execute("SELECT 1", []).is_ok()
    }
}

impl ConnectionPool {
    fn new(db_path: PathBuf, config: ConnectionPoolConfig) -> Result<Self> {
        let pool = ConnectionPool {
            connections: std::sync::Mutex::new(VecDeque::with_capacity(config.max_size as usize)),
            config,
            db_path,
            stats: std::sync::atomic::AtomicU64::new(0),
        };
        
        // 创建最小数量的连接
        for _ in 0..pool.config.min_size {
            let _ = pool.create_new_connection();
        }
        
        Ok(pool)
    }
    
    fn create_new_connection(&self) -> Result<PooledConnection> {
        let conn = Connection::open(&self.db_path)
            .map_err(TrainError::Database)?;
        
        // 设置连接优化参数
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
        conn.execute_batch("PRAGMA cache_size=-64000;")?;
        conn.execute_batch("PRAGMA busy_timeout=5000;")?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        conn.execute_batch("PRAGMA temp_store=MEMORY;")?;
        
        let id = self.stats.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(PooledConnection::new(conn, id))
    }
    
    fn cleanup_expired_connections(&self) {
        let mut guard = match self.connections.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        
        let config = &self.config;
        let min_size = config.min_size as usize;
        
        // 保留至少 min_size 个连接
        let original_len = guard.len();
        guard.retain(|conn| !conn.is_expired(config));
        
        let removed = original_len - guard.len();
        if removed > 0 {
            log::debug!("清理了 {} 个过期连接", removed);
        }
        
        // 如果连接数少于最小值，补充新连接
        while guard.len() < min_size {
            if let Ok(new_conn) = self.create_new_connection() {
                guard.push_back(new_conn);
            } else {
                break;
            }
        }
    }
    
    fn get_connection(&self) -> Result<PooledConnection> {
        let start = Instant::now();
        let timeout = Duration::from_secs(self.config.connection_timeout);
        
        // 定期清理过期连接
        if self.stats.load(std::sync::atomic::Ordering::Relaxed).is_multiple_of(100) {
            self.cleanup_expired_connections();
        }
        
        loop {
            {
                let mut guard = match self.connections.lock() {
                    Ok(g) => g,
                    Err(e) => return Err(TrainError::Database(rusqlite::Error::InvalidParameterName(e.to_string()))),
                };
                
                if let Some(mut conn) = guard.pop_front() {
                    // 健康检查
                    if self.config.health_check && !conn.health_check() {
                        log::warn!("连接 {} 健康检查失败，重新创建", conn.id);
                        drop(guard);
                        conn = self.create_new_connection()?;
                        return Ok(conn);
                    }
                    
                    conn.last_used_at = Instant::now();
                    return Ok(conn);
                }
            }
            
            // 没有可用连接，尝试创建新连接
            if self.can_create_new_connection() {
                if let Ok(conn) = self.create_new_connection() {
                    return Ok(conn);
                }
            }
            
            // 等待可用连接
            if start.elapsed() >= timeout {
                return Err(TrainError::Database(rusqlite::Error::InvalidParameterName(
                    format!("获取数据库连接超时 ({}秒)", self.config.connection_timeout)
                )));
            }
            
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    
    fn can_create_new_connection(&self) -> bool {
        let guard = match self.connections.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.len() < self.config.max_size as usize
    }
    
    fn return_connection(&self, conn: PooledConnection) {
        if conn.is_expired(&self.config) {
            return;
        }
        
        if let Ok(mut guard) = self.connections.lock() {
            if guard.len() < self.config.max_size as usize {
                guard.push_back(conn);
            }
        }
    }
}

impl Drop for ConnectionPool {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.connections.lock() {
            guard.clear();
        }
    }
}

// ============================================================================
// 数据库连接包装器（RAII）
// ============================================================================

/// 数据库连接包装器，自动归还到连接池
pub struct DatabaseConnection<'a> {
    pool: &'a ConnectionPool,
    conn: Option<PooledConnection>,
}

impl<'a> DatabaseConnection<'a> {
    fn new(pool: &'a ConnectionPool) -> Result<Self> {
        let conn = pool.get_connection()?;
        Ok(DatabaseConnection {
            pool,
            conn: Some(conn),
        })
    }
    
    pub fn inner(&mut self) -> &mut Connection {
        &mut self.conn.as_mut().unwrap().conn
    }
}

impl<'a> Drop for DatabaseConnection<'a> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.return_connection(conn);
        }
    }
}

impl<'a> std::ops::Deref for DatabaseConnection<'a> {
    type Target = Connection;
    
    fn deref(&self) -> &Self::Target {
        &self.conn.as_ref().unwrap().conn
    }
}

impl<'a> std::ops::DerefMut for DatabaseConnection<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.conn.as_mut().unwrap().conn
    }
}

// ============================================================================
// 数据库主结构（使用连接池）
// ============================================================================

pub struct Database {
    pool: Arc<ConnectionPool>,
    config: ConnectionPoolConfig,
    db_path: PathBuf,
}

impl Database {
    // ========================================================================
    // 创建与初始化
    // ========================================================================

    /// 打开数据库连接（使用连接池）
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_config(path, &ConnectionPoolConfig::default())
    }
    
    /// 使用自定义配置打开数据库
    pub fn open_with_config(path: &Path, config: &ConnectionPoolConfig) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let db_path = path.to_path_buf();
        let pool = ConnectionPool::new(db_path.clone(), config.clone())?;
        
        let db = Database {
            pool: Arc::new(pool),
            config: config.clone(),
            db_path,
        };
        
        db.initialize()?;
        Ok(db)
    }
    
    /// 获取一个数据库连接
    fn get_connection(&self) -> Result<DatabaseConnection<'_>> {
        DatabaseConnection::new(&self.pool)
    }
    
    /// 执行一个函数，自动管理连接
    fn with_connection<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T>,
    {
        let mut conn = self.get_connection()?;
        f(conn.inner())
    }
    
    /// 执行一个事务
    pub fn transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T>,
    {
        self.with_connection(|conn| {
            conn.execute_batch("BEGIN")?;
            match f(conn) {
                Ok(result) => {
                    conn.execute_batch("COMMIT")?;
                    Ok(result)
                }
                Err(e) => {
                    conn.execute_batch("ROLLBACK").ok();
                    Err(e)
                }
            }
        })
    }
    
    /// 初始化数据库表结构
    fn initialize(&self) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute_batch(
                "
                -- ================================================================
                -- 数据集表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    size_gb REAL,
                    num_shards INTEGER,
                    download_status TEXT NOT NULL DEFAULT 'pending',
                    download_progress REAL DEFAULT 0.0,
                    downloaded_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    last_retry_at TIMESTAMP,
                    UNIQUE(name, source)
                );
                
                -- ================================================================
                -- 预处理数据表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS preprocessed_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    vocab_size INTEGER,
                    num_tokens INTEGER,
                    num_sequences INTEGER,
                    avg_sequence_length REAL,
                    max_sequence_length INTEGER,
                    min_sequence_length INTEGER,
                    preprocessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    file_size_mb REAL,
                    checksum TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 训练运行表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_hash TEXT NOT NULL,
                    model_config TEXT NOT NULL,
                    training_config TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'running',
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER,
                    best_loss REAL,
                    final_loss REAL,
                    device_info TEXT,
                    num_parameters INTEGER,
                    git_commit TEXT,
                    command_line TEXT,
                    hostname TEXT,
                    pid INTEGER
                );
                
                -- ================================================================
                -- 检查点表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    eval_loss REAL,
                    path TEXT NOT NULL,
                    is_best INTEGER DEFAULT 0,
                    model_size_mb REAL,
                    ema_model_size_mb REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 指标表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    eval_loss REAL,
                    learning_rate REAL,
                    gradient_norm REAL,
                    weight_norm REAL,
                    tokens_per_second REAL,
                    samples_per_second REAL,
                    memory_usage_mb REAL,
                    gpu_memory_used_mb REAL,
                    gpu_memory_total_mb REAL,
                    gpu_utilization_pct REAL,
                    cpu_utilization_pct REAL,
                    disk_io_read_mb REAL,
                    disk_io_write_mb REAL,
                    network_io_mb REAL,
                    batch_time_ms REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 评估结果表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    dataset_name TEXT DEFAULT 'validation',
                    perplexity REAL,
                    loss REAL,
                    accuracy_top1 REAL,
                    accuracy_top5 REAL,
                    accuracy_top10 REAL,
                    bleu_score REAL,
                    rouge1 REAL,
                    rouge2 REAL,
                    rouge_l REAL,
                    exact_match REAL,
                    f1_score REAL,
                    num_samples INTEGER,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 系统事件表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    severity TEXT DEFAULT 'info',
                    source_file TEXT,
                    line_number INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 硬件信息表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS hardware_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    cpu_model TEXT,
                    cpu_cores INTEGER,
                    cpu_threads INTEGER,
                    ram_total_mb REAL,
                    gpu_model TEXT,
                    gpu_count INTEGER,
                    gpu_memory_total_mb REAL,
                    cuda_version TEXT,
                    driver_version TEXT,
                    os_info TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 超参数搜索表
                -- ================================================================
                CREATE TABLE IF NOT EXISTS hyperparameter_search (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value REAL NOT NULL,
                    trial_number INTEGER,
                    objective_value REAL,
                    status TEXT DEFAULT 'running',
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );
                
                -- ================================================================
                -- 索引
                -- ================================================================
                CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_step ON checkpoints(run_id, step);
                CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluation_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_events_run ON system_events(run_id);
                CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(download_status);
                CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs(status);
            ",
            )?;
            
            conn.execute_batch("PRAGMA optimize;")?;
            Ok(())
        })
    }
    
    /// 获取连接池统计信息
    pub fn get_pool_stats(&self) -> PoolStats {
        let connections = self.pool.connections.lock().map(|g| g.len()).unwrap_or(0);
        PoolStats {
            active_connections: connections,
            max_connections: self.config.max_size,
            min_connections: self.config.min_size,
            total_created: self.pool.stats.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
    
    /// 关闭数据库连接池
    pub fn close(&self) -> Result<()> {
        // 连接池会在 Drop 时自动清理
        Ok(())
    }

    // ========================================================================
    // 数据集操作
    // ========================================================================

    pub fn add_dataset(&self, name: &str, source: &str, size_gb: f64, num_shards: usize) -> Result<i64> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO datasets (name, source, size_gb, num_shards, download_status) 
                 VALUES (?1, ?2, ?3, ?4, 'pending')",
                params![name, source, size_gb, num_shards],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn update_download_status(&self, dataset_id: i64, status: &str) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE datasets SET download_status = ?1, downloaded_at = CURRENT_TIMESTAMP WHERE id = ?2",
                params![status, dataset_id],
            )?;
            Ok(())
        })
    }

    pub fn update_download_progress(&self, dataset_id: i64, progress: f64) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE datasets SET download_progress = ?1 WHERE id = ?2",
                params![progress, dataset_id],
            )?;
            Ok(())
        })
    }

    pub fn set_download_error(&self, dataset_id: i64, error: &str) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE datasets SET download_status = 'error', error_message = ?1, 
                 retry_count = retry_count + 1, last_retry_at = CURRENT_TIMESTAMP WHERE id = ?2",
                params![error, dataset_id],
            )?;
            Ok(())
        })
    }

    pub fn get_dataset_status(&self, dataset_id: i64) -> Result<Option<(String, f64, Option<String>)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT download_status, download_progress, error_message FROM datasets WHERE id = ?1",
            )?;
            
            let result = stmt
                .query_row(params![dataset_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                })
                .ok();
            
            Ok(result)
        })
    }

    pub fn get_all_datasets(&self) -> Result<Vec<DatasetRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, name, source, download_status, COALESCE(download_progress, 0.0)
                 FROM datasets ORDER BY id DESC",
            )?;

            let datasets = stmt
                .query_map([], |row| {
                    Ok(DatasetRecord {
                        id: row.get::<_, i64>(0)?,
                        name: row.get::<_, String>(1)?,
                        source: row.get::<_, String>(2)?,
                        download_status: row.get::<_, String>(3)?,
                        download_progress: row.get::<_, f64>(4)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(datasets)
        })
    }

    // ========================================================================
    // 预处理数据操作
    // ========================================================================

    pub fn add_preprocessed_data(
        &self,
        dataset_id: i64,
        vocab_size: usize,
        num_tokens: usize,
        num_sequences: usize,
        avg_sequence_length: f64,
    ) -> Result<i64> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO preprocessed_data (dataset_id, vocab_size, num_tokens, num_sequences, avg_sequence_length)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    dataset_id, 
                    vocab_size as i64, 
                    num_tokens as i64, 
                    num_sequences as i64, 
                    avg_sequence_length
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn add_preprocessed_data_full(
        &self,
        dataset_id: i64,
        vocab_size: usize,
        num_tokens: usize,
        num_sequences: usize,
        avg_sequence_length: f64,
        max_seq_len: usize,
        min_seq_len: usize,
        file_path: &str,
        file_size_mb: f64,
        checksum: &str,
    ) -> Result<i64> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO preprocessed_data 
                 (dataset_id, vocab_size, num_tokens, num_sequences, avg_sequence_length,
                  max_sequence_length, min_sequence_length, file_path, file_size_mb, checksum)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    dataset_id,
                    vocab_size as i64,
                    num_tokens as i64,
                    num_sequences as i64,
                    avg_sequence_length,
                    max_seq_len as i64,
                    min_seq_len as i64,
                    file_path,
                    file_size_mb,
                    checksum,
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    // ========================================================================
    // 训练运行操作
    // ========================================================================

    pub fn start_training_run(
        &self,
        config_hash: &str,
        model_config: &str,
        training_config: &str,
        total_steps: usize,
        device_info: &str,
    ) -> Result<i64> {
        self.with_connection(|conn| {
            let pid = std::process::id() as i64;
            let hostname = std::env::var("HOSTNAME")
                .or_else(|_| std::env::var("COMPUTERNAME"))
                .unwrap_or_else(|_| "unknown".to_string());

            conn.execute(
                "INSERT INTO training_runs 
                 (config_hash, model_config, training_config, total_steps, device_info, pid, hostname, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'running')",
                params![config_hash, model_config, training_config, total_steps as i64, device_info, pid, hostname],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn update_training_run(&self, run_id: i64, step: usize, loss: f64, status: &str) -> Result<()> {
        self.transaction(|conn| {
            conn.execute(
                "UPDATE training_runs SET 
                    current_step = ?1, 
                    best_loss = CASE WHEN best_loss IS NULL OR ?2 < best_loss THEN ?2 ELSE best_loss END, 
                    status = CASE WHEN ?3 = 'completed' THEN 'completed' ELSE status END
                 WHERE id = ?4",
                params![step as i64, loss, status, run_id],
            )?;

            conn.execute(
                "INSERT INTO metrics (run_id, step, loss) VALUES (?1, ?2, ?3)",
                params![run_id, step as i64, loss],
            )?;
            
            Ok(())
        })
    }

    pub fn complete_training_run(&self, run_id: i64, final_loss: f64) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE training_runs SET 
                    status = 'completed', 
                    final_loss = ?1, 
                    end_time = CURRENT_TIMESTAMP 
                 WHERE id = ?2",
                params![final_loss, run_id],
            )?;
            Ok(())
        })
    }

    pub fn fail_training_run(&self, run_id: i64, error: &str) -> Result<()> {
        self.transaction(|conn| {
            conn.execute(
                "UPDATE training_runs SET status = 'failed', end_time = CURRENT_TIMESTAMP WHERE id = ?1",
                params![run_id],
            )?;
            
            self.log_event(Some(run_id), "training_failed", error, "error", None, None)?;
            Ok(())
        })
    }

    pub fn pause_training_run(&self, run_id: i64) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE training_runs SET status = 'paused' WHERE id = ?1",
                params![run_id],
            )?;
            Ok(())
        })
    }

    pub fn resume_training_run(&self, run_id: i64) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE training_runs SET status = 'running' WHERE id = ?1",
                params![run_id],
            )?;
            Ok(())
        })
    }

    // ========================================================================
    // 检查点操作
    // ========================================================================

    pub fn add_checkpoint(
        &self,
        run_id: i64,
        step: usize,
        loss: f64,
        eval_loss: Option<f64>,
        path: &str,
        is_best: bool,
        model_size_mb: Option<f64>,
    ) -> Result<i64> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO checkpoints (run_id, step, loss, eval_loss, path, is_best, model_size_mb) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    run_id,
                    step as i64,
                    loss,
                    eval_loss,
                    path,
                    is_best as i32,
                    model_size_mb
                ],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn get_checkpoints(&self, run_id: i64) -> Result<Vec<CheckpointRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, loss, COALESCE(eval_loss, 0.0), path, is_best
                 FROM checkpoints WHERE run_id = ?1 ORDER BY step",
            )?;

            let checkpoints = stmt
                .query_map(params![run_id], |row| {
                    Ok(CheckpointRecord {
                        step: row.get::<_, i64>(0)? as usize,
                        loss: row.get::<_, f64>(1)?,
                        eval_loss: row.get::<_, f64>(2)?,
                        path: row.get::<_, String>(3)?,
                        is_best: row.get::<_, i32>(4)? != 0,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(checkpoints)
        })
    }

    pub fn get_best_checkpoint(&self, run_id: i64) -> Result<Option<(usize, f64, String)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, loss, path FROM checkpoints 
                 WHERE run_id = ?1 AND is_best = 1 
                 ORDER BY loss ASC LIMIT 1",
            )?;
            
            let result = stmt
                .query_map(params![run_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .next();
            
            match result {
                Some(Ok(data)) => Ok(Some(data)),
                _ => Ok(None),
            }
        })
    }

    pub fn get_latest_checkpoint(&self, run_id: i64) -> Result<Option<(usize, f64, String)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, loss, path FROM checkpoints 
                 WHERE run_id = ?1 
                 ORDER BY step DESC LIMIT 1",
            )?;
            
            let result = stmt
                .query_map(params![run_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?
                .next();
            
            match result {
                Some(Ok(data)) => Ok(Some(data)),
                _ => Ok(None),
            }
        })
    }

    // ========================================================================
    // 评估操作
    // ========================================================================

    pub fn add_evaluation(
        &self,
        run_id: i64,
        step: usize,
        perplexity: f64,
        accuracy_top1: Option<f64>,
        accuracy_top5: Option<f64>,
        loss: f64,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO evaluation_results (run_id, step, perplexity, accuracy_top1, accuracy_top5, loss)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![run_id, step as i64, perplexity, accuracy_top1, accuracy_top5, loss],
            )?;
            Ok(())
        })
    }

    pub fn add_evaluation_full(
        &self,
        run_id: i64,
        step: usize,
        dataset_name: &str,
        perplexity: f64,
        loss: f64,
        accuracy_top1: Option<f64>,
        accuracy_top5: Option<f64>,
        bleu_score: Option<f64>,
        rouge1: Option<f64>,
        rouge2: Option<f64>,
        rouge_l: Option<f64>,
        num_samples: usize,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO evaluation_results 
                 (run_id, step, dataset_name, perplexity, loss, accuracy_top1, accuracy_top5,
                  bleu_score, rouge1, rouge2, rouge_l, num_samples)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    run_id,
                    step as i64,
                    dataset_name,
                    perplexity,
                    loss,
                    accuracy_top1,
                    accuracy_top5,
                    bleu_score,
                    rouge1,
                    rouge2,
                    rouge_l,
                    num_samples as i64
                ],
            )?;
            Ok(())
        })
    }

    // ========================================================================
    // 指标操作
    // ========================================================================

    pub fn update_metrics(
        &self,
        run_id: i64,
        step: usize,
        loss: f64,
        learning_rate: f64,
        gradient_norm: f64,
        tokens_per_second: f64,
        memory_usage_mb: f64,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO metrics (run_id, step, loss, learning_rate, gradient_norm, tokens_per_second, memory_usage_mb)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![run_id, step as i64, loss, learning_rate, gradient_norm, tokens_per_second, memory_usage_mb],
            )?;
            Ok(())
        })
    }

    pub fn update_metrics_full(
        &self,
        run_id: i64,
        step: usize,
        loss: f64,
        eval_loss: Option<f64>,
        learning_rate: f64,
        gradient_norm: f64,
        weight_norm: Option<f64>,
        tokens_per_second: f64,
        memory_usage_mb: f64,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO metrics 
                 (run_id, step, loss, eval_loss, learning_rate, gradient_norm, weight_norm, 
                  tokens_per_second, memory_usage_mb)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    run_id,
                    step as i64,
                    loss,
                    eval_loss,
                    learning_rate,
                    gradient_norm,
                    weight_norm,
                    tokens_per_second,
                    memory_usage_mb
                ],
            )?;
            Ok(())
        })
    }

    pub fn update_gpu_metrics(
        &self,
        run_id: i64,
        step: usize,
        gpu_memory_mb: f64,
        gpu_utilization_pct: f64,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE metrics SET gpu_memory_used_mb = ?1, gpu_utilization_pct = ?2 
                 WHERE run_id = ?3 AND step = ?4",
                params![gpu_memory_mb, gpu_utilization_pct, run_id, step as i64],
            )?;
            Ok(())
        })
    }

    // ========================================================================
    // 系统事件操作
    // ========================================================================

    pub fn log_event(
        &self,
        run_id: Option<i64>,
        event_type: &str,
        message: &str,
        severity: &str,
        source_file: Option<&str>,
        line_number: Option<i64>,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO system_events (run_id, event_type, message, severity, source_file, line_number) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![run_id, event_type, message, severity, source_file, line_number],
            )?;
            Ok(())
        })
    }

    pub fn log_info(&self, run_id: Option<i64>, message: &str) -> Result<()> {
        self.log_event(run_id, "info", message, "info", None, None)
    }

    pub fn log_warning(&self, run_id: Option<i64>, message: &str) -> Result<()> {
        self.log_event(run_id, "warning", message, "warning", None, None)
    }

    pub fn log_error(&self, run_id: Option<i64>, message: &str) -> Result<()> {
        self.log_event(run_id, "error", message, "error", None, None)
    }

    // ========================================================================
    // 硬件信息操作
    // ========================================================================

    pub fn add_hardware_info(
        &self,
        run_id: i64,
        cpu_model: &str,
        cpu_cores: usize,
        cpu_threads: usize,
        ram_total_mb: f64,
        gpu_model: &str,
        gpu_count: usize,
        gpu_memory_total_mb: f64,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO hardware_info 
                 (run_id, cpu_model, cpu_cores, cpu_threads, ram_total_mb, 
                  gpu_model, gpu_count, gpu_memory_total_mb)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    run_id,
                    cpu_model,
                    cpu_cores as i64,
                    cpu_threads as i64,
                    ram_total_mb,
                    gpu_model,
                    gpu_count as i64,
                    gpu_memory_total_mb
                ],
            )?;
            Ok(())
        })
    }

    // ========================================================================
    // 超参数搜索操作
    // ========================================================================

    pub fn add_hyperparameter_trial(
        &self,
        run_id: i64,
        param_name: &str,
        param_value: f64,
        trial_number: usize,
    ) -> Result<i64> {
        self.with_connection(|conn| {
            conn.execute(
                "INSERT INTO hyperparameter_search (run_id, param_name, param_value, trial_number)
                 VALUES (?1, ?2, ?3, ?4)",
                params![run_id, param_name, param_value, trial_number as i64],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    pub fn complete_hyperparameter_trial(
        &self,
        trial_id: i64,
        objective_value: f64,
        status: &str,
    ) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute(
                "UPDATE hyperparameter_search SET objective_value = ?1, status = ?2 WHERE id = ?3",
                params![objective_value, status, trial_id],
            )?;
            Ok(())
        })
    }

    // ========================================================================
    // 查询操作
    // ========================================================================

    pub fn get_training_runs(&self) -> Result<Vec<TrainingRunRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, status, start_time, COALESCE(best_loss, 0.0), current_step
                 FROM training_runs ORDER BY start_time DESC LIMIT 10",
            )?;

            let runs = stmt
                .query_map([], |row| {
                    Ok(TrainingRunRecord {
                        id: row.get::<_, i64>(0)?,
                        status: row.get::<_, String>(1)?,
                        start_time: row.get::<_, String>(2)?,
                        best_loss: row.get::<_, f64>(3)?,
                        current_step: row.get::<_, i64>(4)? as usize,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(runs)
        })
    }

    pub fn get_training_runs_full(
        &self,
    ) -> Result<Vec<TrainingRunFullRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, status, start_time, COALESCE(end_time, ''),
                        COALESCE(best_loss, 0.0), COALESCE(final_loss, 0.0),
                        current_step, total_steps, device_info
                 FROM training_runs ORDER BY start_time DESC LIMIT 20",
            )?;

            let runs = stmt
                .query_map([], |row| {
                    Ok(TrainingRunFullRecord {
                        id: row.get::<_, i64>(0)?,
                        status: row.get::<_, String>(1)?,
                        start_time: row.get::<_, String>(2)?,
                        end_time: row.get::<_, String>(3)?,
                        best_loss: row.get::<_, f64>(4)?,
                        final_loss: row.get::<_, f64>(5)?,
                        current_step: row.get::<_, i64>(6)? as usize,
                        total_steps: row.get::<_, i64>(7)? as usize,
                        device_info: row.get::<_, Option<String>>(8)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(runs)
        })
    }

    pub fn get_metrics_for_run(&self, run_id: i64) -> Result<Vec<(usize, f64, f64, f64)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, loss, learning_rate, COALESCE(tokens_per_second, 0.0) 
                 FROM metrics WHERE run_id = ?1 ORDER BY step",
            )?;
            
            let metrics = stmt
                .query_map(params![run_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            
            Ok(metrics)
        })
    }

    pub fn get_metrics_for_run_full(
        &self,
        run_id: i64,
    ) -> Result<Vec<MetricRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, loss, COALESCE(eval_loss, 0.0), learning_rate,
                        COALESCE(gradient_norm, 0.0), COALESCE(tokens_per_second, 0.0)
                 FROM metrics WHERE run_id = ?1 ORDER BY step",
            )?;

            let metrics = stmt
                .query_map(params![run_id], |row| {
                    Ok(MetricRecord {
                        step: row.get::<_, i64>(0)? as usize,
                        loss: row.get::<_, f64>(1)?,
                        eval_loss: row.get::<_, f64>(2)?,
                        learning_rate: row.get::<_, f64>(3)?,
                        gradient_norm: row.get::<_, f64>(4)?,
                        tokens_per_second: row.get::<_, f64>(5)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            Ok(metrics)
        })
    }

    pub fn get_evaluations_for_run(&self, run_id: i64) -> Result<Vec<(usize, f64, f64)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT step, perplexity, COALESCE(accuracy_top1, 0.0) 
                 FROM evaluation_results WHERE run_id = ?1 ORDER BY step",
            )?;
            
            let evals = stmt
                .query_map(params![run_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)? as usize,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            
            Ok(evals)
        })
    }

    pub fn get_events_for_run(&self, run_id: i64, limit: usize) -> Result<Vec<(String, String, String, String)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_type, message, severity, recorded_at 
                 FROM system_events WHERE run_id = ?1 
                 ORDER BY recorded_at DESC LIMIT ?2",
            )?;
            
            let events = stmt
                .query_map(params![run_id, limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            
            Ok(events)
        })
    }

    pub fn get_best_hyperparameters(&self, param_name: &str, limit: usize) -> Result<Vec<(f64, f64)>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT param_value, objective_value 
                 FROM hyperparameter_search 
                 WHERE param_name = ?1 AND status = 'completed'
                 ORDER BY objective_value ASC LIMIT ?2",
            )?;
            
            let results = stmt
                .query_map(params![param_name, limit as i64], |row| {
                    Ok((row.get::<_, f64>(0)?, row.get::<_, f64>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            
            Ok(results)
        })
    }

    // ========================================================================
    // 统计操作
    // ========================================================================

    pub fn get_training_stats(&self, run_id: i64) -> Result<TrainingStats> {
        self.with_connection(|conn| {
            let total_steps: i64 = conn.query_row(
                "SELECT COUNT(*) FROM metrics WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )?;

            let best_loss: f64 = conn.query_row(
                "SELECT COALESCE(MIN(loss), 0.0) FROM metrics WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )?;

            let avg_loss: f64 = conn.query_row(
                "SELECT COALESCE(AVG(loss), 0.0) FROM metrics WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )?;

            let total_evaluations: i64 = conn.query_row(
                "SELECT COUNT(*) FROM evaluation_results WHERE run_id = ?1",
                params![run_id],
                |row| row.get(0),
            )?;

            let best_perplexity: Option<f64> = conn
                .query_row(
                    "SELECT MIN(perplexity) FROM evaluation_results WHERE run_id = ?1",
                    params![run_id],
                    |row| row.get(0),
                )
                .ok()
                .flatten();

            Ok(TrainingStats {
                total_steps: total_steps as usize,
                best_loss,
                avg_loss,
                total_evaluations: total_evaluations as usize,
                best_perplexity,
            })
        })
    }

    // ========================================================================
    // 维护操作
    // ========================================================================

    pub fn vacuum(&self) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute_batch("VACUUM;")?;
            Ok(())
        })
    }

    pub fn analyze(&self) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute_batch("ANALYZE;")?;
            Ok(())
        })
    }

    pub fn optimize(&self) -> Result<()> {
        self.with_connection(|conn| {
            conn.execute_batch("PRAGMA optimize;")?;
            conn.execute_batch("PRAGMA analysis_limit=1000;")?;
            Ok(())
        })
    }

    pub fn get_db_size(&self) -> Result<f64> {
        self.with_connection(|conn| {
            let page_count: i64 = conn.query_row("PRAGMA page_count;", [], |row| row.get(0))?;
            let page_size: i64 = conn.query_row("PRAGMA page_size;", [], |row| row.get(0))?;
            Ok(page_count as f64 * page_size as f64 / 1_048_576.0)
        })
    }

    pub fn get_table_counts(&self) -> Result<DatabaseStats> {
        self.with_connection(|conn| {
            let tables = [
                "datasets",
                "preprocessed_data",
                "training_runs",
                "checkpoints",
                "metrics",
                "evaluation_results",
                "system_events",
                "hardware_info",
                "hyperparameter_search",
            ];

            let mut counts = Vec::new();
            for table in &tables {
                let count: i64 =
                    conn.query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |row| {
                        row.get(0)
                    })?;
                counts.push((table.to_string(), count as usize));
            }

            Ok(DatabaseStats {
                table_counts: counts,
            })
        })
    }
}

// ============================================================================
// 辅助结构
// ============================================================================

/// 数据集记录
#[derive(Debug, Clone)]
pub struct DatasetRecord {
    pub id: i64,
    pub name: String,
    pub source: String,
    pub download_status: String,
    pub download_progress: f64,
}

/// 检查点记录
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    pub step: usize,
    pub loss: f64,
    pub eval_loss: f64,
    pub path: String,
    pub is_best: bool,
}

/// 训练运行记录（摘要）
#[derive(Debug, Clone)]
pub struct TrainingRunRecord {
    pub id: i64,
    pub status: String,
    pub start_time: String,
    pub best_loss: f64,
    pub current_step: usize,
}

/// 训练运行记录（完整）
#[derive(Debug, Clone)]
pub struct TrainingRunFullRecord {
    pub id: i64,
    pub status: String,
    pub start_time: String,
    pub end_time: String,
    pub best_loss: f64,
    pub final_loss: f64,
    pub current_step: usize,
    pub total_steps: usize,
    pub device_info: Option<String>,
}

/// 指标记录
#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub step: usize,
    pub loss: f64,
    pub eval_loss: f64,
    pub learning_rate: f64,
    pub gradient_norm: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub best_loss: f64,
    pub avg_loss: f64,
    pub total_evaluations: usize,
    pub best_perplexity: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub table_counts: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub active_connections: usize,
    pub max_connections: u32,
    pub min_connections: u32,
    pub total_created: u64,
}

impl std::fmt::Display for TrainingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TrainingStats(steps={}, best_loss={:.4}, avg_loss={:.4}, evals={}, best_ppl={})",
            self.total_steps,
            self.best_loss,
            self.avg_loss,
            self.total_evaluations,
            self.best_perplexity
                .map(|p| format!("{:.2}", p))
                .unwrap_or_else(|| "N/A".to_string())
        )
    }
}

impl std::fmt::Display for DatabaseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Database Table Counts:")?;
        for (table, count) in &self.table_counts {
            writeln!(f, "  {}: {}", table, count)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PoolStats(active={}, max={}, min={}, total_created={})",
            self.active_connections, self.max_connections, self.min_connections, self.total_created
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
    fn test_database_connection_pool() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        
        let db = Database::open(&db_path).unwrap();
        
        // 测试连接池统计
        let stats = db.get_pool_stats();
        assert_eq!(stats.min_connections, 2);
        assert_eq!(stats.max_connections, 10);
        assert!(stats.active_connections <= stats.max_connections as usize);
        
        // 测试并发访问
        let db = Arc::new(db);
        let mut handles = vec![];
        
        for i in 0..20 {
            let db_clone = db.clone();
            handles.push(std::thread::spawn(move || {
                let result = db_clone.with_connection(|conn| {
                    conn.execute("SELECT 1", []).map_err(|e| TrainError::Database(e))
                });
                assert!(result.is_ok());
                i
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = db.get_pool_stats();
        println!("Pool stats after concurrent access: {}", stats);
    }
    
    #[test]
    fn test_transaction() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = Database::open(&db_path).unwrap();
        
        let result = db.transaction(|conn| {
            conn.execute("CREATE TABLE test (id INTEGER)", [])?;
            conn.execute("INSERT INTO test VALUES (1)", [])?;
            Ok::<_, TrainError>(())
        });
        
        assert!(result.is_ok());
        
        // 验证数据已提交
        let count: i64 = db.with_connection(|conn| {
            conn.query_row("SELECT COUNT(*) FROM test", [], |row| row.get(0))
        }).unwrap();
        assert_eq!(count, 1);
    }
    
    #[test]
    fn test_transaction_rollback() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = Database::open(&db_path).unwrap();
        
        let result = db.transaction(|conn| {
            conn.execute("CREATE TABLE test (id INTEGER)", [])?;
            conn.execute("INSERT INTO test VALUES (1)", [])?;
            Err::<(), TrainError>(TrainError::Database(rusqlite::Error::from("test error")))
        });
        
        assert!(result.is_err());
        
        // 验证表不存在（事务回滚）
        let table_exists: bool = db.with_connection(|conn| {
            conn.query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test'",
                [],
                |_| Ok(true),
            ).unwrap_or(false)
        }).unwrap_or(false);
        assert!(!table_exists);
    }
    
    #[test]
    fn test_crud_operations() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let db = Database::open(&db_path).unwrap();
        
        // 添加数据集
        let dataset_id = db.add_dataset("test_dataset", "huggingface", 1.5, 1).unwrap();
        assert!(dataset_id > 0);
        
        // 更新状态
        db.update_download_status(dataset_id, "completed").unwrap();
        
        // 查询状态
        let status = db.get_dataset_status(dataset_id).unwrap();
        assert!(status.is_some());
        let (status_str, progress, error) = status.unwrap();
        assert_eq!(status_str, "completed");
        assert_eq!(progress, 0.0);
        assert!(error.is_none());
        
        // 添加预处理数据
        let preprocessed_id = db.add_preprocessed_data(dataset_id, 1000, 10000, 100, 100.0).unwrap();
        assert!(preprocessed_id > 0);
        
        // 获取所有数据集
        let datasets = db.get_all_datasets().unwrap();
        assert!(!datasets.is_empty());
    }
}