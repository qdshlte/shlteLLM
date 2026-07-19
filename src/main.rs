#![allow(dead_code)]
//! ============================================================================
//! 主入口模块
//! ============================================================================
//!
//! 本模块是程序的入口点，提供命令行接口：
//! - train: 训练模型
//! - chat: 运行模形
//! - download: 下载数据集
//! - preprocess: 预处理数据
//! - train-tokenizer: 训练分词器
//! - validate: 验证配置
//! - generate: 生成配置文件
//! - generate-presets: 生成预设配置
//! - benchmark: 基准测试
//! - inspect: 检查数据文件
//! - clean: 清理缓存
//!
//! ============================================================================

// ============================================================================
// 模块声明
// ============================================================================

#[path = "training/config.rs"]
mod config;
#[path = "training/data_one.rs"]
mod data_one;
#[path = "training/data_two.rs"]
mod data_two;
#[path = "training/db.rs"]
mod db;
#[path = "training/error.rs"]
mod error;
#[path = "training/model.rs"]
mod model;
#[path = "training/pipeline.rs"]
mod pipeline;
#[path = "training/tokenizer.rs"]
mod tokenizer;
#[path = "training/train.rs"]
mod train;
#[path = "llm/llm_bridge.rs"]
mod llm_bridge;
#[path = "llm/chat_dashboard.rs"]
mod chat_dashboard;

// ============================================================================
// 标准库导入
// ============================================================================

use clap::{Parser, Subcommand, ValueEnum};
use config::Config;
use error::Result;
use std::path::PathBuf;

// ============================================================================
// 命令行参数结构
// ============================================================================

#[derive(Parser)]
#[command(name = "shlteLLM")]
#[command(about = "LLM工具")]
#[command(version = "3.1.15")]
#[command(author = "QD·shlte")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// 详细输出模式
    #[arg(short, long, global = true)]
    verbose: bool,
}

// ============================================================================
// 预设类型枚举
// ============================================================================

#[derive(ValueEnum, Clone)]
enum Preset {
    Tiny,
    Small,
    Base,
}

// ============================================================================
// 子命令枚举
// ============================================================================

#[derive(Subcommand)]
enum Commands {
    /// 训练模型
    Train {
        /// 配置文件路径
        #[arg(short, long, default_value = "config.toml")]
        config: PathBuf,

        /// 输出目录
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// 恢复检查点路径
        #[arg(long)]
        resume: Option<PathBuf>,

        /// 使用预设配置
        #[arg(long, value_enum)]
        preset: Option<Preset>,
    },

    /// 下载数据集
    Download {
        /// 数据集名称
        #[arg(short, long)]
        dataset: String,

        /// 下载目录
        #[arg(short, long, default_value = "data")]
        output: PathBuf,
    },

    /// 预处理数据
    Preprocess {
        /// 输入数据路径
        #[arg(short, long)]
        input: PathBuf,

        /// 输出路径
        #[arg(short, long)]
        output: PathBuf,

        /// 分词器配置文件路径
        #[arg(short, long, default_value = "tokenizer.toml")]
        tokenizer_config: PathBuf,
    },

    /// 训练分词器
    TrainTokenizer {
        /// 输入文本文件或目录
        #[arg(short, long)]
        input: PathBuf,

        /// 输出路径
        #[arg(short, long, default_value = "tokenizer.json")]
        output: PathBuf,

        /// 分词器算法 (bpe, wordpiece, unigram, sentencepiece)
        #[arg(long, default_value = "bpe")]
        algorithm: String,

        /// 词表大小
        #[arg(long, default_value = "32000")]
        vocab_size: usize,
    },

    /// 验证配置文件
    Validate {
        /// 配置文件路径
        #[arg(short, long, default_value = "config.toml")]
        config: PathBuf,
    },

    /// 生成配置文件
    Generate {
        /// 输出路径
        #[arg(short, long, default_value = "config.toml")]
        output: PathBuf,

        /// 预设名称 (tiny, small, base)
        #[arg(short, long, default_value = "tiny")]
        preset: String,
    },

    /// 生成所有预设配置文件
    GeneratePresets {
        /// 输出目录
        #[arg(short, long, default_value = "presets")]
        output: PathBuf,
    },

    /// 运行基准测试
    Benchmark {
        /// 配置文件路径
        #[arg(short, long, default_value = "config.toml")]
        config: PathBuf,

        /// 输出目录
        #[arg(short, long, default_value = "benchmark_output")]
        output: PathBuf,
    },

    /// 检查数据文件
    Inspect {
        /// 数据文件路径
        #[arg(short, long)]
        path: PathBuf,
    },

    /// 清理缓存
    Clean {
        /// 缓存目录
        #[arg(short, long, default_value = "cache")]
        cache_dir: PathBuf,
    },
    
    ///运行
    Chat {
        /// 直接使用提示词（非交互模式）
        #[arg(short, long)]
        prompt: Option<String>,
    
        /// 模型文件路径（默认搜索当前目录）
        #[arg(short, long)]
        model: Option<PathBuf>,
    
        /// 最大生成 token 数
        #[arg(long, default_value = "512")]
        max_tokens: usize,
    
        /// 温度参数 (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    
        /// Top-p 采样参数
        #[arg(long, default_value = "0.9")]
        top_p: f32,
    
        /// 重复惩罚系数
        #[arg(long, default_value = "1.1")]
        repeat_penalty: f32,
    
        /// 上下文长度
        #[arg(long, default_value = "2048")]
        context_size: usize,
    
        /// 线程数
        #[arg(long, default_value = "4")]
        threads: usize,
    },
    
}

// ============================================================================
// 终端保护守卫
// ============================================================================

/// 确保终端在 panic 或异常退出时恢复到正常模式
struct TerminalGuard;

impl TerminalGuard {
    fn new() -> Self {
        // 设置自定义 panic hook，确保终端恢复
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // 尝试恢复终端
            let _ = crossterm::terminal::disable_raw_mode();
            let mut stdout = std::io::stdout();
            let _ = crossterm::execute!(
                stdout,
                crossterm::terminal::LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture
            );
            // 调用之前的 hook
            prev_hook(info);
        }));
        TerminalGuard
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        // 尝试恢复终端（正常退出时）
        let _ = crossterm::terminal::disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = crossterm::execute!(
            stdout,
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        );
    }
}

// ============================================================================
// 主函数
// ============================================================================

fn main() {
    // 初始化 panic hook 和终端守卫，确保终端不会停留在原始模式
    let _guard = TerminalGuard::new();

    let cli = Cli::parse();

    // 设置日志级别
    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 执行子命令
    if let Err(e) = run_command(cli) {
        eprintln!("\n❌ {}", e);
        if let Some(suggestion) = e.suggestion() {
            eprintln!("   💡 建议: {}", suggestion);
        }
        std::process::exit(1);
    }
}

fn run_command(cli: Cli) -> Result<()> {
    match cli.command {
        // ====================================================================
        // 训练命令
        // ====================================================================
        Commands::Train {
            config,
            output,
            resume,
            preset,
        } => {
            let config = if let Some(preset_name) = preset {
                match preset_name {
                    Preset::Tiny => Config::tiny_preset(),
                    Preset::Small => Config::small_preset(),
                    Preset::Base => Config::base_preset(),
                }
            } else {
                if !config.exists() {
                    println!("📝 配置文件不存在，正在生成默认配置...");
                    Config::generate_default(&config)?;
                }
                Config::from_file(&config)?
            };

            println!("🔍 验证配置...");
            config.validate()?;
            println!("✅ 配置验证通过");

            pipeline::run_training(config, output, resume)?;
        }

        // ====================================================================
        // 下载命令
        // ====================================================================
        Commands::Download { dataset, output } => {
            pipeline::download_dataset(&dataset, &output)?;
        }

        // ====================================================================
        // 预处理命令
        // ====================================================================
        Commands::Preprocess {
            input,
            output,
            tokenizer_config,
        } => {
            let tokenizer = tokenizer::Tokenizer::from_file(&tokenizer_config)?;
            pipeline::preprocess_data(&input, &output, &tokenizer)?;
        }

        // ====================================================================
        // 训练分词器命令
        // ====================================================================
        Commands::TrainTokenizer {
            input,
            output,
            algorithm,
            vocab_size,
        } => {
            println!("🔤 训练分词器...");
            println!("   输入: {}", input.display());
            println!("   算法: {}", algorithm);
            println!("   词表大小: {}", vocab_size);

            let mut config = Config::tiny_preset();
            config.tokenizer.vocab_size = vocab_size;
            config.tokenizer.algorithm = match algorithm.to_lowercase().as_str() {
                "bpe" => config::TokenizationAlgorithm::BPE,
                "wordpiece" => config::TokenizationAlgorithm::WordPiece,
                "unigram" => config::TokenizationAlgorithm::Unigram,
                "sentencepiece" => config::TokenizationAlgorithm::SentencePiece,
                _ => {
                    eprintln!("❌ 不支持的算法: {}", algorithm);
                    println!("   支持的算法: bpe, wordpiece, unigram, sentencepiece");
                    return Err(error::TrainError::Config(format!("不支持的算法: {}", algorithm)));
                }
            };

            let mut tokenizer = tokenizer::Tokenizer::from_file(&PathBuf::from("tokenizer.toml"))
                .unwrap_or_else(|_| {
                    let tokenizer_path = PathBuf::from("tokenizer.toml");
                    let tokenizer_config = config::TokenizerConfig {
                        algorithm: config.tokenizer.algorithm.clone(),
                        vocab_size: config.tokenizer.vocab_size,
                        special_tokens: config.tokenizer.special_tokens.clone(),
                        normalization: config.tokenizer.normalization,
                        add_prefix_space: config.tokenizer.add_prefix_space,
                    };

                    let content = toml::to_string_pretty(&tokenizer_config).unwrap();
                    let _ = std::fs::write(&tokenizer_path, content);

                    tokenizer::Tokenizer::from_file(&tokenizer_path).unwrap()
                });

            // 读取文本数据
            let texts = if input.is_dir() {
                let mut all_texts = Vec::new();
                for entry in std::fs::read_dir(&input)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_file() {
                        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

                        match extension {
                            "parquet" => {
                                println!(
                                    "   ⚠️  跳过Parquet文件: {} (请先使用preprocess命令预处理)",
                                    path.display()
                                );
                            }
                            _ => {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    all_texts.extend(content.lines().map(|l| l.to_string()));
                                }
                            }
                        }
                    }
                }
                all_texts
            } else {
                let content = std::fs::read_to_string(&input)?;
                content.lines().map(|l| l.to_string()).collect()
            };

            if texts.is_empty() {
                eprintln!("❌ 未找到可用的文本数据");
                return Err(error::TrainError::Data("未找到可用的文本数据".into()));
            }

            println!("   文本数量: {}", texts.len());
            tokenizer.train_on_texts(&texts)?;
            tokenizer.save(&output)?;

            println!("✅ 分词器已保存: {}", output.display());
            println!("   词表大小: {}", tokenizer.vocab_size());
        }

        // ====================================================================
        // 验证配置命令
        // ====================================================================
        Commands::Validate { config } => {
            let config = Config::from_file(&config)?;
            config.validate()?;
            println!("✅ 配置验证通过");

            println!("\n📋 配置概览:");
            println!(
                "   模型: {}层, {}维, {}头",
                config.model.num_layers, config.model.hidden_dim, config.model.num_heads
            );
            println!(
                "   训练: {}步, 批大小{}, 序列长度{}",
                config.training.num_steps,
                config.training.batch_size,
                config.training.sequence_length
            );
            println!("   学习率: {:.2e}", config.training.learning_rate);
            println!("   优化器: {:?}", config.training.optimizer);
            println!("   注意力: {:?}", config.model.attention);
            println!("   激活函数: {:?}", config.model.activation);
            println!("   位置编码: {:?}", config.model.position_encoding);

            let approx_params = config.model.num_layers as f64
                * config.model.hidden_dim as f64
                * config.model.hidden_dim as f64
                * 12.0;
            println!("   总参数量约: {:.1}M", approx_params / 1e6);
        }

        // ====================================================================
        // 生成配置文件命令
        // ====================================================================
        Commands::Generate { output, preset } => {
            let config = match preset.as_str() {
                "tiny" => Config::tiny_preset(),
                "small" => Config::small_preset(),
                "base" => Config::base_preset(),
                _ => {
                    eprintln!("⚠️  未知预设: {}，使用tiny", preset);
                    println!("   可用预设: tiny, small, base");
                    Config::tiny_preset()
                }
            };

            let content = toml::to_string_pretty(&config)
                .map_err(|e| error::TrainError::Config(format!("序列化失败: {}", e)))?;

            std::fs::write(&output, content)?;
            println!("✅ 配置文件已生成: {}", output.display());
        }

        // ====================================================================
        // 生成预设配置命令
        // ====================================================================
        Commands::GeneratePresets { output } => {
            Config::save_presets(&output)?;
            println!("✅ 所有预设配置已保存到: {}", output.display());
        }

        // ====================================================================
        // 基准测试命令
        // ====================================================================
        Commands::Benchmark { config, output } => {
            let config = Config::from_file(&config)?;
            config.validate()?;
            pipeline::benchmark(&config, &output)?;
        }

        // ====================================================================
        // 检查文件命令
        // ====================================================================
        Commands::Inspect { path } => {
            println!("🔍 检查文件: {}", path.display());

            if !path.exists() {
                println!("❌ 文件不存在");
                return Ok(());
            }

            let metadata = std::fs::metadata(&path)?;
            println!("   大小: {:.2} MB", metadata.len() as f64 / 1_048_576.0);

            if path.is_file() {
                let extension = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown");

                println!("   格式: {}", extension);

                if let Ok(content) = std::fs::read_to_string(&path) {
                    let lines: Vec<&str> = content.lines().take(5).collect();
                    println!("   前{}行预览:", lines.len());
                    for (i, line) in lines.iter().enumerate() {
                        let preview = if line.len() > 100 {
                            format!("{}...", &line[..100])
                        } else {
                            line.to_string()
                        };
                        println!("     {}: {}", i + 1, preview);
                    }

                    println!("   总行数: {}", content.lines().count());
                }
            } else if path.is_dir() {
                let entries: Vec<_> = std::fs::read_dir(&path)?.filter_map(|e| e.ok()).collect();

                println!("   文件数: {}", entries.len());

                for entry in entries.iter().take(10) {
                    let path = entry.path();
                    let metadata = entry.metadata().unwrap();
                    let size = metadata.len();

                    println!(
                        "     {} ({:.2} KB)",
                        path.file_name().unwrap().to_string_lossy(),
                        size as f64 / 1024.0
                    );
                }

                if entries.len() > 10 {
                    println!("     ... 还有{}个文件", entries.len() - 10);
                }
            }
        }

        // ====================================================================
        // 清理缓存命令
        // ====================================================================
        Commands::Clean { cache_dir } => {
            pipeline::clean_cache(&cache_dir)?;
        }
        
        // ====================================================================
        // 运行LLM命令
        // ====================================================================
        Commands::Chat {
            prompt,
            model,
            max_tokens,
            temperature,
            top_p,
            repeat_penalty,
            context_size,
            threads,
        } => {
            use chat_dashboard::Dashboard;
            use llm_bridge::LlamaContext;
            
            // 查找模型
            let model_path = if let Some(path) = model {
                path
            } else {
                // 自动搜索当前目录的 .gguf 文件
                let files: Vec<_> = std::fs::read_dir(".")
                    .ok()
                    .into_iter()
                    .flat_map(|d| d.filter_map(|e| e.ok()))
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
                    .collect();
                
                if files.is_empty() {
                    eprintln!("❌ 未找到 .gguf 模型文件");
                    eprintln!("   请使用 --model 指定模型路径，或将模型放在当前目录");
                    return Err(error::TrainError::Model("未找到 .gguf 模型文件".into()));
                }

                if files.len() > 1 {
                    println!("📂 找到多个模型文件，使用第一个:");
                    for f in &files {
                        println!("   - {}", f.display());
                    }
                }

                files[0].clone()
            };

            if !model_path.exists() {
                eprintln!("❌ 模型文件不存在: {}", model_path.display());
                return Err(error::TrainError::Model(format!(
                    "模型文件不存在: {}",
                    model_path.display()
                )));
            }

            // 非交互模式：直接执行提示词
            if let Some(prompt_text) = prompt {
                println!("🎯 加载模型: {}", model_path.display());

                let mut ctx = match LlamaContext::new(&model_path, context_size as u32, threads as i32) {
                    Ok(c) => c,
                    Err(e) => {
                        return Err(error::TrainError::Model(format!("加载模型失败: {}", e)));
                    }
                };

                let info = ctx.get_model_info();
                println!("📊 模型信息: {} 层, {} 头, 词表 {}", info.n_layer, info.n_head, info.n_vocab);
                println!("💬 提示词: {}", prompt_text);
                println!("📝 生成中...\n");

                let response = ctx.generate(
                    &prompt_text,
                    max_tokens,
                    temperature,
                    top_p,
                    repeat_penalty,
                ).map_err(|e| error::TrainError::Model(format!("生成失败: {}", e)))?;

                println!("{}", response);
                return Ok(());
            }
            
            // 交互模式：启动仪表板
            println!("🎨 启动聊天仪表板...");
            
            let mut dashboard = Dashboard::new();
            
            // 自动加载模型
            if let Err(e) = dashboard.load_model(&model_path) {
                eprintln!("❌ 加载模型失败: {}", e);
                eprintln!("   仍然可以进入仪表板，使用 /load 加载");
            }

            // 运行仪表板
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async {
                    dashboard.run().await
                })
                .map_err(|e| error::TrainError::Io(e))?;
        }
    }

    Ok(())
}