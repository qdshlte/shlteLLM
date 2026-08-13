//! LLM 聊天仪表板 - 交互式 TUI
//!
//! 提供完整的聊天界面，支持：
//! - 模型加载和管理
//! - 流式对话生成
//! - 命令系统
//! - 对话历史管理
//! - 滚动浏览

use crate::llm_bridge::{LlamaContext, ModelInfo};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Scrollbar, ScrollbarState},
    Frame, Terminal,
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use walkdir::WalkDir;

/// 仪表板状态
pub struct Dashboard {
    /// 当前加载的模型上下文
    pub ctx: Option<LlamaContext>,
    /// 模型文件路径
    pub model_path: Option<PathBuf>,
    /// 模型信息
    pub model_info: Option<ModelInfo>,
    /// 对话历史
    pub messages: Vec<ChatMessage>,
    /// 当前输入
    pub input: String,
    /// 光标位置
    pub cursor_pos: usize,
    /// 输入模式
    pub input_mode: InputMode,
    /// 是否正在生成
    pub is_generating: bool,
    /// 当前生成的流式文本
    pub streaming_text: String,
    /// 滚动位置
    pub scroll_offset: usize,
    /// 生成任务句柄
    pub generation_task: Option<std::thread::JoinHandle<()>>,
    /// 消息接收器（用于接收流式生成结果）
    pub message_receiver: Option<mpsc::UnboundedReceiver<GenerationMessage>>,
    /// 消息发送器
    pub message_sender: Option<mpsc::UnboundedSender<GenerationMessage>>,
    /// 最后生成的结果
    pub last_generation_result: Option<String>,
    /// 生成参数
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    /// 保存的历史对话文件路径（用于自动恢复）
    pub history_file: Option<PathBuf>,
}

/// 生成过程中的消息
#[derive(Debug, Clone)]
pub enum GenerationMessage {
    /// 文本块
    Chunk(String),
    /// 完成
    Complete(String),
    /// 错误
    Error(String),
}

/// 输入模式
#[derive(Debug, Clone, PartialEq)]
pub enum InputMode {
    Normal,
    Insert,
}

/// 聊天消息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: String,
}

impl ChatMessage {
    /// 创建新消息
    pub fn new(role: MessageRole, content: &str) -> Self {
        ChatMessage {
            role,
            content: content.to_string(),
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
        }
    }
}

/// 消息角色
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl Dashboard {
    /// 创建新的仪表板
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        Dashboard {
            ctx: None,
            model_path: None,
            model_info: None,
            messages: vec![ChatMessage::new(
                MessageRole::System,
                "欢迎使用 SHLTE LLM 聊天助手\n\n\
                 输入 /help 查看命令\n\n\
                 加载模型: /load <路径>\n\n\
                 支持的格式: .gguf",
            )],
            input: String::new(),
            cursor_pos: 0,
            input_mode: InputMode::Normal,
            is_generating: false,
            streaming_text: String::new(),
            scroll_offset: 0,
            generation_task: None,
            message_receiver: Some(receiver),
            message_sender: Some(sender),
            last_generation_result: None,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            history_file: None,
        }
    }

    /// 加载模型
    pub fn load_model(&mut self, path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("文件不存在: {}", path.display()));
        }

        let path_str = path.to_string_lossy();
        if !path_str.ends_with(".gguf") {
            return Err("不是 GGUF 模型文件".to_string());
        }

        // 释放旧模型
        self.ctx = None;

        // 加载新模型
        let ctx = LlamaContext::new(path, 2048, 4)
            .map_err(|e| format!("加载模型失败: {}", e))?;

        let info = ctx.get_model_info();

        self.ctx = Some(ctx);
        self.model_path = Some(path.to_path_buf());
        self.model_info = Some(info);
        
        self.add_system_message(&format!(
            "✅ 模型已加载\n   - 词表大小: {}\n   - 上下文长度: {}\n   - 嵌入维度: {}\n   - 层数: {}\n   - 注意力头数: {}",
            self.model_info.as_ref().unwrap().n_vocab,
            self.model_info.as_ref().unwrap().n_ctx,
            self.model_info.as_ref().unwrap().n_embd,
            self.model_info.as_ref().unwrap().n_layer,
            self.model_info.as_ref().unwrap().n_head,
        ));
        
        Ok(())
    }
    
    /// 搜索当前目录及子目录中的 .gguf 文件
    pub fn search_gguf_files(&self, dir: &PathBuf) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for entry in WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                files.push(path.to_path_buf());
            }
        }
        files
    }
    
    /// 列出可用模型
    pub fn list_models(&self, dir: &PathBuf) -> String {
        let files = self.search_gguf_files(dir);
        if files.is_empty() {
            "没有找到 .gguf 模型文件".to_string()
        } else {
            let mut result = format!("找到 {} 个模型文件:\n", files.len());
            for (i, f) in files.iter().enumerate() {
                let size = fs::metadata(f)
                    .map(|m| m.len() as f64 / 1_073_741_824.0)
                    .unwrap_or(0.0);
                result.push_str(&format!("  {}. {} ({:.1} GB)\n", i + 1, f.file_name().unwrap().to_string_lossy(), size));
            }
            result
        }
    }
    
    /// 添加用户消息
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage::new(MessageRole::User, content));
        // 重置滚动到底部
        self.scroll_offset = self.messages.len().saturating_sub(1).saturating_sub(20);
    }

    /// 添加助手消息
    pub fn add_assistant_message(&mut self, content: &str) {
        self.messages.push(ChatMessage::new(MessageRole::Assistant, content));
        // 重置滚动到底部
        self.scroll_offset = self.messages.len().saturating_sub(1).saturating_sub(20);
    }

    /// 添加系统消息
    pub fn add_system_message(&mut self, content: &str) {
        self.messages.push(ChatMessage::new(MessageRole::System, content));
        // 重置滚动到底部
        self.scroll_offset = self.messages.len().saturating_sub(1).saturating_sub(20);
    }
    
    /// 更新最后一条消息的内容（用于流式生成）
    pub fn update_last_assistant_message(&mut self, content: &str) {
        if let Some(last_msg) = self.messages.last_mut() {
            if last_msg.role == MessageRole::Assistant {
                last_msg.content = content.to_string();
            } else {
                self.messages.push(ChatMessage::new(MessageRole::Assistant, content));
            }
        } else {
            self.messages.push(ChatMessage::new(MessageRole::Assistant, content));
        }
        
        // 重置滚动到底部
        self.scroll_offset = self.messages.len().saturating_sub(1).saturating_sub(20);
    }

    /// 处理用户输入（命令或聊天）
    pub fn process_input(&mut self) -> Result<(), String> {
        let input = self.input.trim().to_string();
        if input.is_empty() {
            return Ok(());
        }

        // 处理命令
        if input.starts_with('/') {
            self.input.clear();
            self.cursor_pos = 0;
            return self.process_command(&input);
        }

        // 普通聊天模式
        if self.ctx.is_none() {
            self.add_system_message("请先使用 /load <路径> 加载模型");
            return Ok(());
        }

        let user_input = input;
        self.add_user_message(&user_input);

        // 清空输入
        self.input.clear();
        self.cursor_pos = 0;

        // 获取模型路径
        let model_path = self.model_path.clone().ok_or("未加载模型")?;

        // 启动生成
        self.start_generation(user_input, model_path)?;

        Ok(())
    }
    
    /// 启动生成任务
    fn start_generation(&mut self, prompt: String, model_path: PathBuf) -> Result<(), String> {
        if self.is_generating {
            return Err("已有正在进行的生成任务".to_string());
        }

        // 创建消息通道
        let (sender, receiver) = mpsc::unbounded_channel::<GenerationMessage>();

        // 准备生成参数（使用 Dashboard 配置）
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;
        let top_p = self.top_p;
        let repeat_penalty = self.repeat_penalty;

        // 构建完整提示词
        let full_prompt = self.build_chat_prompt(&prompt);

        // 标记正在生成
        self.is_generating = true;
        self.streaming_text.clear();

        // 创建一个临时助手消息占位
        self.messages.push(ChatMessage::new(MessageRole::Assistant, ""));

        // 保存消息通道用于后续接收
        let sender_for_thread = sender.clone();
        let old_sender = self.message_sender.replace(sender);
        drop(old_sender);
        self.message_receiver = Some(receiver);

        // 使用 std::thread::spawn 启动生成，并保存句柄
        let task = std::thread::spawn(move || {
            let mut ctx = match LlamaContext::new(&model_path, 2048, 4) {
                Ok(c) => c,
                Err(e) => {
                    let _ = sender_for_thread.send(GenerationMessage::Error(format!("无法加载模型: {}", e)));
                    return;
                }
            };

            match ctx.generate_stream(
                &full_prompt,
                max_tokens,
                temperature,
                top_p,
                repeat_penalty,
                |chunk| {
                    let _ = sender_for_thread.send(GenerationMessage::Chunk(chunk.to_string()));
                },
            ) {
                Ok(text) => {
                    let _ = sender_for_thread.send(GenerationMessage::Complete(text));
                }
                Err(e) => {
                    let _ = sender_for_thread.send(GenerationMessage::Error(e));
                }
            }
        });

        self.generation_task = Some(task);
        Ok(())
    }

    /// 设置生成参数
    pub fn set_generation_params(&mut self, max_tokens: usize, temperature: f32, top_p: f32, repeat_penalty: f32) {
        self.max_tokens = max_tokens;
        self.temperature = temperature;
        self.top_p = top_p;
        self.repeat_penalty = repeat_penalty;
    }

    /// 保存当前对话历史到文件
    pub fn save_history(&mut self, path: &Path) -> Result<(), String> {
        let messages_json = serde_json::to_string_pretty(&self.messages)
            .map_err(|e| format!("序列化失败: {}", e))?;
        fs::write(path, messages_json)
            .map_err(|e| format!("写入失败: {}", e))?;
        self.history_file = Some(path.to_path_buf());
        Ok(())
    }

    /// 从文件加载对话历史
    pub fn load_history(&mut self, path: &Path) -> Result<(), String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("读取失败: {}", e))?;
        let messages: Vec<ChatMessage> = serde_json::from_str(&content)
            .map_err(|e| format!("解析失败: {}", e))?;
        if messages.is_empty() {
            return Err("文件内容为空".to_string());
        }
        self.messages = messages;
        self.history_file = Some(path.to_path_buf());
        Ok(())
    }
    
    /// 构建聊天提示词
    fn build_chat_prompt(&self, user_input: &str) -> String {
        let mut prompt = String::new();
        
        // 添加系统消息
        for msg in &self.messages {
            if msg.role == MessageRole::System {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
        }
        
        // 添加最近的对话历史（最多保留10轮）
        let recent_start = if self.messages.len() > 21 { self.messages.len() - 21 } else { 0 };
        for msg in &self.messages[recent_start..] {
            match msg.role {
                MessageRole::User => {
                    prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
                }
                MessageRole::Assistant => {
                    if !msg.content.is_empty() {
                        prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
                    }
                }
                MessageRole::System => {
                    // 系统消息已处理
                }
            }
        }
        
        // 添加当前用户输入
        prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", user_input));
        
        prompt
    }
    
    /// 处理命令（完整实现）
    fn process_command(&mut self, input: &str) -> Result<(), String> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0];
        
        match cmd {
            "/load" => {
                if parts.len() < 2 {
                    self.add_system_message("用法: /load <模型路径>");
                    return Ok(());
                }
                let path = PathBuf::from(parts[1]);
                self.load_model(&path)?;
            }
            "/ls" => {
                let dir = if parts.len() >= 2 {
                    PathBuf::from(parts[1])
                } else {
                    PathBuf::from(".")
                };
                let list = self.list_models(&dir);
                self.add_system_message(&list);
            }
            "/clear" => {
                self.messages.retain(|m| m.role == MessageRole::System);
                self.input.clear();
                self.cursor_pos = 0;
                self.streaming_text.clear();
                self.is_generating = false;
                self.scroll_offset = 0;
            }
            "/save" => {
                let path = if parts.len() > 1 {
                    PathBuf::from(&parts[1])
                } else {
                    PathBuf::from("chat_history.json")
                };
                match self.save_history(&path) {
                    Ok(()) => self.add_system_message(&format!("✅ 历史对话已保存到: {}", path.display())),
                    Err(e) => self.add_system_message(&format!("❌ 保存失败: {}", e)),
                }
            }
            "/loadhist" => {
                if parts.len() < 2 {
                    self.add_system_message("用法: /loadhist <路径>");
                } else {
                    match self.load_history(&PathBuf::from(&parts[1])) {
                        Ok(()) => self.add_system_message("✅ 历史对话已加载"),
                        Err(e) => self.add_system_message(&format!("❌ 加载失败: {}", e)),
                    }
                }
            }
            "/info" => {
                if let Some(info) = &self.model_info {
                    self.add_system_message(&format!(
                        "📊 模型信息\n   - 词表大小: {}\n   - 上下文长度: {}\n   - 嵌入维度: {}\n   - 层数: {}\n   - 注意力头数: {}\n   - 模型路径: {}",
                        info.n_vocab, info.n_ctx, info.n_embd, info.n_layer, info.n_head,
                        self.model_path.as_ref().map(|p| p.display().to_string()).unwrap_or("未加载".to_string())
                    ));
                } else {
                    self.add_system_message("未加载模型，使用 /load 加载");
                }
            }
            "/stop" => {
                if self.is_generating {
                    drop(self.generation_task.take());
                    self.is_generating = false;
                    self.streaming_text.clear();
                    self.add_system_message("生成已停止");
                } else {
                    self.add_system_message("没有正在进行的生成任务");
                }
            }
            "/params" => {
                if parts.len() < 5 {
                    self.add_system_message(&format!(
                        "用法: /params <max_tokens> <temperature> <top_p> <repeat_penalty>\n\
                         当前参数: max_tokens={}, temperature={}, top_p={}, repeat_penalty={}",
                        self.max_tokens, self.temperature, self.top_p, self.repeat_penalty
                    ));
                    return Ok(());
                }
                let max_tokens: usize = parts[1].parse().unwrap_or(self.max_tokens);
                let temperature: f32 = parts[2].parse().unwrap_or(self.temperature);
                let top_p: f32 = parts[3].parse().unwrap_or(self.top_p);
                let repeat_penalty: f32 = parts[4].parse().unwrap_or(self.repeat_penalty);
                self.set_generation_params(max_tokens, temperature, top_p, repeat_penalty);
                self.add_system_message(&format!(
                    "参数已更新: max_tokens={}, temperature={}, top_p={}, repeat_penalty={}",
                    self.max_tokens, self.temperature, self.top_p, self.repeat_penalty
                ));
            }
            "/help" => {
                self.add_system_message(
                    "可用命令:\n\
                     /load <路径>  - 加载 GGUF 模型\n\
                     /ls [目录]    - 列出 .gguf 文件\n\
                     /info         - 显示模型信息\n\
                     /clear        - 清空对话历史\n\
                     /stop         - 停止当前生成\n\
                     /params <n> <t> <p> <r> - 设置生成参数\n\
                     /quit         - 退出\n\
                     /help         - 显示帮助\n\n\
                     直接输入文本即可与模型对话\n\n\
                     快捷键:\n\
                     i - 进入插入模式\n\
                     Esc - 退出插入模式\n\
                     j/k - 上下滚动\n\
                     q - 退出"
                );
            }
            "/quit" | "/exit" => {
                std::process::exit(0);
            }
            _ => {
                self.add_system_message(&format!("未知命令: {}\n输入 /help 查看可用命令", cmd));
            }
        }
        
        Ok(())
    }
    
    /// 运行仪表板
    pub async fn run(&mut self) -> io::Result<()> {
        // 初始化终端
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        
        // 主循环
        let tick_rate = Duration::from_millis(50);
        let mut last_tick = Instant::now();
        
        let result = self.run_loop(&mut terminal, tick_rate, &mut last_tick).await;
        
        // 恢复终端
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        
        result
    }
    
    async fn run_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        tick_rate: Duration,
        last_tick: &mut Instant,
    ) -> io::Result<()> {
        loop {
            terminal.draw(|f| self.draw(f))?;
            
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match self.input_mode {
                            InputMode::Normal => self.handle_normal_mode(key.code),
                            InputMode::Insert => self.handle_insert_mode(key.code),
                        }
                    }
                }
            }
            
            if last_tick.elapsed() >= tick_rate {
                *last_tick = Instant::now();
                
                // 检查生成任务状态
                if self.is_generating {
                    if let Some(task) = &self.generation_task {
                        if task.is_finished() {
                            self.is_generating = false;
                            self.streaming_text.clear();
                        }
                    }
                }
            }
            
            // 处理异步消息（非阻塞）
            if let Some(receiver) = &mut self.message_receiver {
                while let Ok(msg) = receiver.try_recv() {
                    match msg {
                        GenerationMessage::Chunk(chunk) => {
                            self.streaming_text.push_str(&chunk);
                            // 更新最后一条助手消息
                            if let Some(msg) = self.messages.last_mut() {
                                if msg.role == MessageRole::Assistant {
                                    msg.content.push_str(&chunk);
                                }
                            }
                        }
                        GenerationMessage::Complete(full_text) => {
                            self.is_generating = false;
                            self.streaming_text.clear();
                            self.last_generation_result = Some(full_text.clone());
                            // 滚动到底部
                            self.scroll_offset = self.messages.len().saturating_sub(1).saturating_sub(20);
                        }
                        GenerationMessage::Error(err) => {
                            self.is_generating = false;
                            self.streaming_text.clear();
                            if let Some(msg) = self.messages.last_mut() {
                                if msg.role == MessageRole::Assistant {
                                    msg.content = format!("生成失败: {}", err);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    fn handle_normal_mode(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('i') => self.input_mode = InputMode::Insert,
            KeyCode::Char('q') => std::process::exit(0),
            KeyCode::Char('j') => {
                // 向下滚动，带边界检查
                let max_scroll = if self.messages.len() > 20 {
                    self.messages.len().saturating_sub(20)
                } else {
                    0
                };
                self.scroll_offset = (self.scroll_offset + 1).min(max_scroll);
            }
            KeyCode::Char('k') => {
                // 向上滚动，带边界检查
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            KeyCode::Char('g') => {
                // 滚动到顶部
                self.scroll_offset = 0;
            }
            KeyCode::Char('G') => {
                // 滚动到底部
                let max_scroll = if self.messages.len() > 20 {
                    self.messages.len().saturating_sub(20)
                } else {
                    0
                };
                self.scroll_offset = max_scroll;
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Insert;
            }
            KeyCode::Char('c') => {
                drop(self.generation_task.take());
                self.is_generating = false;
                self.streaming_text.clear();
                self.add_system_message("生成已停止");
            }
            _ => {}
        }
    }
    
    fn handle_insert_mode(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.input_mode = InputMode::Normal;
            }
            KeyCode::Enter => {
                self.input_mode = InputMode::Normal;
                let _ = self.process_input();
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 && !self.input.is_empty() {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.cursor_pos < self.input.len() {
                    self.cursor_pos += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_pos = 0;
            }
            KeyCode::End => {
                self.cursor_pos = self.input.len();
            }
            _ => {}
        }
    }
    
    fn draw(&mut self, f: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(f.area());
        
        // 聊天区域
        self.draw_chat_area(f, chunks[0]);
        
        // 输入区域
        self.draw_input_area(f, chunks[1]);
        
        // 状态栏
        self.draw_status_bar(f, chunks[2]);
    }
    
    fn draw_chat_area(&mut self, f: &mut Frame<'_>, area: Rect) {
        let visible_height = area.height as usize;
        
        // 计算可见消息范围
        let start_idx = self.scroll_offset;
        let end_idx = (start_idx + visible_height).min(self.messages.len());
        
        let visible_messages: Vec<ListItem> = self.messages[start_idx..end_idx]
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let color = match msg.role {
                    MessageRole::User => Color::Cyan,
                    MessageRole::Assistant => Color::Green,
                    MessageRole::System => Color::Yellow,
                };
                
                let prefix = match msg.role {
                    MessageRole::User => "You",
                    MessageRole::Assistant => "AI",
                    MessageRole::System => "System",
                };
                
                let is_last_streaming = self.is_generating 
                    && i == end_idx - start_idx - 1 
                    && msg.role == MessageRole::Assistant;
                
                let content = if is_last_streaming && !self.streaming_text.is_empty() {
                    format!("{}: {}{}", prefix, msg.content, self.streaming_text)
                } else {
                    format!("{}: {}", prefix, msg.content)
                };
                
                // 添加时间戳
                let styled = Line::from(vec![
                    Span::styled(format!("[{}] ", msg.timestamp), Style::default().fg(Color::DarkGray)),
                    Span::styled(content, Style::default().fg(color)),
                ]);
                
                ListItem::new(styled)
            })
            .collect();
        
        let messages_widget = List::new(visible_messages)
            .block(Block::default().borders(Borders::ALL).title("Chat"))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));
        
        f.render_widget(messages_widget, area);
        
        // 滚动条指示器
        if self.messages.len() > visible_height {
            let scroll_percent = self.scroll_offset as f32 / (self.messages.len() - visible_height).max(1) as f32;
            let scrollbar_height = ((visible_height as f32 * 0.3).max(2.0)) as usize;
            let _scrollbar_pos = (scroll_percent * (visible_height - scrollbar_height) as f32) as u16;

            let scrollbar = Scrollbar::default()
                .orientation(ratatui::widgets::ScrollbarOrientation::VerticalRight);

            let mut scroll_state = ScrollbarState::new(self.messages.len())
                .position(self.scroll_offset);

            f.render_stateful_widget(scrollbar, area, &mut scroll_state);
        }
    }
    
    fn draw_input_area(&mut self, f: &mut Frame<'_>, area: Rect) {
        let input_style = if self.input_mode == InputMode::Insert {
            Style::default().fg(Color::White).bg(Color::DarkGray)
        } else {
            Style::default().fg(Color::Gray)
        };
        
        let mode_indicator = if self.input_mode == InputMode::Normal {
            "NORMAL"
        } else {
            "INSERT"
        };
        
        let display_text = if self.is_generating {
            format!("[{}] ⏳ Generating...", mode_indicator)
        } else {
            format!("[{}] {}", mode_indicator, self.input)
        };
        
        let input_widget = Paragraph::new(display_text)
            .block(Block::default().borders(Borders::ALL).title("Input"))
            .style(input_style);
        
        f.render_widget(input_widget, area);
        
        // 光标位置（只在插入模式且未生成时显示）
        if self.input_mode == InputMode::Insert && !self.is_generating {
            let cursor_x = area.x + 2 + 2 + self.cursor_pos as u16; // +2 for mode indicator
            let cursor_y = area.y + 1;
            f.set_cursor_position(ratatui::layout::Position::new(cursor_x, cursor_y));
        }
    }
    
    fn draw_status_bar(&mut self, f: &mut Frame<'_>, area: Rect) {
        // 模型状态
        let model_info = if let Some(info) = &self.model_info {
            format!(
                "🤖 {}L·{}H·V{}", info.n_layer, info.n_head, info.n_vocab
            )
        } else {
            "❌ 未加载".to_string()
        };

        // 上下文利用率估算（基于消息 token 数）
        let ctx_tokens: usize = self.messages.iter().map(|m| m.content.chars().count() / 2).sum();
        let ctx_used = if self.ctx.as_ref().map_or(false, |c| c.context_size() > 0) {
            let size = self.ctx.as_ref().map(|c| c.context_size()).unwrap_or(2048);
            format!("{:.0}%", (ctx_tokens as f64 / size as f64) * 100.0)
        } else {
            "?".to_string()
        };

        // 历史文件
        let hist = if let Some(p) = &self.history_file {
            format!("💾{}", p.file_name().unwrap_or_default().to_string_lossy())
        } else {
            "".to_string()
        };

        let status = format!(
            " {} | {} | Ctx: {}% | {} | Mode: {} | {} | Msgs: {} | {}",
            model_info,
            if self.is_generating { "⏳Generating" } else { "✅ Ready" },
            ctx_used,
            self.last_generation_result
                .as_ref()
                .map(|r| format!("TOK:{:.0}", r.chars().count()))
                .unwrap_or_else(|| "TOKENS:--".to_string()),
            if self.input_mode == InputMode::Normal { "NORMAL" } else { "INSERT" },
            if self.is_generating { "STOP:c" } else { "HELP:/help" },
            self.messages.len(),
            hist
        );

        let status_widget = Paragraph::new(status)
            .style(Style::default().fg(Color::Black).bg(Color::Gray));

        f.render_widget(status_widget, area);
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}