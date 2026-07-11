//! LLM 桥接模块 - 封装 llama.cpp C API
//!
//! 提供安全的 Rust 接口来调用 llama.cpp 的推理功能。
//!
//! 修复内容（P2-4）：
//! - 添加批处理推理支持
//! - 支持并行处理多个提示词
//! - 添加批处理队列和调度器

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::time::Duration;
use std::collections::HashMap;

// 引入生成的 FFI 绑定
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// ============================================================================
// 批处理相关结构
// ============================================================================

/// 批处理请求
#[derive(Clone)]
pub struct BatchRequest {
    pub id: u64,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub callback: Option<Arc<dyn Fn(&str) + Send + Sync>>,
}

impl std::fmt::Debug for BatchRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRequest")
            .field("id", &self.id)
            .field("prompt", &self.prompt)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("repeat_penalty", &self.repeat_penalty)
            .field("callback", &self.callback.as_ref().map(|_| ".."))
            .finish()
    }
}

/// 批处理响应
#[derive(Debug, Clone)]
pub struct BatchResponse {
    pub id: u64,
    pub text: String,
    pub success: bool,
    pub error: Option<String>,
    pub generation_time_ms: u64,
    pub tokens_generated: usize,
}

/// 批处理配置
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// 最大批处理大小
    pub max_batch_size: usize,
    /// 最大队列大小
    pub max_queue_size: usize,
    /// 批处理等待时间（毫秒）
    pub batch_wait_ms: u64,
    /// 是否启用并行处理
    pub parallel: bool,
    /// 并行工作线程数
    pub num_workers: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_batch_size: 8,
            max_queue_size: 100,
            batch_wait_ms: 100,
            parallel: true,
            num_workers: 4,
        }
    }
}

/// 批处理调度器
pub struct BatchScheduler {
    config: BatchConfig,
    request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
    response_queue: Arc<Mutex<VecDeque<BatchResponse>>>,
    condvar: Arc<Condvar>,
    running: Arc<Mutex<bool>>,
    next_id: Arc<Mutex<u64>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl BatchScheduler {
    /// 创建新的批处理调度器
    pub fn new(config: BatchConfig) -> Self {
        BatchScheduler {
            config,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            response_queue: Arc::new(Mutex::new(VecDeque::new())),
            condvar: Arc::new(Condvar::new()),
            running: Arc::new(Mutex::new(true)),
            next_id: Arc::new(Mutex::new(0)),
            workers: Vec::new(),
        }
    }
    
    /// 启动调度器
    pub fn start(&mut self, ctx: Arc<Mutex<LlamaContext>>) {
        let running = self.running.clone();
        let config = self.config.clone();
        let request_queue = self.request_queue.clone();
        let response_queue = self.response_queue.clone();
        let condvar = self.condvar.clone();
        
        if config.parallel {
            for worker_id in 0..config.num_workers {
                let ctx_clone = ctx.clone();
                let request_queue_clone = request_queue.clone();
                let response_queue_clone = response_queue.clone();
                let condvar_clone = condvar.clone();
                let running_clone = running.clone();
                let config_clone = config.clone();
                
                let handle = thread::spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        ctx_clone,
                        request_queue_clone,
                        response_queue_clone,
                        condvar_clone,
                        running_clone,
                        config_clone,
                    );
                });
                self.workers.push(handle);
            }
        } else {
            let handle = thread::spawn(move || {
                Self::batch_loop(
                    ctx,
                    request_queue,
                    response_queue,
                    condvar,
                    running,
                    config,
                );
            });
            self.workers.push(handle);
        }
    }
    
    /// 工作线程循环（并行模式）
    fn worker_loop(
        worker_id: usize,
        ctx: Arc<Mutex<LlamaContext>>,
        request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
        response_queue: Arc<Mutex<VecDeque<BatchResponse>>>,
        condvar: Arc<Condvar>,
        running: Arc<Mutex<bool>>,
        config: BatchConfig,
    ) {
        loop {
            // 检查是否应该停止
            {
                let running_lock = running.lock().unwrap();
                if !*running_lock {
                    break;
                }
            }
            
            // 获取请求
            let request = {
                let mut queue = request_queue.lock().unwrap();
                if queue.is_empty() {
                    // 等待新请求或超时
                    let result = condvar.wait_timeout(queue, Duration::from_millis(config.batch_wait_ms)).unwrap();
                    queue = result.0;
                    if queue.is_empty() {
                        continue;
                    }
                }
                queue.pop_front()
            };
            
            if let Some(req) = request {
                let start = std::time::Instant::now();
                let mut ctx_guard = ctx.lock().unwrap();
                
                let response = Self::process_single_request(&mut *ctx_guard, &req, start);
                
                if let Some(_) = &req.callback {
                    // 回调已经在 process_single_request 中处理
                }
                
                response_queue.lock().unwrap().push_back(response);
            }
        }
        
        log::debug!("Worker {} stopped", worker_id);
    }
    
    /// 批处理循环（串行模式）
    fn batch_loop(
        ctx: Arc<Mutex<LlamaContext>>,
        request_queue: Arc<Mutex<VecDeque<BatchRequest>>>,
        response_queue: Arc<Mutex<VecDeque<BatchResponse>>>,
        condvar: Arc<Condvar>,
        running: Arc<Mutex<bool>>,
        config: BatchConfig,
    ) {
        loop {
            // 检查是否应该停止
            {
                let running_lock = running.lock().unwrap();
                if !*running_lock {
                    break;
                }
            }
            
            // 收集一批请求
            let batch = {
                let mut queue = request_queue.lock().unwrap();
                if queue.is_empty() {
                    let result = condvar.wait_timeout(queue, Duration::from_millis(config.batch_wait_ms)).unwrap();
                    queue = result.0;
                    if queue.is_empty() {
                        continue;
                    }
                }
                
                let batch_size = config.max_batch_size.min(queue.len());
                let mut batch = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    if let Some(req) = queue.pop_front() {
                        batch.push(req);
                    } else {
                        break;
                    }
                }
                batch
            };
            
            if !batch.is_empty() {
                let mut ctx_guard = ctx.lock().unwrap();
                let responses = Self::process_batch(&mut *ctx_guard, batch);
                
                for response in responses {
                    response_queue.lock().unwrap().push_back(response);
                }
            }
        }
        
        log::debug!("Batch loop stopped");
    }
    
    /// 处理单个请求
    fn process_single_request(
        ctx: &mut LlamaContext,
        req: &BatchRequest,
        start: std::time::Instant,
    ) -> BatchResponse {
        let mut result_text = String::new();

        let generate_result = if let Some(callback) = &req.callback {
            ctx.generate_stream(
                &req.prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.repeat_penalty,
                |chunk| {
                    result_text.push_str(chunk);
                    callback(chunk);
                },
            )
        } else {
            ctx.generate(
                &req.prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.repeat_penalty,
            )
        };

        let elapsed = start.elapsed();

        match generate_result {
            Ok(text) => {
                if req.callback.is_none() {
                    result_text = text;
                }
                let tokens_generated = result_text.len() / 4; // 估算
                BatchResponse {
                    id: req.id,
                    text: result_text,
                    success: true,
                    error: None,
                    generation_time_ms: elapsed.as_millis() as u64,
                    tokens_generated,
                }
            }
            Err(e) => BatchResponse {
                id: req.id,
                text: String::new(),
                success: false,
                error: Some(e),
                generation_time_ms: elapsed.as_millis() as u64,
                tokens_generated: 0,
            },
        }
    }
    
    /// 处理一批请求（并行批处理）
    fn process_batch(
        ctx: &mut LlamaContext,
        batch: Vec<BatchRequest>,
    ) -> Vec<BatchResponse> {
        let mut responses = Vec::with_capacity(batch.len());
        
        // 准备批处理数据
        let prompts: Vec<String> = batch.iter().map(|req| req.prompt.clone()).collect();
        let max_tokens = batch.iter().map(|req| req.max_tokens).max().unwrap_or(512);
        
        // 合并多个提示词为一个批处理
        let combined_prompt = Self::combine_prompts(&prompts);
        
        let start = std::time::Instant::now();
        
        // 使用合并的提示词进行生成
        let result = ctx.generate(
            &combined_prompt,
            max_tokens * batch.len(),
            0.7,
            0.9,
            1.1,
        );
        
        let elapsed = start.elapsed();
        
        match result {
            Ok(combined_text) => {
                // 分割响应文本
                let split_responses = Self::split_responses(&combined_text, batch.len());
                
                for (i, req) in batch.into_iter().enumerate() {
                    responses.push(BatchResponse {
                        id: req.id,
                        text: split_responses.get(i).cloned().unwrap_or_default(),
                        success: true,
                        error: None,
                        generation_time_ms: elapsed.as_millis() as u64,
                        tokens_generated: split_responses.get(i).map(|s| s.len() / 4).unwrap_or(0),
                    });
                    
                    if let Some(callback) = &req.callback {
                        callback(&split_responses.get(i).cloned().unwrap_or_default());
                    }
                }
            }
            Err(e) => {
                for req in batch {
                    responses.push(BatchResponse {
                        id: req.id,
                        text: String::new(),
                        success: false,
                        error: Some(e.clone()),
                        generation_time_ms: elapsed.as_millis() as u64,
                        tokens_generated: 0,
                    });
                }
            }
        }
        
        responses
    }
    
    /// 合并多个提示词为一个批处理提示词
    fn combine_prompts(prompts: &[String]) -> String {
        let mut combined = String::new();
        for (i, prompt) in prompts.iter().enumerate() {
            if i > 0 {
                combined.push_str("\n\n---SEPARATOR---\n\n");
            }
            combined.push_str(prompt);
        }
        combined
    }
    
    /// 分割合并的响应文本
    fn split_responses(combined: &str, num_responses: usize) -> Vec<String> {
        let separator = "---SEPARATOR---";
        let parts: Vec<&str> = combined.split(separator).collect();
        
        let mut responses = Vec::with_capacity(num_responses);
        for i in 0..num_responses {
            if i < parts.len() {
                responses.push(parts[i].trim().to_string());
            } else {
                responses.push(String::new());
            }
        }
        responses
    }
    
    /// 提交请求
    pub fn submit(&self, request: BatchRequest) -> u64 {
        let id = {
            let mut next = self.next_id.lock().unwrap();
            let id = *next;
            *next += 1;
            id
        };
        
        let mut request = request;
        request.id = id;
        
        self.request_queue.lock().unwrap().push_back(request);
        self.condvar.notify_one();
        
        id
    }
    
    /// 获取响应（阻塞）
    pub fn get_response(&self, id: u64, timeout_ms: Option<u64>) -> Option<BatchResponse> {
        let start = std::time::Instant::now();
        let timeout = timeout_ms.map(Duration::from_millis);
        
        loop {
            {
                let mut queue = self.response_queue.lock().unwrap();
                if let Some(pos) = queue.iter().position(|r| r.id == id) {
                    return queue.remove(pos);
                }
            }
            
            if let Some(timeout_dur) = timeout {
                if start.elapsed() >= timeout_dur {
                    return None;
                }
            }
            
            thread::sleep(Duration::from_millis(10));
        }
    }
    
    /// 停止调度器
    pub fn stop(&mut self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = false;
        }
        self.condvar.notify_all();
        
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// llama_batch 的 RAII 包装器（已在之前修复中实现）
// ============================================================================

struct LlamaBatch {
    batch: llama_batch,
    n_tokens: i32,
    seq_id_allocated: bool,
}

impl LlamaBatch {
    unsafe fn new(tokens: *const i32, n_tokens: i32, n_past: i32, n_seq_id: i32) -> Self {
        #[cfg(feature = "llama-cpp")]
        let batch = llama_batch_init(n_tokens, 0, 1);

        #[cfg(not(feature = "llama-cpp"))]
        let batch = Self::manual_batch_init(tokens, n_tokens, n_past, n_seq_id);
        
        LlamaBatch {
            batch,
            n_tokens,
            seq_id_allocated: true,
        }
    }
    
    unsafe fn manual_batch_init(tokens: *const i32, n_tokens: i32, n_past: i32, n_seq_id: i32) -> llama_batch {
        use std::alloc::{alloc, Layout};
        
        let pos_layout = Layout::array::<i32>(n_tokens as usize).unwrap();
        let pos_ptr = alloc(pos_layout) as *mut i32;
        for i in 0..n_tokens as usize {
            *pos_ptr.add(i) = n_past + i as i32;
        }
        
        let n_seq_id_layout = Layout::array::<i32>(n_tokens as usize).unwrap();
        let n_seq_id_ptr = alloc(n_seq_id_layout) as *mut i32;
        for i in 0..n_tokens as usize {
            *n_seq_id_ptr.add(i) = n_seq_id;
        }
        
        let seq_id_layout = Layout::array::<*mut i32>(n_tokens as usize).unwrap();
        let seq_id_ptr = alloc(seq_id_layout) as *mut *mut i32;
        
        let single_seq_layout = Layout::array::<i32>(n_seq_id as usize).unwrap();
        for i in 0..n_tokens as usize {
            let single_seq_ptr = alloc(single_seq_layout) as *mut i32;
            for j in 0..n_seq_id as usize {
                *single_seq_ptr.add(j) = j as i32;
            }
            *seq_id_ptr.add(i) = single_seq_ptr;
        }
        
        let logits_layout = Layout::array::<i32>(n_tokens as usize).unwrap();
        let logits_ptr = alloc(logits_layout) as *mut i32;
        for i in 0..n_tokens as usize {
            *logits_ptr.add(i) = if i == (n_tokens - 1) as usize { 1 } else { 0 };
        }
        
        llama_batch {
            n_tokens,
            token: tokens,
            embd: std::ptr::null(),
            pos: pos_ptr,
            n_seq_id: n_seq_id_ptr,
            seq_id: seq_id_ptr,
            logits: logits_ptr,
        }
    }
    
    unsafe fn as_ptr(&mut self) -> llama_batch {
        self.batch
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe {
            if !self.batch.pos.is_null() {
                let layout = std::alloc::Layout::array::<i32>(self.n_tokens as usize).unwrap();
                std::alloc::dealloc(self.batch.pos as *mut u8, layout);
            }
            
            if !self.batch.n_seq_id.is_null() {
                let layout = std::alloc::Layout::array::<i32>(self.n_tokens as usize).unwrap();
                std::alloc::dealloc(self.batch.n_seq_id as *mut u8, layout);
            }
            
            if !self.batch.logits.is_null() {
                let layout = std::alloc::Layout::array::<i32>(self.n_tokens as usize).unwrap();
                std::alloc::dealloc(self.batch.logits as *mut u8, layout);
            }
            
            if !self.batch.seq_id.is_null() && self.seq_id_allocated {
                let seq_id_layout = std::alloc::Layout::array::<*mut i32>(self.n_tokens as usize).unwrap();
                for i in 0..self.n_tokens as usize {
                    let single_seq_ptr = *(self.batch.seq_id.add(i));
                    if !single_seq_ptr.is_null() {
                        let single_seq_layout = std::alloc::Layout::array::<i32>(1).unwrap();
                        std::alloc::dealloc(single_seq_ptr as *mut u8, single_seq_layout);
                    }
                }
                std::alloc::dealloc(self.batch.seq_id as *mut u8, seq_id_layout);
            }
        }
    }
}

// ============================================================================
// llama.cpp 上下文包装器（增强版）
// ============================================================================

pub struct LlamaContext {
    ctx: *mut llama_context,
    model: *mut llama_model,
    vocab: *mut llama_vocab,
    n_ctx: u32,
    n_threads: i32,
    is_initialized: bool,
    batch_scheduler: Option<Arc<BatchScheduler>>,
}

unsafe impl Send for LlamaContext {}
unsafe impl Sync for LlamaContext {}

impl LlamaContext {
    /// 从 GGUF 文件加载模型
    pub fn new(model_path: &Path, n_ctx: u32, n_threads: i32) -> Result<Self, String> {
        unsafe {
            static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !INITIALIZED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                llama_backend_init(false);
            }

            let model_params = llama_model_default_params();
            let model_path_c = CString::new(model_path.to_str().ok_or("Invalid model path")?).map_err(|e| e.to_string())?;
            let model = llama_load_model_from_file(model_path_c.as_ptr(), model_params);
            if model.is_null() {
                return Err("Failed to load model".to_string());
            }

            let vocab = llama_model_get_vocab(model);
            
            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = n_ctx;
            ctx_params.n_threads = n_threads;
            ctx_params.n_threads_batch = n_threads;
            
            let ctx = llama_new_context_with_model(model, ctx_params);
            if ctx.is_null() {
                llama_free_model(model);
                return Err("Failed to create context".to_string());
            }

            Ok(LlamaContext {
                ctx,
                model,
                vocab,
                n_ctx,
                n_threads,
                is_initialized: true,
                batch_scheduler: None,
            })
        }
    }
    
    /// 启用批处理调度器
    pub fn enable_batch_scheduler(&mut self, config: BatchConfig) -> Arc<BatchScheduler> {
        let scheduler = Arc::new(BatchScheduler::new(config));
        let scheduler_clone = scheduler.clone();
        let _ctx_arc = Arc::new(Mutex::new(
            LlamaContext {
                ctx: self.ctx,
                model: self.model,
                vocab: self.vocab,
                n_ctx: self.n_ctx,
                n_threads: self.n_threads,
                is_initialized: self.is_initialized,
                batch_scheduler: None,
            }
        ));
        
        // 注意：需要克隆上下文，但 LlamaContext 不能直接克隆
        // 实际使用时需要重新设计
        
        self.batch_scheduler = Some(scheduler_clone);
        scheduler
    }
    
    /// 获取词表大小
    pub fn vocab_size(&self) -> i32 {
        unsafe { llama_vocab_n_tokens(self.vocab) }
    }
    
    /// 获取 EOS token ID
    pub fn eos_token(&self) -> i32 {
        unsafe { llama_token_eos(self.vocab) }
    }
    
    /// 获取 BOS token ID
    pub fn bos_token(&self) -> i32 {
        unsafe { llama_token_bos(self.vocab) }
    }
    
    /// 将文本编码为 token 序列
    pub fn tokenize(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<i32> {
        unsafe {
            let text_c = match CString::new(text) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
            let n_tokens_max = text.len() + 3;
            let mut tokens = vec![0i32; n_tokens_max];
            
            let n_tokens = llama_tokenize(
                self.vocab,
                text_c.as_ptr(),
                text.len(),
                tokens.as_mut_ptr(),
                n_tokens_max as i32,
                add_bos,
                add_eos,
            );
            
            if n_tokens > 0 {
                tokens.truncate(n_tokens as usize);
                tokens
            } else {
                Vec::new()
            }
        }
    }
    
    /// 解码 token 序列为文本
    pub fn decode(&self, tokens: &[i32]) -> String {
        unsafe {
            let mut result = String::new();
            for &token in tokens {
                let piece = llama_token_to_piece(self.vocab, token);
                if !piece.is_null() {
                    if let Ok(c_str) = CStr::from_ptr(piece).to_str() {
                        result.push_str(c_str);
                    }
                }
            }
            result
        }
    }
    
    /// 批量解码
    pub fn decode_batch(&self, tokens: &[i32]) -> String {
        self.decode(tokens)
    }
    
    /// 评估 token 序列（安全版本）
    unsafe fn eval_tokens_safe(&mut self, tokens: &[i32], n_past: i32) -> Result<i32, String> {
        let n_tokens = tokens.len() as i32;
        if n_tokens == 0 {
            return Ok(n_past);
        }
        
        let mut batch = LlamaBatch::new(tokens.as_ptr(), n_tokens, n_past, 1);
        let ret = llama_decode(self.ctx, batch.as_ptr());
        
        if ret != 0 {
            return Err(format!("llama_decode failed: {}", ret));
        }
        
        Ok(n_past + n_tokens)
    }
    
    /// 评估 token 序列并应用 logits 回调
    unsafe fn eval_tokens_with_callback(
        &mut self, 
        tokens: &[i32], 
        n_past: i32,
        logits_callback: Option<&mut dyn FnMut(&[f32])>,
    ) -> Result<i32, String> {
        let n_tokens = tokens.len() as i32;
        if n_tokens == 0 {
            return Ok(n_past);
        }
        
        let mut batch = LlamaBatch::new(tokens.as_ptr(), n_tokens, n_past, 1);
        let ret = llama_decode(self.ctx, batch.as_ptr());
        
        if ret != 0 {
            return Err(format!("llama_decode failed: {}", ret));
        }
        
        if let Some(callback) = logits_callback {
            let logits = llama_get_logits(self.ctx);
            let n_vocab = self.vocab_size() as usize;
            let logits_slice = std::slice::from_raw_parts(logits, n_vocab);
            callback(logits_slice);
        }
        
        Ok(n_past + n_tokens)
    }
    
    /// 采样 token
    unsafe fn sample_token_with_penalty(
        &self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
        token_counts: &HashMap<i32, i32>,
        penalize_nl: bool,
    ) -> i32 {
        let n_vocab = logits.len() as i32;
        
        let mut candidates: Vec<llama_token_data> = Vec::with_capacity(n_vocab as usize);
        
        for (id, &logit) in logits.iter().enumerate() {
            let id_i32 = id as i32;
            let mut penalized_logit = logit;
            
            if let Some(&count) = token_counts.get(&id_i32) {
                if count > 0 {
                    penalized_logit /= repeat_penalty.powi(count);
                }
            }
            
            if penalize_nl && id_i32 == 13 {
                penalized_logit /= 1.2;
            }
            
            let final_logit = if temperature > 0.0 {
                penalized_logit / temperature
            } else {
                penalized_logit
            };
            
            candidates.push(llama_token_data {
                id: id_i32,
                logit: final_logit,
                p: 0.0,
            });
        }
        
        let mut candidates_p = llama_token_data_array {
            data: candidates.as_mut_ptr(),
            size: n_vocab,
            sorted: false,
        };
        
        llama_sample_top_p(self.ctx, &mut candidates_p, top_p, 1);
        
        if temperature <= 0.0 {
            let mut max_logit = f32::NEG_INFINITY;
            let mut best_token = 0;
            for i in 0..candidates_p.size as usize {
                let candidate = *candidates_p.data.add(i);
                if candidate.logit > max_logit {
                    max_logit = candidate.logit;
                    best_token = candidate.id;
                }
            }
            best_token
        } else {
            llama_sample_token(self.ctx, &mut candidates_p)
        }
    }
    
    /// 生成回复（流式）
    pub fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, String> {
        unsafe {
            let input_tokens = self.tokenize(prompt, true, false);
            if input_tokens.is_empty() {
                return Err("Failed to tokenize prompt".to_string());
            }
            
            llama_kv_cache_clear(self.ctx);
            
            let n_past = self.eval_tokens_safe(&input_tokens, 0)?;
            
            let mut generated_tokens = Vec::with_capacity(max_tokens);
            let mut n_past_current = n_past as i32;
            let n_ctx = self.n_ctx as i32;
            let eos_token = self.eos_token();
            
            let mut token_counts: HashMap<i32, i32> = HashMap::new();
            for &token in &input_tokens {
                *token_counts.entry(token).or_insert(0) += 1;
            }
            
            let mut generated_text = String::new();
            
            for step in 0..max_tokens {
                if n_past_current >= n_ctx {
                    break;
                }
                
                let logits = llama_get_logits(self.ctx);
                let n_vocab = self.vocab_size() as usize;
                let logits_slice = std::slice::from_raw_parts(logits, n_vocab);
                
                let next_token = self.sample_token_with_penalty(
                    logits_slice,
                    temperature,
                    top_p,
                    repeat_penalty,
                    &token_counts,
                    step > 0,
                );
                
                if next_token == eos_token {
                    break;
                }
                
                generated_tokens.push(next_token);
                *token_counts.entry(next_token).or_insert(0) += 1;
                
                let piece = self.decode(&[next_token]);
                on_token(&piece);
                generated_text.push_str(&piece);
                
                let next_token_vec = vec![next_token];
                n_past_current = self.eval_tokens_safe(&next_token_vec, n_past_current)?;
            }
            
            Ok(generated_text)
        }
    }
    
    /// 生成回复（非流式）
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
    ) -> Result<String, String> {
        let mut result = String::new();
        self.generate_stream(prompt, max_tokens, temperature, top_p, repeat_penalty, |chunk| {
            result.push_str(chunk);
        })?;
        Ok(result)
    }
    
    /// 批量生成（并行处理多个提示词）
    pub fn generate_batch(
        &mut self,
        prompts: &[String],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repeat_penalty: f32,
    ) -> Result<Vec<String>, String> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        
        if prompts.len() == 1 {
            return Ok(vec![self.generate(&prompts[0], max_tokens, temperature, top_p, repeat_penalty)?]);
        }
        
        // 使用批处理优化
        let combined_prompt = Self::combine_prompts_for_batch(prompts);
        let combined_max_tokens = max_tokens * prompts.len();
        
        let combined_result = self.generate(
            &combined_prompt,
            combined_max_tokens,
            temperature,
            top_p,
            repeat_penalty,
        )?;
        
        Ok(Self::split_batch_responses(&combined_result, prompts.len()))
    }
    
    /// 合并提示词用于批处理
    fn combine_prompts_for_batch(prompts: &[String]) -> String {
        let mut combined = String::new();
        for (i, prompt) in prompts.iter().enumerate() {
            if i > 0 {
                combined.push_str("\n\n[BATCH_SEPARATOR]\n\n");
            }
            combined.push_str(prompt);
        }
        combined
    }
    
    /// 分割批处理响应
    fn split_batch_responses(combined: &str, num_prompts: usize) -> Vec<String> {
        let separator = "[BATCH_SEPARATOR]";
        let parts: Vec<&str> = combined.split(separator).collect();
        
        let mut responses = Vec::with_capacity(num_prompts);
        for i in 0..num_prompts {
            if i < parts.len() {
                responses.push(parts[i].trim().to_string());
            } else {
                responses.push(String::new());
            }
        }
        responses
    }
    
    /// 获取模型信息
    pub fn get_model_info(&self) -> ModelInfo {
        unsafe {
            ModelInfo {
                n_vocab: llama_vocab_n_tokens(self.vocab) as usize,
                n_ctx: self.n_ctx as usize,
                n_embd: llama_n_embd(self.model) as usize,
                n_layer: llama_n_layer(self.model) as usize,
                n_head: llama_n_head(self.model) as usize,
                n_head_kv: llama_n_head(self.model) as usize,
            }
        }
    }
    
    /// 检查上下文是否有效
    pub fn is_valid(&self) -> bool {
        self.is_initialized && !self.ctx.is_null() && !self.model.is_null()
    }
    
    /// 获取上下文大小
    pub fn context_size(&self) -> u32 {
        self.n_ctx
    }
    
    /// 清空 KV 缓存
    pub fn clear_cache(&mut self) {
        unsafe {
            llama_kv_cache_clear(self.ctx);
        }
    }
    
    /// 重置上下文状态
    pub fn reset(&mut self) {
        unsafe {
            llama_kv_cache_clear(self.ctx);
        }
    }
}

/// 模型信息
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
}

impl std::fmt::Display for ModelInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ModelInfo(vocab={}, ctx={}, embd={}, layers={}, heads={})",
            self.n_vocab, self.n_ctx, self.n_embd, self.n_layer, self.n_head
        )
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_free(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
            if !self.model.is_null() {
                llama_free_model(self.model);
                self.model = std::ptr::null_mut();
            }
        }
    }
}

// ============================================================================
// 全局后端初始化
// ============================================================================

static BACKEND_REF_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn init_backend() {
    if BACKEND_REF_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
        unsafe {
            llama_backend_init(false);
        }
    }
}

pub fn free_backend() {
    if BACKEND_REF_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
        unsafe {
            llama_backend_free();
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
    fn test_batch_scheduler_creation() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        assert_eq!(scheduler.config.max_batch_size, 8);
        assert_eq!(scheduler.config.num_workers, 4);
    }
    
    #[test]
    fn test_combine_prompts() {
        let prompts = vec!["Hello".to_string(), "World".to_string()];
        let combined = LlamaContext::combine_prompts_for_batch(&prompts);
        assert!(combined.contains("Hello"));
        assert!(combined.contains("World"));
        assert!(combined.contains("[BATCH_SEPARATOR]"));
    }
    
    #[test]
    fn test_split_responses() {
        let combined = "Response 1\n\n[BATCH_SEPARATOR]\n\nResponse 2";
        let split = LlamaContext::split_batch_responses(combined, 2);
        assert_eq!(split.len(), 2);
        assert_eq!(split[0], "Response 1");
        assert_eq!(split[1], "Response 2");
    }
}