//! build.rs - 编译时配置
//! 启用 llama-cpp 特性时从 llama.cpp 源码生成 FFI 绑定。
//! 默认使用最小化存根绑定，无需外部依赖。

use std::env;
use std::path::PathBuf;
use std::fs;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_dir.join("bindings.rs");

    #[cfg(feature = "llama-cpp")]
    try_generate_bindings(&bindings_path);

    #[cfg(not(feature = "llama-cpp"))]
    generate_stub_bindings(&bindings_path);

    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "llama-cpp")]
fn try_generate_bindings(bindings_path: &PathBuf) {
    // Try to find llama.h and generate bindings
    let search_paths = vec![
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default()).join("llama.cpp/llama.h"),
        PathBuf::from("/usr/include/llama.h"),
        PathBuf::from("/usr/local/include/llama.h"),
    ];

    let llama_h = search_paths.into_iter().find(|p| p.exists());
    
    let llama_h = match llama_h {
        Some(p) => p,
        None => {
            println!("cargo:warning=llama.h not found. Install llama.cpp or enable stub mode.");
            generate_stub_bindings(bindings_path);
            return;
        }
    };

    match bindgen::Builder::default()
        .header(llama_h.to_str().unwrap())
        .clang_arg("-x").clang_arg("c++")
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("LLAMA_.*")
        .generate_comments(false).use_core()
        .ctypes_prefix("std::os::raw")
        .generate()
    {
        Ok(bindings) => { bindings.write_to_file(bindings_path).ok(); }
        Err(e) => {
            println!("cargo:warning=bindgen failed: {}, using stub", e);
            generate_stub_bindings(bindings_path);
        }
    }
}

fn generate_stub_bindings(bindings_path: &PathBuf) {
    fs::write(bindings_path, STUB_BINDINGS).expect("write stub bindings");
}

const STUB_BINDINGS: &str = r#"#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
use std::os::raw::{c_char, c_void};
pub type llama_pos = i32; pub type llama_token = i32; pub type llama_seq_id = i32;
#[repr(C)] pub struct llama_context { _unused: [u8; 0] }
#[repr(C)] pub struct llama_model { _unused: [u8; 0] }
#[repr(C)] pub struct llama_vocab { _unused: [u8; 0] }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_model_params { pub n_gpu_layers: i32, pub split_mode: i32, pub main_gpu: i32, pub tensor_split: *const f32, pub progress_callback: Option<unsafe extern "C" fn(f32, *mut c_void)>, pub progress_callback_user_data: *mut c_void, pub vocab_only: bool, pub use_mmap: bool, pub use_mlock: bool, pub check_tensors: bool }
impl Default for llama_model_params { fn default() -> Self { unsafe { std::mem::zeroed() } } }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_context_params { pub seed: u32, pub n_ctx: u32, pub n_batch: u32, pub n_ubatch: u32, pub n_seq_max: u32, pub n_threads: i32, pub n_threads_batch: i32, pub rope_scaling_type: i32, pub pooling_type: i32, pub rope_freq_base: f32, pub rope_freq_scale: f32, pub yarn_ext_factor: f32, pub yarn_attn_factor: f32, pub yarn_beta_fast: f32, pub yarn_beta_slow: f32, pub yarn_orig_ctx: u32, pub defrag_thold: f32, pub cb_eval: Option<unsafe extern "C" fn(*mut c_void)>, pub cb_eval_user_data: *mut c_void, pub type_k: i32, pub type_v: i32, pub logits_all: bool, pub embeddings: bool, pub offload_kqv: bool, pub flash_attn: bool, pub no_perf: bool, pub abort_callback: Option<unsafe extern "C" fn(*mut c_void) -> bool>, pub abort_callback_user_data: *mut c_void }
impl Default for llama_context_params { fn default() -> Self { unsafe { std::mem::zeroed() } } }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_batch { pub n_tokens: i32, pub token: *const i32, pub embd: *const f32, pub pos: *mut i32, pub n_seq_id: *mut i32, pub seq_id: *mut *mut i32, pub logits: *mut i32 }
impl Default for llama_batch { fn default() -> Self { unsafe { std::mem::zeroed() } } }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_token_data { pub id: i32, pub logit: f32, pub p: f32 }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_token_data_array { pub data: *mut llama_token_data, pub size: i32, pub sorted: bool }
#[repr(C)] #[derive(Debug, Copy, Clone)]
pub struct llama_model_info { pub n_vocab: i32, pub n_ctx_train: i32, pub n_embd: i32, pub n_layer: i32, pub n_head: i32, pub n_head_kv: i32, pub n_rot: i32, pub n_swa: i32, pub n_embd_head_k: i32, pub n_embd_head_v: i32, pub n_expert: i32, pub n_expert_used: i32, pub f_norm_eps: f32, pub f_norm_rms_eps: f32, pub f_rope_freq_base: f32, pub f_rope_freq_scale: f32, pub size: u64, pub has_encoder: bool, pub has_decoder: bool, pub pooling_type: i32 }
extern "C" { pub fn llama_backend_init(numa: bool); pub fn llama_backend_free(); pub fn llama_model_default_params() -> llama_model_params; pub fn llama_context_default_params() -> llama_context_params; pub fn llama_load_model_from_file(path: *const c_char, params: llama_model_params) -> *mut llama_model; pub fn llama_free_model(model: *mut llama_model); pub fn llama_new_context_with_model(model: *mut llama_model, params: llama_context_params) -> *mut llama_context; pub fn llama_free(ctx: *mut llama_context); pub fn llama_model_get_vocab(model: *mut llama_model) -> *mut llama_vocab; pub fn llama_vocab_n_tokens(vocab: *mut llama_vocab) -> i32; pub fn llama_token_eos(vocab: *mut llama_vocab) -> i32; pub fn llama_token_bos(vocab: *mut llama_vocab) -> i32; pub fn llama_tokenize(vocab: *mut llama_vocab, text: *const c_char, text_len: usize, tokens: *mut i32, n_tokens_max: i32, add_bos: bool, add_eos: bool) -> i32; pub fn llama_token_to_piece(vocab: *mut llama_vocab, token: i32) -> *const c_char; pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32; pub fn llama_get_logits(ctx: *mut llama_context) -> *mut f32; pub fn llama_sample_top_p(ctx: *mut llama_context, candidates: *mut llama_token_data_array, p: f32, min_keep: usize); pub fn llama_sample_temp(ctx: *mut llama_context, candidates: *mut llama_token_data_array, temp: f32); pub fn llama_sample_token(ctx: *mut llama_context, candidates: *mut llama_token_data_array) -> i32; pub fn llama_kv_cache_clear(ctx: *mut llama_context); pub fn llama_n_embd(model: *mut llama_model) -> i32; pub fn llama_n_layer(model: *mut llama_model) -> i32; pub fn llama_n_head(model: *mut llama_model) -> i32; pub fn llama_model_info(model: *mut llama_model) -> llama_model_info; pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch; }
"#;
