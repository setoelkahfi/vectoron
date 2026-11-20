//! A minimal, modular machine-learning framework for inference.
//!
//! vectoron runs quantized language models on the CPU, natively and on
//! `wasm32-unknown-unknown`. It has no runtime dependency on another ML
//! framework.
//!
//! - [`gguf`] reads GGUF files without copying them.
//! - [`quant`] dequantizes the ggml block formats, one block at a time.
//! - [`tensor`] multiplies quantized matrices with `f32` vectors and holds
//!   the vector operations a transformer needs.
//! - [`qwen2`] is the Qwen2 / Qwen2.5 decoder.
//! - [`sample`] picks the next token.
//!
//! Tokenization is left to the caller.
//!
//! ```no_run
//! use vectoron::{Qwen2Model, Sampler};
//!
//! # fn main() -> vectoron::Result<()> {
//! let bytes = std::fs::read("model.gguf").expect("read the model");
//! let mut model = Qwen2Model::load(bytes)?;
//! let prompt: Vec<u32> = vec![/* token ids from your tokenizer */];
//! let mut sampler = Sampler::greedy();
//!
//! let mut logits = model.forward(&prompt, 0)?;
//! let mut generated = Vec::new();
//! for _ in 0..32 {
//!     let next = sampler.sample(&mut logits, &generated);
//!     generated.push(next);
//!     logits = model.forward(&[next], model.cache_len())?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Build for the browser with `RUSTFLAGS="-C target-feature=+simd128"` to
//! get the SIMD inner loop.

pub mod error;
pub mod gguf;
pub mod quant;
pub mod qwen2;
pub mod sample;
pub mod simd;
pub mod tensor;

pub use error::{Error, Result};
pub use gguf::Gguf;
pub use qwen2::{Qwen2Config, Qwen2Model};
pub use sample::{Sampler, SamplerConfig};
