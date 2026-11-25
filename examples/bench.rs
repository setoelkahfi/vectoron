//! Times prefill and decode for both kinds of kernels on one model.
//!
//! ```sh
//! cargo run --release --example bench -- model.gguf [prompt-tokens] [new-tokens]
//! ```
//!
//! Token ids are arbitrary: the cost of a forward pass does not depend on
//! them, so no tokenizer is needed.

use std::time::Instant;

use vectoron::sample::argmax;
use vectoron::{Kernels, Qwen2Model};

fn main() -> vectoron::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: bench <model.gguf> [prompt-tokens] [new-tokens]");
    let mut number = |default: usize| {
        args.next()
            .map_or(default, |v| v.parse().expect("a number"))
    };
    let prompt_len = number(179);
    let new_tokens = number(32);

    let mut model = Qwen2Model::load(std::fs::read(path).expect("read the file"))?;
    let vocab = model.config().vocab as u32;
    let prompt: Vec<u32> = (0..prompt_len as u32)
        .map(|i| (i * 977 + 13) % vocab)
        .collect();

    for kernels in [Kernels::F32, Kernels::Integer] {
        model.set_kernels(kernels);
        model.clear_kv_cache();
        let started = Instant::now();
        let mut logits = Vec::new();
        for chunk in prompt.chunks(64) {
            logits = model.forward(chunk, model.cache_len())?;
        }
        let prefill = started.elapsed().as_secs_f64();

        let started = Instant::now();
        for _ in 0..new_tokens {
            let next = argmax(&logits);
            logits = model.forward(&[next], model.cache_len())?;
        }
        let decode = started.elapsed().as_secs_f64();
        println!(
            "{kernels:?}: prefill {prompt_len} tokens in {:.0} ms, decode {:.2} tok/s",
            prefill * 1000.0,
            new_tokens as f64 / decode
        );
    }
    Ok(())
}
