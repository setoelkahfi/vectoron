//! Chats with a Qwen2 GGUF model from the command line.
//!
//! ```sh
//! cargo run --release --example generate -- model.gguf tokenizer.json "Why is the sky blue?"
//! ```
//!
//! Set `TEMPERATURE` to sample instead of decoding greedily.

use std::io::Write;
use std::time::Instant;

use tokenizers::Tokenizer;
use vectoron::{Qwen2Model, Sampler, SamplerConfig};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: generate <model.gguf> <tokenizer.json> <prompt>";
    let model_path = args.next().ok_or(usage)?;
    let tokenizer_path = args.next().ok_or(usage)?;
    let question = args.next().ok_or(usage)?;
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
    let started = Instant::now();
    let mut model = Qwen2Model::load(std::fs::read(&model_path)?)?;
    eprintln!(
        "loaded in {:.2}s: {:?}",
        started.elapsed().as_secs_f32(),
        model.config()
    );

    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    );
    let ids = tokenizer.encode(prompt, false)?.get_ids().to_vec();
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    let mut sampler = match std::env::var("TEMPERATURE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(temperature) => Sampler::new(SamplerConfig {
            temperature,
            seed: 7,
            ..Default::default()
        }),
        None => Sampler::greedy(),
    };

    let started = Instant::now();
    let mut logits = model.forward(&ids, 0)?;
    eprintln!(
        "prompt: {} tokens in {:.2}s",
        ids.len(),
        started.elapsed().as_secs_f32()
    );

    let started = Instant::now();
    let mut generated = Vec::new();
    let mut printed = 0;
    for _ in 0..max_tokens {
        let next = sampler.sample(&mut logits, &generated);
        if eos.contains(&next) {
            break;
        }
        generated.push(next);
        let text = tokenizer.decode(&generated, true)?;
        if !text.ends_with('\u{fffd}') {
            print!("{}", &text[printed..]);
            std::io::stdout().flush()?;
            printed = text.len();
        }
        model.forward_into(&[next], model.cache_len(), &mut logits)?;
    }
    let secs = started.elapsed().as_secs_f32();
    println!();
    eprintln!(
        "{} tokens in {secs:.2}s, {:.2} tokens/s",
        generated.len(),
        generated.len() as f32 / secs
    );
    Ok(())
}
