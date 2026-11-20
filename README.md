# vectoron

A minimal, modular machine-learning framework for inference, written in Rust.

vectoron runs quantized language models on the CPU, natively and in the
browser (`wasm32-unknown-unknown`). It reads GGUF files in place and
dequantizes one block at a time inside the matrix multiply, so a model's
weights are never expanded to `f32` in memory. It has two dependencies,
`half` and `thiserror`, and none on another ML framework.

The API is young and will change.

## What it runs

- Architectures: Qwen2 and Qwen2.5.
- Storage types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.
- Sampling: greedy, temperature, top-k, top-p, frequency and presence penalties, seeded.

Tokenization is up to you. The examples use the `tokenizers` crate.

## Example

```rust,no_run
use vectoron::{Qwen2Model, Sampler};

let mut model = Qwen2Model::load(std::fs::read("model.gguf")?)?;
let prompt: Vec<u32> = vec![/* token ids */];
let mut sampler = Sampler::greedy();

let mut logits = model.forward(&prompt, 0)?;
let mut generated = Vec::new();
for _ in 0..64 {
    let next = sampler.sample(&mut logits, &generated);
    generated.push(next);
    logits = model.forward(&[next], model.cache_len())?;
}
```

From the command line, with a GGUF file and its `tokenizer.json`:

```sh
cargo run --release --example generate -- model.gguf tokenizer.json "Why is the sky blue?"
cargo run --release --example inspect -- model.gguf
```

## In the browser

Build with WebAssembly SIMD enabled:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

The inner loop then uses 128-bit SIMD. The crate is single threaded and
needs no special headers.

## Tests

`cargo test` runs the unit tests. The end-to-end test decodes greedily with
Qwen2.5-0.5B-Instruct Q4_K_M and compares 32 tokens against stored ids:

```sh
VECTORON_TEST_GGUF=/path/to/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf cargo test --release -- --ignored
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Copyright (c) 2026 Seto Elkahfi.
