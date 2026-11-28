# vectoron

A minimal, modular machine-learning framework for inference, written in Rust.

vectoron runs quantized language models on the CPU, natively and in the
browser (`wasm32-unknown-unknown`). It reads GGUF files in place and works on
one weight block at a time inside the matrix multiply, so a model's weights
are never expanded to `f32` in memory. For Q4_K, Q6_K, Q5_0 and Q8_0 weights
it quantizes the activations to 8 bits and multiplies in integers, as
llama.cpp does; `Kernels::F32` selects the slower `f32` path. It has two
dependencies, `half` and `thiserror`, and none on another ML framework.

The API is young and will change.

## What it runs

- Architectures: Qwen2, Qwen2.5 and Qwen3 (dense, not the mixture-of-experts variant).
- Storage types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.
- Sampling: greedy, temperature, top-k, top-p, frequency and presence penalties, seeded.

Tokenization is up to you. The examples use the `tokenizers` crate.

## Example

```rust,no_run
use vectoron::{QwenModel, Sampler};

let mut model = QwenModel::load(std::fs::read("model.gguf")?)?;
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

`cargo test` runs the unit tests. The end-to-end tests need model files.
Qwen2.5-0.5B-Instruct Q4_K_M is checked against stored greedy tokens, and
Qwen3-0.6B Q4_K_M against candle's. Both also check the integer kernels
against the `f32` path:

```sh
VECTORON_TEST_GGUF=/path/to/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
VECTORON_TEST_QWEN3_GGUF=/path/to/Qwen_Qwen3-0.6B-Q4_K_M.gguf \
    cargo test --release -- --ignored
```

`cargo run --release --example bench -- model.gguf` times both kinds of kernels.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Copyright (c) 2026 Seto Elkahfi.
