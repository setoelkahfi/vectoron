//! Greedy decoding of Qwen2.5-0.5B-Instruct Q4_K_M against stored token ids.
//!
//! Needs the model file, so it is ignored by default:
//!
//! ```sh
//! VECTORON_TEST_GGUF=/path/to/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
//!     cargo test --release -- --ignored
//! ```
//!
//! The fixture holds the prompt's token ids on its first line and the 32
//! expected greedy tokens on its second. The stored tokens come from the
//! `f32` kernels; the integer kernels are checked against those, position by
//! position.

mod common;

use common::{fixture, ids};
use vectoron::{Kernels, Qwen2Model, Sampler};

#[test]
#[ignore = "needs VECTORON_TEST_GGUF"]
fn greedy_tokens_match_the_reference() {
    let path = std::env::var("VECTORON_TEST_GGUF").expect("set VECTORON_TEST_GGUF");
    let (prompt, expected) = fixture(include_str!("fixtures/qwen2_0_5b_greedy.txt"));

    let mut model = Qwen2Model::load(std::fs::read(path).unwrap()).unwrap();
    model.set_kernels(Kernels::F32);
    let mut sampler = Sampler::greedy();
    let mut logits = model.forward(&prompt, 0).unwrap();
    let mut got = Vec::new();
    for _ in 0..expected.len() {
        let next = sampler.sample(&mut logits, &got);
        got.push(next);
        logits = model.forward(&[next], model.cache_len()).unwrap();
    }
    assert_eq!(got, expected);
}

/// Both kernels read the same token sequence (teacher forcing). They must
/// pick the same next token almost everywhere, and the logits near the top
/// must stay close.
///
/// Rounding the activations to `i8` moves a logit by about 0.1 on average
/// (measured: native 0.105, wasm simd128 0.098) with a worst case of 0.6 to
/// 1.0 over these 32 positions. The bounds leave room for that tail. A broken
/// kernel is off by whole units and flips the top token.
#[test]
#[ignore = "needs VECTORON_TEST_GGUF"]
fn integer_kernels_agree_with_f32() {
    let path = std::env::var("VECTORON_TEST_GGUF").expect("set VECTORON_TEST_GGUF");
    let (prompt, expected) = fixture(include_str!("fixtures/qwen2_0_5b_greedy.txt"));
    let result = common::compare(std::fs::read(path).unwrap(), &prompt, &expected);
    let n = result.positions;
    assert!(result.agreed * 20 >= n * 19, "agreed {}", result.agreed);
    assert!(result.mean < 0.25, "mean logit difference {}", result.mean);
    assert!(
        result.worst < 2.0,
        "worst logit difference {}",
        result.worst
    );
}

#[test]
#[ignore = "needs VECTORON_TEST_GGUF"]
fn rewinding_the_cache_repeats_the_logits() {
    let path = std::env::var("VECTORON_TEST_GGUF").expect("set VECTORON_TEST_GGUF");
    let prompt = ids(include_str!("fixtures/qwen2_0_5b_greedy.txt")
        .lines()
        .next()
        .unwrap());
    let mut model = Qwen2Model::load(std::fs::read(path).unwrap()).unwrap();

    let batched = model.forward(&prompt, 0).unwrap();
    model.clear_kv_cache();
    let (head, tail) = prompt.split_at(prompt.len() - 3);
    model.forward(head, 0).unwrap();
    let stepped = model.forward(tail, head.len()).unwrap();
    // A reused buffer holds exactly the new logits, whatever was in it.
    model.clear_kv_cache();
    let mut reused = vec![f32::NAN; 3];
    model.forward_into(head, 0, &mut reused).unwrap();
    model.forward_into(tail, head.len(), &mut reused).unwrap();
    assert_eq!(reused, stepped);

    let diff = batched
        .iter()
        .zip(&stepped)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 1e-3, "max difference {diff}");
}
