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
//! expected greedy tokens on its second.

use vectoron::{Qwen2Model, Sampler};

fn ids(line: &str) -> Vec<u32> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

#[test]
#[ignore = "needs VECTORON_TEST_GGUF"]
fn greedy_tokens_match_the_reference() {
    let path = std::env::var("VECTORON_TEST_GGUF").expect("set VECTORON_TEST_GGUF");
    let fixture = include_str!("fixtures/qwen2_0_5b_greedy.txt");
    let mut lines = fixture.lines();
    let prompt = ids(lines.next().unwrap());
    let expected = ids(lines.next().unwrap());

    let mut model = Qwen2Model::load(std::fs::read(path).unwrap()).unwrap();
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
    let diff = batched
        .iter()
        .zip(&stepped)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 1e-3, "max difference {diff}");
}
