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

use vectoron::sample::argmax;
use vectoron::{Kernels, Qwen2Model, Sampler};

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

/// Both kernels read the same token sequence (teacher forcing), so one early
/// difference cannot cascade. They must pick the same next token almost
/// everywhere, and the logits near the top must stay close.
///
/// Rounding the activations to `i8` moves a logit by about 0.1 on average
/// (measured: native 0.105, wasm simd128 0.098) with a worst case of 0.6 to
/// 1.0 over these 32 positions. The bounds leave room for that tail. A broken
/// kernel is off by whole units and flips the top token.
#[test]
#[ignore = "needs VECTORON_TEST_GGUF"]
fn integer_kernels_agree_with_f32() {
    let path = std::env::var("VECTORON_TEST_GGUF").expect("set VECTORON_TEST_GGUF");
    let fixture = include_str!("fixtures/qwen2_0_5b_greedy.txt");
    let mut lines = fixture.lines();
    let prompt = ids(lines.next().unwrap());
    let expected = ids(lines.next().unwrap());

    let bytes = std::fs::read(path).unwrap();
    let mut reference = Qwen2Model::load(bytes.clone()).unwrap();
    reference.set_kernels(Kernels::F32);
    let mut integer = Qwen2Model::load(bytes).unwrap();

    let mut input = prompt;
    let mut agreed = 0;
    let mut worst = 0f32;
    let (mut total, mut count) = (0f64, 0usize);
    for &next in &expected {
        let a = reference.forward(&input, reference.cache_len()).unwrap();
        let b = integer.forward(&input, integer.cache_len()).unwrap();
        agreed += usize::from(argmax(&a) == argmax(&b));
        // Compare the logits that can matter: those within reach of the top.
        let top = a.iter().copied().fold(f32::MIN, f32::max);
        for (x, y) in a.iter().zip(&b) {
            if *x > top - 5.0 {
                worst = worst.max((x - y).abs());
                total += f64::from((x - y).abs());
                count += 1;
            }
        }
        input = vec![next];
    }
    let mean = total / count as f64;
    println!(
        "top-1 agreement {agreed}/{}, mean difference {mean:.3}, worst {worst:.3}",
        expected.len()
    );
    assert!(
        agreed * 20 >= expected.len() * 19,
        "{agreed} of {}",
        expected.len()
    );
    assert!(mean < 0.25, "mean logit difference {mean}");
    assert!(worst < 2.0, "worst logit difference {worst}");
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
