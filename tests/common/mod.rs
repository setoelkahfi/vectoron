//! Helpers shared by the model reference tests.

use vectoron::sample::argmax;
use vectoron::{Kernels, QwenModel};

pub fn ids(line: &str) -> Vec<u32> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

/// A fixture: the prompt's token ids, then the expected continuation.
pub fn fixture(text: &str) -> (Vec<u32>, Vec<u32>) {
    let mut lines = text.lines();
    (ids(lines.next().unwrap()), ids(lines.next().unwrap()))
}

/// How the integer kernels compare with the `f32` ones, and both with a
/// reference continuation.
pub struct Comparison {
    pub positions: usize,
    /// Positions where each kind of kernel picks the reference's next token.
    pub f32_matches: usize,
    pub integer_matches: usize,
    /// Positions where the two kinds of kernels pick the same token.
    pub agreed: usize,
    /// Logit differences between the kinds of kernels, over the logits near
    /// the top (the ones that can change a choice).
    pub mean: f64,
    pub worst: f32,
}

/// Feeds both kinds of kernels the same sequence, `prompt` then `expected`
/// (teacher forcing), so one early difference cannot cascade.
pub fn compare(bytes: Vec<u8>, prompt: &[u32], expected: &[u32]) -> Comparison {
    let mut reference = QwenModel::load(bytes.clone()).unwrap();
    reference.set_kernels(Kernels::F32);
    let mut integer = QwenModel::load(bytes).unwrap();

    let mut input = prompt.to_vec();
    let (mut f32_matches, mut integer_matches, mut agreed) = (0, 0, 0);
    let mut worst = 0f32;
    let (mut total, mut count) = (0f64, 0usize);
    for &next in expected {
        let a = reference.forward(&input, reference.cache_len()).unwrap();
        let b = integer.forward(&input, integer.cache_len()).unwrap();
        f32_matches += usize::from(argmax(&a) == next);
        integer_matches += usize::from(argmax(&b) == next);
        agreed += usize::from(argmax(&a) == argmax(&b));
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
    let result = Comparison {
        positions: expected.len(),
        f32_matches,
        integer_matches,
        agreed,
        mean: if count > 0 { total / count as f64 } else { 0.0 },
        worst,
    };
    println!(
        "reference matched: f32 {}/{n}, integer {}/{n}; kernels agree {}/{n}; \
         mean difference {:.3}, worst {:.3}",
        result.f32_matches,
        result.integer_matches,
        result.agreed,
        result.mean,
        result.worst,
        n = result.positions,
    );
    result
}
