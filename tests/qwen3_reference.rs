//! Qwen3-0.6B Q4_K_M against candle's greedy continuation of one prompt.
//!
//! Needs the model file (bartowski/Qwen_Qwen3-0.6B-GGUF, SHA-256
//! `9acfc1e001311f34b4252001b626f2e466d592a42065f66571bff3790d4e1b14`), so it
//! is ignored by default:
//!
//! ```sh
//! VECTORON_TEST_QWEN3_GGUF=/path/to/Qwen_Qwen3-0.6B-Q4_K_M.gguf \
//!     cargo test --release --test qwen3_reference -- --ignored
//! ```
//!
//! The fixture holds the prompt's token ids on its first line and 32 greedy
//! tokens from candle 0.11.1's `quantized_qwen3`, an independent
//! implementation, on its second.

mod common;

/// The `f32` path must reproduce candle's tokens: that is what shows the
/// architecture is right. The integer kernels must then agree with `f32` at 19
/// of 20 positions or more.
///
/// Measured over these 32 positions: `f32` matches candle at 32 natively and
/// under wasm simd128; the integer kernels agree with it at 31 and 32, with a
/// mean logit difference of 0.28 and 0.19 and a worst of 2.3 and 1.3. Qwen3's
/// logits drift more than Qwen2's, so its bounds are wider.
#[test]
#[ignore = "needs VECTORON_TEST_QWEN3_GGUF"]
fn qwen3_matches_the_reference() {
    let path = std::env::var("VECTORON_TEST_QWEN3_GGUF").expect("set VECTORON_TEST_QWEN3_GGUF");
    let (prompt, expected) = common::fixture(include_str!("fixtures/qwen3_0_6b_candle.txt"));
    let result = common::compare(std::fs::read(path).unwrap(), &prompt, &expected);
    let n = result.positions;
    assert!(
        result.f32_matches * 20 >= n * 19,
        "f32 {}",
        result.f32_matches
    );
    assert!(result.agreed * 20 >= n * 19, "agreed {}", result.agreed);
    assert!(result.mean < 0.5, "mean logit difference {}", result.mean);
    assert!(
        result.worst < 4.0,
        "worst logit difference {}",
        result.worst
    );
}
