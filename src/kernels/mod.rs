//! Matrix products against quantized weights without expanding them to `f32`.
//!
//! The input vectors are quantized to `i8` once, with one scale per block, and
//! each weight block is then multiplied in integer arithmetic (the llama.cpp
//! design). Only the per-block scales are floating point. This trades a little
//! precision in the activations for far less work per weight.

pub mod scalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm_simd;

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
use scalar as backend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm_simd as backend;

use crate::quant::{GgmlType, QK_K};

/// Values per sum in [`Quantized::sums`], the sub-block size of Q4_K.
const SUM_GROUP: usize = 32;

/// How many values share one activation scale for weights of type `ty`, or
/// `None` if `ty` has no integer kernel and is dequantized instead.
pub fn activation_block(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::Q5_0 | GgmlType::Q8_0 => Some(32),
        GgmlType::Q4_K | GgmlType::Q6_K => Some(QK_K),
        _ => None,
    }
}

/// `t` vectors quantized to `i8` in blocks.
#[derive(Debug, Default)]
pub struct Quantized {
    block: usize,
    cols: usize,
    values: Vec<i8>,
    /// One scale per block: `x ≈ scales[b] * values[..]`.
    scales: Vec<f32>,
    /// Sum of each group of 32 values, which Q4_K's per-sub-block minimums
    /// multiply.
    sums: Vec<i32>,
}

impl Quantized {
    /// Quantizes `x`, vectors of length `cols` stored back to back, with one
    /// scale per `block` values. `cols` is a multiple of `block`, and `block`
    /// a multiple of 32.
    pub fn new(x: &[f32], cols: usize, block: usize) -> Self {
        let mut out = Self::default();
        out.quantize(x, cols, block);
        out
    }

    /// As [`Quantized::new`], reusing this value's buffers.
    pub fn quantize(&mut self, x: &[f32], cols: usize, block: usize) {
        debug_assert_eq!(block % SUM_GROUP, 0);
        debug_assert_eq!(cols % block, 0);
        debug_assert_eq!(x.len() % cols, 0);
        self.block = block;
        self.cols = cols;
        self.values.clear();
        self.values.resize(x.len(), 0);
        self.scales.clear();
        self.sums.clear();
        for (src, dst) in x
            .chunks_exact(block)
            .zip(self.values.chunks_exact_mut(block))
        {
            let max = src.iter().fold(0f32, |m, v| m.max(v.abs()));
            let (scale, inverse) = if max > 0.0 {
                (max / 127.0, 127.0 / max)
            } else {
                (0.0, 0.0)
            };
            for (d, &s) in dst.iter_mut().zip(src) {
                // Round half away from zero; the cast saturates.
                let v = s * inverse;
                *d = (v + 0.5f32.copysign(v)) as i8;
            }
            self.scales.push(scale);
            for group in dst.chunks_exact(SUM_GROUP) {
                self.sums.push(group.iter().map(|&v| i32::from(v)).sum());
            }
        }
    }

    /// Number of vectors held.
    pub fn len(&self) -> usize {
        self.values.len().checked_div(self.cols).unwrap_or(0)
    }

    /// Whether no vectors are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Input vectors a row kernel reads per pass over the weights.
const TILE: usize = 4;

/// `N` quantized vectors, as a row kernel reads them.
pub struct Vectors<'a, const N: usize> {
    values: [&'a [i8]; N],
    scales: [&'a [f32]; N],
    sums: [&'a [i32]; N],
}

impl Quantized {
    fn vectors<const N: usize>(&self, first: usize) -> Vectors<'_, N> {
        let blocks = self.cols / self.block;
        let groups = self.cols / SUM_GROUP;
        Vectors {
            values: std::array::from_fn(|n| &self.values[(first + n) * self.cols..][..self.cols]),
            scales: std::array::from_fn(|n| &self.scales[(first + n) * blocks..][..blocks]),
            sums: std::array::from_fn(|n| &self.sums[(first + n) * groups..][..groups]),
        }
    }
}

/// Writes `row · x[t]` to `out[t * stride]` for every vector in `x`, where
/// `row` is one row of a weight matrix of type `ty` with `x.cols` values.
///
/// Vectors are taken [`TILE`] at a time, so each weight block is unpacked
/// once for all of them.
///
/// # Panics
///
/// If `ty` has no integer kernel, or `x` was quantized with a different block
/// size than [`activation_block`] gives for `ty`.
pub fn dot_row(ty: GgmlType, row: &[u8], x: &Quantized, out: &mut [f32], stride: usize) {
    assert_eq!(
        activation_block(ty),
        Some(x.block),
        "activations quantized for another weight type"
    );
    let t = x.len();
    let mut first = 0;
    while first + TILE <= t {
        let sums = kernel::<TILE>(ty, row, &x.vectors(first));
        for (n, sum) in sums.into_iter().enumerate() {
            out[(first + n) * stride] = sum;
        }
        first += TILE;
    }
    while first < t {
        let [sum] = kernel::<1>(ty, row, &x.vectors(first));
        out[first * stride] = sum;
        first += 1;
    }
}

#[inline]
fn kernel<const N: usize>(ty: GgmlType, row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    match ty {
        GgmlType::Q8_0 => backend::row_q8_0(row, x),
        GgmlType::Q5_0 => backend::row_q5_0(row, x),
        GgmlType::Q4_K => backend::row_q4_k(row, x),
        GgmlType::Q6_K => backend::row_q6_k(row, x),
        _ => unreachable!("checked by activation_block"),
    }
}

fn array<const N: usize>(bytes: &[u8]) -> &[u8; N] {
    bytes[..N].try_into().expect("block is long enough")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequantize;

    /// A small deterministic generator, so failures reproduce.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    /// A row of random blocks with positive, moderate `f16` scales.
    fn random_row(rng: &mut Rng, ty: GgmlType, cols: usize) -> Vec<u8> {
        let mut row = vec![0u8; ty.row_bytes(cols)];
        for b in &mut row {
            *b = rng.next() as u8;
        }
        let scale = half::f16::from_f32(0.03).to_le_bytes();
        let min = half::f16::from_f32(0.2).to_le_bytes();
        for block in row.chunks_exact_mut(ty.type_size()) {
            match ty {
                GgmlType::Q8_0 => {
                    block[..2].copy_from_slice(&scale);
                    // GGUF writers keep Q8_0 values in -127..=127.
                    for q in &mut block[2..] {
                        if *q == 0x80 {
                            *q = 0x81;
                        }
                    }
                }
                GgmlType::Q5_0 => block[..2].copy_from_slice(&scale),
                GgmlType::Q4_K => {
                    block[..2].copy_from_slice(&scale);
                    block[2..4].copy_from_slice(&min);
                }
                GgmlType::Q6_K => block[208..].copy_from_slice(&scale),
                _ => unreachable!(),
            }
        }
        row
    }

    const TYPES: [GgmlType; 4] = [
        GgmlType::Q8_0,
        GgmlType::Q5_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ];

    /// The integer product equals the `f32` product up to the error made by
    /// rounding the activations: each is off by at most half a step, so the
    /// sum is off by at most `Σ |w_i| * scale / 2`.
    #[test]
    fn integer_dot_matches_dequantized_dot() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let cols = 4 * QK_K;
        let t = 3;
        for ty in TYPES {
            let block = activation_block(ty).unwrap();
            for _ in 0..20 {
                let row = random_row(&mut rng, ty, cols);
                let x: Vec<f32> = (0..t * cols).map(|_| rng.unit() * 4.0).collect();
                let q = Quantized::new(&x, cols, block);
                let mut out = vec![0f32; t];
                dot_row(ty, &row, &q, &mut out, 1);

                let mut w = vec![0f32; cols];
                dequantize(ty, &row, &mut w);
                for ti in 0..t {
                    let xs = &x[ti * cols..][..cols];
                    let exact: f64 = w
                        .iter()
                        .zip(xs)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let bound: f64 = w
                        .chunks_exact(block)
                        .zip(xs.chunks_exact(block))
                        .map(|(w, x)| {
                            let max = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                            let sum: f64 = w.iter().map(|&v| f64::from(v.abs())).sum();
                            sum * f64::from(max / 127.0) / 2.0
                        })
                        .sum();
                    let error = (f64::from(out[ti]) - exact).abs();
                    assert!(
                        error <= bound * 1.01 + 1e-3,
                        "{ty:?}: error {error} over bound {bound} (exact {exact})"
                    );
                }
            }
        }
    }

    /// The active backend and the scalar reference give the same result, for
    /// one vector and for a full tile. On native targets they are the same
    /// code; under wasm simd128 this checks the SIMD kernels.
    #[test]
    fn backend_matches_scalar() {
        let mut rng = Rng(0x1234_5678);
        let cols = 3 * QK_K;
        for ty in TYPES {
            let block = activation_block(ty).unwrap();
            for _ in 0..50 {
                let row = random_row(&mut rng, ty, cols);
                let x: Vec<f32> = (0..TILE * cols).map(|_| rng.unit()).collect();
                let q = Quantized::new(&x, cols, block);
                let got = kernel::<TILE>(ty, &row, &q.vectors(0));
                let want = scalar_kernel::<TILE>(ty, &row, &q.vectors(0));
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= w.abs() * 1e-5 + 1e-4, "{ty:?}: {g} vs {w}");
                }
                let [one] = kernel::<1>(ty, &row, &q.vectors(2));
                assert!((one - want[2]).abs() <= want[2].abs() * 1e-5 + 1e-4);
            }
        }
    }

    fn scalar_kernel<const N: usize>(ty: GgmlType, row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
        match ty {
            GgmlType::Q8_0 => scalar::row_q8_0(row, x),
            GgmlType::Q5_0 => scalar::row_q5_0(row, x),
            GgmlType::Q4_K => scalar::row_q4_k(row, x),
            GgmlType::Q6_K => scalar::row_q6_k(row, x),
            _ => unreachable!(),
        }
    }

    #[test]
    fn zero_vectors_quantize_to_zero() {
        let q = Quantized::new(&[0.0; 64], 32, 32);
        assert_eq!(q.len(), 2);
        assert!(q.values.iter().all(|&v| v == 0));
        assert!(q.scales.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn every_stride_slot_gets_its_vector() {
        let mut rng = Rng(7);
        let row = random_row(&mut rng, GgmlType::Q8_0, 64);
        // One full tile and a remainder.
        let t = TILE + 2;
        let x: Vec<f32> = (0..t * 64).map(|_| rng.unit()).collect();
        let all = Quantized::new(&x, 64, 32);
        let mut out = vec![f32::NAN; t * 3];
        dot_row(GgmlType::Q8_0, &row, &all, &mut out, 3);
        for i in 0..t {
            let one = Quantized::new(&x[i * 64..][..64], 64, 32);
            let mut single = [0f32];
            dot_row(GgmlType::Q8_0, &row, &one, &mut single, 1);
            let diff = (out[i * 3] - single[0]).abs();
            assert!(diff <= single[0].abs() * 1e-5 + 1e-5, "vector {i}");
            assert!(out[i * 3 + 1].is_nan() && out[i * 3 + 2].is_nan());
        }
    }
}
