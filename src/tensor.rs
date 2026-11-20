//! Quantized weight matrices and the vector operations a transformer needs.

use crate::error::{Error, Result};
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{GgmlType, QK_K, dequantize};
use crate::simd;

/// A weight matrix stored in a GGUF buffer, `rows` by `cols`, row major.
///
/// The matrix is a description of bytes owned elsewhere. Multiplication
/// dequantizes one block at a time into a small stack buffer, so the
/// matrix is never expanded to `f32` in memory.
#[derive(Debug, Clone)]
pub struct QMatrix {
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    offset: usize,
    row_bytes: usize,
}

impl QMatrix {
    /// Describes a 1-D or 2-D tensor as a matrix.
    pub fn new(info: &TensorInfo) -> Result<Self> {
        let cols = *info
            .dims
            .first()
            .ok_or_else(|| Error::Model(format!("tensor `{}` has no dimensions", info.name)))?;
        let rows = info.dims[1..].iter().product();
        Ok(Self {
            ty: info.ty,
            rows,
            cols,
            offset: info.offset,
            row_bytes: info.ty.row_bytes(cols),
        })
    }

    /// Looks up `name` in `gguf` and describes it as a matrix.
    pub fn from_gguf(gguf: &Gguf, name: &str) -> Result<Self> {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| Error::MissingTensor(name.into()))?;
        Self::new(info)
    }

    fn row<'a>(&self, data: &'a [u8], r: usize) -> &'a [u8] {
        let start = self.offset + r * self.row_bytes;
        &data[start..start + self.row_bytes]
    }

    /// Writes row `r` as `f32` values into `out` (length `cols`).
    pub fn dequantize_row(&self, data: &[u8], r: usize, out: &mut [f32]) {
        dequantize(self.ty, self.row(data, r), &mut out[..self.cols]);
    }

    /// `out[t][o] = row(o) · x[t]` for `t` input vectors of length `cols`
    /// stored back to back in `x`. `out` holds `t` vectors of length `rows`.
    ///
    /// Each block of a row is dequantized once and then applied to all `t`
    /// inputs, so a batch costs one pass over the weights.
    pub fn matmul(&self, data: &[u8], x: &[f32], t: usize, out: &mut [f32]) {
        debug_assert_eq!(x.len(), t * self.cols);
        debug_assert_eq!(out.len(), t * self.rows);
        out.fill(0.0);
        let mut buf = [0f32; QK_K];
        let chunk = QK_K - QK_K % self.ty.block_size();
        for o in 0..self.rows {
            let row = self.row(data, o);
            let mut col = 0;
            let mut byte = 0;
            while col < self.cols {
                let n = chunk.min(self.cols - col);
                let nb = self.ty.row_bytes(n);
                dequantize(self.ty, &row[byte..byte + nb], &mut buf[..n]);
                for ti in 0..t {
                    let xs = &x[ti * self.cols + col..][..n];
                    out[ti * self.rows + o] += simd::dot(&buf[..n], xs);
                }
                col += n;
                byte += nb;
            }
        }
    }
}

/// Reads a whole tensor as `f32` values. Used for small tensors such as
/// norm weights and biases.
pub fn load_vector(gguf: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| Error::MissingTensor(name.into()))?;
    let mut out = vec![0f32; info.elements()];
    dequantize(info.ty, gguf.data(info), &mut out);
    Ok(out)
}

/// RMS normalization of each row of `x` (row length `w.len()`), in place.
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(w.len()) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for (v, w) in row.iter_mut().zip(w) {
            *v *= scale * w;
        }
    }
}

/// Adds `b` to each row of `x` (row length `b.len()`).
pub fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        for (v, b) in row.iter_mut().zip(b) {
            *v += b;
        }
    }
}

/// Rotary position embedding, rotate-half form: element `i` pairs with
/// `i + d/2`. `v` is one head of length `d = 2 * inv_freq.len()`.
pub fn rope(v: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = inv_freq.len();
    for (i, f) in inv_freq.iter().enumerate() {
        let (s, c) = (pos as f32 * f).sin_cos();
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

/// Softmax in place.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        rope(&mut v, 0, &[1.0, 0.01]);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_pairs_half_apart() {
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        rope(&mut v, 1, &[std::f32::consts::FRAC_PI_2, 0.0]);
        assert!(v[0].abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_scales_to_unit_rms() {
        let mut x = vec![3.0, 4.0];
        rms_norm(&mut x, &[1.0, 1.0], 0.0);
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }
}
