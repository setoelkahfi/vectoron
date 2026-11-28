//! Portable row kernels: the fallback on targets without a SIMD backend, and
//! the reference the SIMD backend is tested against. Each unpacks a weight
//! block to `i8` and takes plain integer dot products.

use super::{Vectors, array};
use crate::quant::{f16_at, scale_min_k4};

/// `a · b` over `i8` values. Both slices have the same length.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// `Σ_g scale[g] * (a_g · b_g)` over consecutive groups of `GROUP` values.
#[inline]
fn dot_i8_scaled<const GROUP: usize>(a: &[i8], b: &[i8], scales: &[i32]) -> i32 {
    a.chunks_exact(GROUP)
        .zip(b.chunks_exact(GROUP))
        .zip(scales)
        .map(|((a, b), &s)| s * dot_i8(a, b))
        .sum()
}

// d: f16, qs: [i8; 32].
pub fn row_q8_0<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let mut out = [0f32; N];
    for (b, block) in row.chunks_exact(34).enumerate() {
        let d = f16_at(block, 0);
        let mut w = [0i8; 32];
        for (w, &q) in w.iter_mut().zip(&block[2..]) {
            *w = q as i8;
        }
        for n in 0..N {
            out[n] += d * x.scales[n][b] * dot_i8(&w, &x.values[n][b * 32..][..32]) as f32;
        }
    }
    out
}

// d: f16, qh: [u8; 4], qs: [u8; 16].
pub fn row_q5_0<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let mut out = [0f32; N];
    for (b, block) in row.chunks_exact(22).enumerate() {
        let d = f16_at(block, 0);
        let qh = u32::from_le_bytes(*array(&block[2..]));
        let mut w = [0i8; 32];
        unpack_q5_0(qh, array(&block[6..]), &mut w);
        for n in 0..N {
            out[n] += d * x.scales[n][b] * dot_i8(&w, &x.values[n][b * 32..][..32]) as f32;
        }
    }
    out
}

// d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128].
pub fn row_q4_k<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let mut out = [0f32; N];
    for (b, block) in row.chunks_exact(144).enumerate() {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let mut scales = [0i32; 8];
        let mut mins = [0i32; 8];
        for j in 0..8 {
            let (s, m) = scale_min_k4(j, &block[4..16]);
            scales[j] = i32::from(s);
            mins[j] = i32::from(m);
        }
        let mut w = [0i8; 256];
        unpack_q4_k(array(&block[16..]), &mut w);
        for n in 0..N {
            let products = dot_i8_scaled::<32>(&w, &x.values[n][b * 256..][..256], &scales);
            let offsets: i32 = mins
                .iter()
                .zip(&x.sums[n][b * 8..][..8])
                .map(|(m, s)| m * s)
                .sum();
            out[n] += x.scales[n][b] * (d * products as f32 - dmin * offsets as f32);
        }
    }
    out
}

// ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16.
pub fn row_q6_k<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let mut out = [0f32; N];
    for (b, block) in row.chunks_exact(210).enumerate() {
        let d = f16_at(block, 208);
        let mut scales = [0i32; 16];
        for (s, &q) in scales.iter_mut().zip(&block[192..208]) {
            *s = i32::from(q as i8);
        }
        let mut w = [0i8; 256];
        unpack_q6_k(array(block), array(&block[128..]), &mut w);
        for n in 0..N {
            let products = dot_i8_scaled::<16>(&w, &x.values[n][b * 256..][..256], &scales);
            out[n] += d * x.scales[n][b] * products as f32;
        }
    }
    out
}

/// Values of a Q5_0 block (`qh` holds the fifth bit of value `k` in bit `k`).
#[inline]
fn unpack_q5_0(qh: u32, qs: &[u8; 16], out: &mut [i8; 32]) {
    for j in 0..16 {
        let h0 = (((qh >> j) & 1) << 4) as u8;
        let h1 = (((qh >> (j + 16)) & 1) << 4) as u8;
        out[j] = ((qs[j] & 0x0F) | h0) as i8 - 16;
        out[j + 16] = ((qs[j] >> 4) | h1) as i8 - 16;
    }
}

/// The unsigned 4-bit values of a Q4_K block, before scales and mins.
#[inline]
fn unpack_q4_k(qs: &[u8; 128], out: &mut [i8; 256]) {
    for (q, y) in qs.chunks_exact(32).zip(out.chunks_exact_mut(64)) {
        for l in 0..32 {
            y[l] = (q[l] & 0x0F) as i8;
            y[l + 32] = (q[l] >> 4) as i8;
        }
    }
}

/// The signed 6-bit values of a Q6_K block, before scales.
#[inline]
fn unpack_q6_k(ql: &[u8; 128], qh: &[u8; 64], out: &mut [i8; 256]) {
    for half in 0..2 {
        let ql = &ql[64 * half..][..64];
        let qh = &qh[32 * half..][..32];
        let y = &mut out[128 * half..][..128];
        for l in 0..32 {
            let h = qh[l];
            y[l] = ((ql[l] & 0x0F) | ((h & 3) << 4)) as i8 - 32;
            y[l + 32] = ((ql[l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) as i8 - 32;
            y[l + 64] = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i8 - 32;
            y[l + 96] = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}
