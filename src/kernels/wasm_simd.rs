//! wasm simd128 versions of the row kernels in [`super::scalar`].
//!
//! Weight blocks are unpacked straight into registers and applied to `N`
//! input vectors before the next block is read. Sums stay in vector lanes and
//! are reduced once per row.
//!
//! simd128 has no `maddubs` or `sdot`, so an `i8 × i8` product widens with
//! `extmul` (`i16`, at most 128 · 127, which fits) and pairs are added into
//! `i32` lanes with `extadd_pairwise`.

use std::arch::wasm32::*;

use super::Vectors;
use crate::quant::{f16_at, scale_min_k4};

#[inline(always)]
fn load(bytes: &[u8]) -> v128 {
    let bytes: &[u8; 16] = bytes[..16].try_into().expect("16 bytes");
    // SAFETY: `bytes` is 16 readable bytes, and v128_load has no alignment
    // requirement.
    unsafe { v128_load(bytes.as_ptr() as *const v128) }
}

#[inline(always)]
fn load_i8(values: &[i8]) -> v128 {
    let values: &[i8; 16] = values[..16].try_into().expect("16 values");
    // SAFETY: as in `load`.
    unsafe { v128_load(values.as_ptr() as *const v128) }
}

/// The four `i32` partial sums of `a · b` for two 16-lane `i8` vectors.
#[inline(always)]
fn dot16(a: v128, b: v128) -> v128 {
    i32x4_add(
        i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(a, b)),
        i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(a, b)),
    )
}

#[inline(always)]
fn sum_lanes<const N: usize>(acc: [v128; N]) -> [f32; N] {
    acc.map(|v| {
        (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v))
            + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v))
    })
}

/// `acc[n] += (a0 · x[n][at..] + a1 · x[n][at + 16..]) * scale * dx[n][b]`
/// for a 32-value block held in two registers.
#[inline(always)]
fn add_block32<const N: usize>(
    acc: &mut [v128; N],
    w: (v128, v128),
    scale: f32,
    x: &Vectors<'_, N>,
    b: usize,
) {
    for n in 0..N {
        let xs = &x.values[n][b * 32..][..32];
        let products = i32x4_add(
            dot16(w.0, load_i8(&xs[..16])),
            dot16(w.1, load_i8(&xs[16..])),
        );
        let scaled = f32x4_mul(
            f32x4_convert_i32x4(products),
            f32x4_splat(scale * x.scales[n][b]),
        );
        acc[n] = f32x4_add(acc[n], scaled);
    }
}

// d: f16, qs: [i8; 32].
pub fn row_q8_0<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let mut acc = [f32x4_splat(0.0); N];
    for (b, block) in row.chunks_exact(34).enumerate() {
        let w = (load(&block[2..]), load(&block[18..]));
        add_block32(&mut acc, w, f16_at(block, 0), x, b);
    }
    sum_lanes(acc)
}

// d: f16, qh: [u8; 4], qs: [u8; 16].
pub fn row_q5_0<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    // Byte k of the unpacked block takes bit k of `qh`: pick the byte of
    // `qh` that holds it, mask the bit, and turn "set" into 16.
    let lanes_low = i8x16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    let lanes_high = i8x16(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    let mask = i8x16(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    let nibble = u8x16_splat(0x0F);
    let sixteen = i8x16_splat(16);

    let mut acc = [f32x4_splat(0.0); N];
    for (b, block) in row.chunks_exact(22).enumerate() {
        let qh = u32x4_splat(u32::from_le_bytes([block[2], block[3], block[4], block[5]]));
        let fifth = |lanes: v128| {
            let picked = v128_and(i8x16_swizzle(qh, lanes), mask);
            v128_and(i8x16_eq(picked, mask), sixteen)
        };
        let q = load(&block[6..]);
        let w = (
            i8x16_sub(v128_or(v128_and(q, nibble), fifth(lanes_low)), sixteen),
            i8x16_sub(v128_or(u8x16_shr(q, 4), fifth(lanes_high)), sixteen),
        );
        add_block32(&mut acc, w, f16_at(block, 0), x, b);
    }
    sum_lanes(acc)
}

// d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128].
pub fn row_q4_k<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let nibble = u8x16_splat(0x0F);
    let mut acc = [0f32; N];
    for (b, block) in row.chunks_exact(144).enumerate() {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let qs = &block[16..144];
        let mut sum = [i32x4_splat(0); N];
        let mut mins = [0i32; 8];
        // Sub-block `j` holds values 32j..32j+32: sub-blocks 2i and 2i+1 are
        // the low and high nibbles of bytes 32i..32i+32.
        for j in 0..8 {
            let (scale, min) = scale_min_k4(j, &block[4..16]);
            mins[j] = i32::from(min);
            let scale = i32x4_splat(i32::from(scale));
            let bytes = &qs[32 * (j / 2)..][..32];
            let shift = 4 * (j as u32 % 2);
            let w0 = v128_and(u8x16_shr(load(&bytes[..16]), shift), nibble);
            let w1 = v128_and(u8x16_shr(load(&bytes[16..]), shift), nibble);
            for n in 0..N {
                let xs = &x.values[n][b * 256 + 32 * j..][..32];
                let products =
                    i32x4_add(dot16(w0, load_i8(&xs[..16])), dot16(w1, load_i8(&xs[16..])));
                sum[n] = i32x4_add(sum[n], i32x4_mul(products, scale));
            }
        }
        for n in 0..N {
            let products = i32x4_extract_lane::<0>(sum[n])
                + i32x4_extract_lane::<1>(sum[n])
                + i32x4_extract_lane::<2>(sum[n])
                + i32x4_extract_lane::<3>(sum[n]);
            let offsets: i32 = mins
                .iter()
                .zip(&x.sums[n][b * 8..][..8])
                .map(|(m, s)| m * s)
                .sum();
            acc[n] += x.scales[n][b] * (d * products as f32 - dmin * offsets as f32);
        }
    }
    acc
}

// ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16.
pub fn row_q6_k<const N: usize>(row: &[u8], x: &Vectors<'_, N>) -> [f32; N] {
    let nibble = u8x16_splat(0x0F);
    let two = u8x16_splat(3);
    let bias = i8x16_splat(32);
    let mut acc = [f32x4_splat(0.0); N];
    for (b, block) in row.chunks_exact(210).enumerate() {
        let d = f16_at(block, 208);
        let mut sum = [i32x4_splat(0); N];
        // Group `c` holds values 16c..16c+16, with its own scale. Within a
        // half, groups walk: which quarter of `qh`'s bits and `ql` (2), then
        // the offset into it (2).
        for c in 0..16 {
            let (half, quarter, part) = (c / 8, (c % 8) / 2, c % 2);
            let at = 64 * half + 32 * (quarter % 2) + 16 * part;
            let low = load(&block[at..]);
            let low = if quarter < 2 {
                v128_and(low, nibble)
            } else {
                u8x16_shr(low, 4)
            };
            let high = load(&block[128 + 32 * half + 16 * part..]);
            let high = u8x16_shl(v128_and(u8x16_shr(high, 2 * quarter as u32), two), 4);
            let w = i8x16_sub(v128_or(low, high), bias);
            let scale = i32x4_splat(i32::from(block[192 + c] as i8));
            for n in 0..N {
                let xs = load_i8(&x.values[n][b * 256 + 16 * c..][..16]);
                sum[n] = i32x4_add(sum[n], i32x4_mul(dot16(w, xs), scale));
            }
        }
        for n in 0..N {
            let scaled = f32x4_mul(f32x4_convert_i32x4(sum[n]), f32x4_splat(d * x.scales[n][b]));
            acc[n] = f32x4_add(acc[n], scaled);
        }
    }
    sum_lanes(acc)
}
