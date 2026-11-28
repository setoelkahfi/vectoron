//! Tensor storage types and their dequantization.
//!
//! Block layouts follow the ggml definitions used by GGUF files. Every
//! function here turns a run of whole blocks into `f32` values; nothing
//! keeps a dequantized copy of a weight matrix.

use half::{bf16, f16};

use crate::error::{Error, Result};

/// Values per K-quant super-block.
pub const QK_K: usize = 256;

/// How a tensor's values are stored.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl GgmlType {
    /// Maps a ggml type id to a storage type.
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            other => return Err(Error::UnsupportedType(other)),
        })
    }

    /// Values per block.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => QK_K,
        }
    }

    /// Bytes per block.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    /// Bytes needed to store `n` values. `n` must be a multiple of the block size.
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.block_size() * self.type_size()
    }
}

#[inline]
pub(crate) fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Dequantizes `out.len()` values from `src`, which holds exactly those
/// values as whole blocks of type `ty`.
pub fn dequantize(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    let bs = ty.block_size();
    let ts = ty.type_size();
    debug_assert_eq!(out.len() % bs, 0);
    debug_assert_eq!(src.len(), out.len() / bs * ts);
    match ty {
        GgmlType::F32 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (o, b) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = bf16::from_le_bytes([b[0], b[1]]).to_f32();
            }
        }
        _ => {
            for (block, y) in src.chunks_exact(ts).zip(out.chunks_exact_mut(bs)) {
                match ty {
                    GgmlType::Q4_0 => q4_0(block, y),
                    GgmlType::Q4_1 => q4_1(block, y),
                    GgmlType::Q5_0 => q5_0(block, y),
                    GgmlType::Q5_1 => q5_1(block, y),
                    GgmlType::Q8_0 => q8_0(block, y),
                    GgmlType::Q4_K => q4_k(block, y),
                    GgmlType::Q5_K => q5_k(block, y),
                    GgmlType::Q6_K => q6_k(block, y),
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => unreachable!(),
                }
            }
        }
    }
}

// d: f16, qs: [u8; 16]. Low nibbles are values 0..16, high nibbles 16..32.
fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
        y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
    }
}

// d: f16, m: f16, qs: [u8; 16].
fn q4_1(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    let qs = &b[4..20];
    for j in 0..16 {
        y[j] = (qs[j] & 0x0F) as f32 * d + m;
        y[j + 16] = (qs[j] >> 4) as f32 * d + m;
    }
}

// d: f16, qh: [u8; 4] (fifth bits), qs: [u8; 16].
fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    for j in 0..16 {
        let h0 = (((qh >> j) << 4) & 0x10) as u8;
        let h1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = (((qs[j] & 0x0F) | h0) as i32 - 16) as f32 * d;
        y[j + 16] = (((qs[j] >> 4) | h1) as i32 - 16) as f32 * d;
    }
}

// d: f16, m: f16, qh: [u8; 4], qs: [u8; 16].
fn q5_1(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let qs = &b[8..24];
    for j in 0..16 {
        let h0 = (((qh >> j) << 4) & 0x10) as u8;
        let h1 = ((qh >> (j + 12)) & 0x10) as u8;
        y[j] = ((qs[j] & 0x0F) | h0) as f32 * d + m;
        y[j + 16] = ((qs[j] >> 4) | h1) as f32 * d + m;
    }
}

// d: f16, qs: [i8; 32].
fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    for (o, &q) in y.iter_mut().zip(&b[2..34]) {
        *o = q as i8 as f32 * d;
    }
}

/// The 6-bit scale and min of sub-block `j` (0..8) of a Q4_K or Q5_K block.
#[inline]
pub(crate) fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

// d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128].
fn q4_k(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    for (n, (q, y)) in qs.chunks_exact(32).zip(y.chunks_exact_mut(64)).enumerate() {
        let (sc1, m1) = scale_min_k4(2 * n, scales);
        let (sc2, m2) = scale_min_k4(2 * n + 1, scales);
        let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
        for l in 0..32 {
            y[l] = d1 * (q[l] & 0x0F) as f32 - m1;
            y[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
        }
    }
}

// d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128].
fn q5_k(b: &[u8], y: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    for (n, (q, y)) in qs.chunks_exact(32).zip(y.chunks_exact_mut(64)).enumerate() {
        let (sc1, m1) = scale_min_k4(2 * n, scales);
        let (sc2, m2) = scale_min_k4(2 * n + 1, scales);
        let (d1, m1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, m2) = (d * sc2 as f32, dmin * m2 as f32);
        let u1 = 1u8 << (2 * n);
        let u2 = 2u8 << (2 * n);
        for l in 0..32 {
            let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[l] = d1 * ((q[l] & 0x0F) + h1) as f32 - m1;
            y[l + 32] = d2 * ((q[l] >> 4) + h2) as f32 - m2;
        }
    }
}

// ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16.
fn q6_k(b: &[u8], y: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16_at(b, 208);
    for half in 0..2 {
        let ql = &ql[64 * half..];
        let qh = &qh[32 * half..];
        let sc = &sc[8 * half..];
        let y = &mut y[128 * half..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bytes(v: f32) -> [u8; 2] {
        f16::from_f32(v).to_le_bytes()
    }

    #[test]
    fn q8_0_scales_each_value() {
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&f16_bytes(0.5));
        for i in 0..32 {
            b[2 + i] = (i as i8 - 16) as u8;
        }
        let mut y = [0f32; 32];
        dequantize(GgmlType::Q8_0, &b, &mut y);
        for (i, v) in y.iter().enumerate() {
            assert_eq!(*v, (i as f32 - 16.0) * 0.5);
        }
    }

    #[test]
    fn q4_0_splits_nibbles() {
        let mut b = vec![0u8; 18];
        b[..2].copy_from_slice(&f16_bytes(2.0));
        b[2] = 0xF1; // low 1, high 15
        let mut y = [0f32; 32];
        dequantize(GgmlType::Q4_0, &b, &mut y);
        assert_eq!(y[0], (1.0 - 8.0) * 2.0);
        assert_eq!(y[16], (15.0 - 8.0) * 2.0);
        assert_eq!(y[1], -16.0);
    }

    #[test]
    fn q5_0_uses_the_fifth_bit() {
        let mut b = vec![0u8; 22];
        b[..2].copy_from_slice(&f16_bytes(1.0));
        // fifth bit set for value 0 (bit 0) and value 16 (bit 16)
        b[2..6].copy_from_slice(&(1u32 | (1 << 16)).to_le_bytes());
        b[6] = 0x00;
        let mut y = [0f32; 32];
        dequantize(GgmlType::Q5_0, &b, &mut y);
        assert_eq!(y[0], 0.0); // 16 - 16
        assert_eq!(y[16], 0.0);
        assert_eq!(y[1], -16.0);
    }

    #[test]
    fn q4_k_applies_scale_and_min() {
        let mut b = vec![0u8; 144];
        b[..2].copy_from_slice(&f16_bytes(1.0));
        b[2..4].copy_from_slice(&f16_bytes(1.0));
        // sub-block 0: scale 2, min 3. sub-block 1: scale 4, min 5.
        b[4] = 2;
        b[5] = 4;
        b[8] = 3;
        b[9] = 5;
        b[16] = 0x37; // value 0 low nibble 7, value 32 high nibble 3
        let mut y = [0f32; 256];
        dequantize(GgmlType::Q4_K, &b, &mut y);
        assert_eq!(y[0], 2.0 * 7.0 - 3.0);
        assert_eq!(y[32], 4.0 * 3.0 - 5.0);
        assert_eq!(y[1], -3.0);
    }

    #[test]
    fn scale_min_k4_high_sub_blocks() {
        let mut s = [0u8; 12];
        // sub-block 4: scale = (s[8] & 0xF) | (s[0] >> 6) << 4, min = (s[8] >> 4) | (s[4] >> 6) << 4
        s[8] = 0x21;
        s[0] = 0b1100_0000;
        s[4] = 0b0100_0000;
        assert_eq!(scale_min_k4(4, &s), (1 | (3 << 4), 2 | (1 << 4)));
    }

    #[test]
    fn q6_k_combines_low_and_high_bits() {
        let mut b = vec![0u8; 210];
        b[0] = 0x05; // value 0 low 4 bits = 5, value 64 low 4 bits = 0
        b[128] = 0b0000_0010; // value 0 high 2 bits = 2
        b[192] = 1; // scale for values 0..16
        b[208..210].copy_from_slice(&f16_bytes(1.0));
        let mut y = [0f32; 256];
        dequantize(GgmlType::Q6_K, &b, &mut y);
        assert_eq!(y[0], (5 | (2 << 4)) as f32 - 32.0);
        assert_eq!(y[1], -32.0);
    }
}
