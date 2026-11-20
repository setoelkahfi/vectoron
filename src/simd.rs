//! The inner product every matrix multiply reduces to.
//!
//! On `wasm32` built with `target-feature=+simd128` this uses 128-bit
//! WebAssembly SIMD. Elsewhere it uses eight independent accumulators,
//! which the compiler vectorizes for the host.

/// `a · b` over the shorter of the two slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::dot(a, b)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        portable::dot(a, b)
    }
}

/// `y += alpha * x` over the shorter of the two slices.
#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod portable {
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut acc = [0f32; 8];
        let mut ca = a.chunks_exact(8);
        let mut cb = b.chunks_exact(8);
        for (x, y) in (&mut ca).zip(&mut cb) {
            for i in 0..8 {
                acc[i] += x[i] * y[i];
            }
        }
        let mut sum = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
        for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
            sum += x * y;
        }
        sum
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::*;

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: i + 8 <= n, so both 16-byte loads at i and i + 4 stay
            // inside each slice. WebAssembly loads do not require alignment.
            let (a0, a1, b0, b1) = unsafe {
                (
                    v128_load(a.as_ptr().add(i) as *const v128),
                    v128_load(a.as_ptr().add(i + 4) as *const v128),
                    v128_load(b.as_ptr().add(i) as *const v128),
                    v128_load(b.as_ptr().add(i + 4) as *const v128),
                )
            };
            acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
            acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
            i += 8;
        }
        let acc = f32x4_add(acc0, acc1);
        let mut sum = (f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<2>(acc))
            + (f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<3>(acc));
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_a_plain_sum() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.25).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-3);
    }
}
