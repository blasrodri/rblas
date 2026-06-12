//! AArch64 NEON Level-1 kernels (128-bit: 4×f32 / 2×f64).
//!
//! NEON is part of the mandatory aarch64 baseline, so these are always safe to
//! reach on this architecture. Each function is `#[target_feature(enable = "neon")]`
//! to let the compiler emit the intrinsics without per-call detection overhead;
//! callers reach them only on aarch64.

use crate::Float;
use core::arch::aarch64::*;

// ---------------------------------------------------------------- f32 (4-wide)

#[target_feature(enable = "neon")]
pub unsafe fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let va = vdupq_n_f32(alpha);
    let mut i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vy = vld1q_f32(y.as_ptr().add(i));
        let r = vfmaq_f32(vy, va, vx); // y + alpha*x
        vst1q_f32(y.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < n {
        *y.get_unchecked_mut(i) = Float::mul_add(alpha, *x.get_unchecked(i), *y.get_unchecked(i));
        i += 1;
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;
    // Two vector accumulators (8 lanes) to hide FMA latency.
    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, x0, y0);
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let y1 = vld1q_f32(y.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, x1, y1);
        i += 8;
    }
    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, x0, y0);
        i += 4;
    }
    let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        acc = Float::mul_add(*x.get_unchecked(i), *y.get_unchecked(i), acc);
        i += 1;
    }
    acc
}

#[target_feature(enable = "neon")]
pub unsafe fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let va = vdupq_n_f32(alpha);
    let mut i = 0;
    while i + 4 <= n {
        let vx = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(va, vx));
        i += 4;
    }
    while i < n {
        *x.get_unchecked_mut(i) *= alpha;
        i += 1;
    }
}

// ---------------------------------------------------------------- f64 (2-wide)

#[target_feature(enable = "neon")]
pub unsafe fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let va = vdupq_n_f64(alpha);
    let mut i = 0;
    while i + 2 <= n {
        let vx = vld1q_f64(x.as_ptr().add(i));
        let vy = vld1q_f64(y.as_ptr().add(i));
        vst1q_f64(y.as_mut_ptr().add(i), vfmaq_f64(vy, va, vx));
        i += 2;
    }
    while i < n {
        *y.get_unchecked_mut(i) = Float::mul_add(alpha, *x.get_unchecked(i), *y.get_unchecked(i));
        i += 1;
    }
}

#[target_feature(enable = "neon")]
pub unsafe fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut i = 0;
    while i + 4 <= n {
        let x0 = vld1q_f64(x.as_ptr().add(i));
        let y0 = vld1q_f64(y.as_ptr().add(i));
        acc0 = vfmaq_f64(acc0, x0, y0);
        let x1 = vld1q_f64(x.as_ptr().add(i + 2));
        let y1 = vld1q_f64(y.as_ptr().add(i + 2));
        acc1 = vfmaq_f64(acc1, x1, y1);
        i += 4;
    }
    while i + 2 <= n {
        let x0 = vld1q_f64(x.as_ptr().add(i));
        let y0 = vld1q_f64(y.as_ptr().add(i));
        acc0 = vfmaq_f64(acc0, x0, y0);
        i += 2;
    }
    let mut acc = vaddvq_f64(vaddq_f64(acc0, acc1));
    while i < n {
        acc = Float::mul_add(*x.get_unchecked(i), *y.get_unchecked(i), acc);
        i += 1;
    }
    acc
}

#[target_feature(enable = "neon")]
pub unsafe fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let va = vdupq_n_f64(alpha);
    let mut i = 0;
    while i + 2 <= n {
        let vx = vld1q_f64(x.as_ptr().add(i));
        vst1q_f64(x.as_mut_ptr().add(i), vmulq_f64(va, vx));
        i += 2;
    }
    while i < n {
        *x.get_unchecked_mut(i) *= alpha;
        i += 1;
    }
}
