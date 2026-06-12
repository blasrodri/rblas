//! x86-64 AVX2 + FMA Level-1 kernels (256-bit: 8×f32 / 4×f64).
//!
//! Reached only after `is_x86_feature_detected!("avx2")` (and `"fma"` for the
//! FMA paths) succeeds at runtime, so the `#[target_feature]` intrinsics are
//! safe to emit. `scal` needs only AVX2; `axpy`/`dot` additionally need FMA.

use core::arch::x86_64::*;

// ---------------------------------------------------------------- f32 (8-wide)

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let vy = _mm256_loadu_ps(y.as_ptr().add(i));
        let r = _mm256_fmadd_ps(va, vx, vy); // alpha*x + y
        _mm256_storeu_ps(y.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < n {
        *y.get_unchecked_mut(i) = alpha.mul_add(*x.get_unchecked(i), *y.get_unchecked(i));
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(x1, y1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);
        i += 8;
    }
    let mut acc = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < n {
        acc = x.get_unchecked(i).mul_add(*y.get_unchecked(i), acc);
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2")]
pub unsafe fn scal_f32(alpha: f32, x: &mut [f32]) {
    let n = x.len();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(va, vx));
        i += 8;
    }
    while i < n {
        *x.get_unchecked_mut(i) *= alpha;
        i += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

// ---------------------------------------------------------------- f64 (4-wide)

#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len();
    let va = _mm256_set1_pd(alpha);
    let mut i = 0;
    while i + 4 <= n {
        let vx = _mm256_loadu_pd(x.as_ptr().add(i));
        let vy = _mm256_loadu_pd(y.as_ptr().add(i));
        _mm256_storeu_pd(y.as_mut_ptr().add(i), _mm256_fmadd_pd(va, vx, vy));
        i += 4;
    }
    while i < n {
        *y.get_unchecked_mut(i) = alpha.mul_add(*x.get_unchecked(i), *y.get_unchecked(i));
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut i = 0;
    while i + 8 <= n {
        let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
        let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        let x1 = _mm256_loadu_pd(x.as_ptr().add(i + 4));
        let y1 = _mm256_loadu_pd(y.as_ptr().add(i + 4));
        acc1 = _mm256_fmadd_pd(x1, y1, acc1);
        i += 8;
    }
    while i + 4 <= n {
        let x0 = _mm256_loadu_pd(x.as_ptr().add(i));
        let y0 = _mm256_loadu_pd(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        i += 4;
    }
    let mut acc = hsum256_pd(_mm256_add_pd(acc0, acc1));
    while i < n {
        acc = x.get_unchecked(i).mul_add(*y.get_unchecked(i), acc);
        i += 1;
    }
    acc
}

#[target_feature(enable = "avx2")]
pub unsafe fn scal_f64(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let va = _mm256_set1_pd(alpha);
    let mut i = 0;
    while i + 4 <= n {
        let vx = _mm256_loadu_pd(x.as_ptr().add(i));
        _mm256_storeu_pd(x.as_mut_ptr().add(i), _mm256_mul_pd(va, vx));
        i += 4;
    }
    while i < n {
        *x.get_unchecked_mut(i) *= alpha;
        i += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_pd(v: __m256d) -> f64 {
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let s = _mm_add_pd(lo, hi);
    let s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    _mm_cvtsd_f64(s)
}
