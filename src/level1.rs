//! Level-1 BLAS: vector–vector operations.
//!
//! Each routine accepts an explicit `incx`/`incy` stride (the BLAS convention)
//! so non-contiguous slices — e.g. a matrix column in row-major storage — can be
//! operated on without copying. A stride of `1` hits the fast contiguous path,
//! which dispatches to SIMD kernels where available.

use crate::kernel;
use crate::Float;

/// `y := alpha * x + y` (AXPY).
///
/// # Panics
/// Panics if the index ranges implied by `n`, `incx`, `incy` would read or write
/// out of bounds of `x` / `y`.
#[inline]
pub fn axpy<T: Float>(n: usize, alpha: T, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if n == 0 || alpha == T::ZERO {
        return;
    }
    if incx == 1 && incy == 1 {
        assert!(x.len() >= n && y.len() >= n, "axpy: slice shorter than n");
        kernel::axpy_contig(alpha, &x[..n], &mut y[..n]);
    } else {
        strided_axpy(n, alpha, x, incx, y, incy);
    }
}

#[inline]
fn strided_axpy<T: Float>(n: usize, alpha: T, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    assert!(incx > 0 && incy > 0, "axpy: stride must be positive");
    assert!((n - 1) * incx < x.len(), "axpy: x out of bounds");
    assert!((n - 1) * incy < y.len(), "axpy: y out of bounds");
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = alpha.mul_add(x[ix], y[iy]);
        ix += incx;
        iy += incy;
    }
}

/// Dot product `xᵀ · y`.
#[inline]
pub fn dot<T: Float>(n: usize, x: &[T], incx: usize, y: &[T], incy: usize) -> T {
    if n == 0 {
        return T::ZERO;
    }
    if incx == 1 && incy == 1 {
        assert!(x.len() >= n && y.len() >= n, "dot: slice shorter than n");
        kernel::dot_contig(&x[..n], &y[..n])
    } else {
        assert!(incx > 0 && incy > 0, "dot: stride must be positive");
        assert!((n - 1) * incx < x.len(), "dot: x out of bounds");
        assert!((n - 1) * incy < y.len(), "dot: y out of bounds");
        let mut acc = T::ZERO;
        let mut ix = 0;
        let mut iy = 0;
        for _ in 0..n {
            acc = x[ix].mul_add(y[iy], acc);
            ix += incx;
            iy += incy;
        }
        acc
    }
}

/// `x := alpha * x` (SCAL).
#[inline]
pub fn scal<T: Float>(n: usize, alpha: T, x: &mut [T], incx: usize) {
    if n == 0 || alpha == T::ONE {
        return;
    }
    if incx == 1 {
        assert!(x.len() >= n, "scal: slice shorter than n");
        kernel::scal_contig(alpha, &mut x[..n]);
    } else {
        assert!(incx > 0, "scal: stride must be positive");
        assert!((n - 1) * incx < x.len(), "scal: x out of bounds");
        let mut ix = 0;
        for _ in 0..n {
            x[ix] = alpha.mul(x[ix]);
            ix += incx;
        }
    }
}

/// Euclidean norm `‖x‖₂ = sqrt(Σ xᵢ²)`.
///
/// Uses the scaled sum-of-squares algorithm (as in reference LAPACK `*nrm2`):
/// a running `scale = max|xᵢ|` and `ssq` accumulated relative to it, so the
/// result is `scale·√ssq`. This avoids spurious overflow when some `|xᵢ|` is
/// near `sqrt(MAX)` and spurious underflow for tiny elements — cases where the
/// naive `sqrt(Σ xᵢ²)` returns `inf`/`0`.
#[inline]
pub fn nrm2<T: Float>(n: usize, x: &[T], incx: usize) -> T {
    if n == 0 {
        return T::ZERO;
    }
    assert!(incx > 0, "nrm2: stride must be positive");
    assert!((n - 1) * incx < x.len(), "nrm2: x out of bounds");

    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    let mut ix = 0;
    for _ in 0..n {
        let xi = x[ix];
        if xi != T::ZERO {
            let a = xi.abs();
            if scale.lt(a) {
                // New max: rescale the accumulated ssq down to the larger scale.
                let r = scale.div(a);
                ssq = T::ONE.add(ssq.mul(r.mul(r)));
                scale = a;
            } else {
                let r = a.div(scale);
                ssq = ssq.add(r.mul(r));
            }
        }
        ix += incx;
    }
    scale.mul(ssq.sqrt())
}

/// Sum of absolute values `Σ |xᵢ|` (ASUM).
#[inline]
pub fn asum<T: Float>(n: usize, x: &[T], incx: usize) -> T {
    let mut acc = T::ZERO;
    let mut ix = 0;
    for _ in 0..n {
        acc = acc.add(x[ix].abs());
        ix += incx;
    }
    acc
}

/// `y := x` (COPY).
#[inline]
pub fn copy<T: Float>(n: usize, x: &[T], incx: usize, y: &mut [T], incy: usize) {
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = x[ix];
        ix += incx;
        iy += incy;
    }
}

/// Swap the contents of `x` and `y` (SWAP).
#[inline]
pub fn swap<T: Float>(n: usize, x: &mut [T], incx: usize, y: &mut [T], incy: usize) {
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        core::mem::swap(&mut x[ix], &mut y[iy]);
        ix += incx;
        iy += incy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_contiguous() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [10.0f32, 20.0, 30.0, 40.0];
        axpy(4, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, [12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn axpy_strided() {
        let x = [1.0f64, 0.0, 2.0, 0.0, 3.0];
        let mut y = [10.0f64, 20.0, 30.0];
        axpy(3, 1.0, &x, 2, &mut y, 1);
        assert_eq!(y, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn dot_basic() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        assert_eq!(dot(3, &x, 1, &y, 1), 32.0);
    }

    #[test]
    fn scal_basic() {
        let mut x = [1.0f64, 2.0, 3.0];
        scal(3, 10.0, &mut x, 1);
        assert_eq!(x, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn nrm2_basic() {
        let x = [3.0f32, 4.0];
        assert!((nrm2(2, &x, 1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn nrm2_no_overflow() {
        // Each element near sqrt(f64::MAX): the naive sqrt(Σ xᵢ²) overflows to
        // inf, but the scaled algorithm gives the true norm sqrt(3)·big.
        let big = 1.0e200f64;
        let x = [big, big, big];
        let got = nrm2(3, &x, 1);
        assert!(got.is_finite(), "nrm2 overflowed to {got}");
        let want = big * 3.0f64.sqrt();
        assert!((got - want).abs() / want < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn nrm2_matches_naive_for_moderate() {
        let x = [1.0f64, -2.0, 3.0, -4.0, 5.0];
        let naive = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((nrm2(5, &x, 1) - naive).abs() < 1e-12);
    }

    #[test]
    fn nrm2_strided() {
        let x = [3.0f64, 99.0, 4.0]; // stride 2 picks 3.0 and 4.0
        assert!((nrm2(2, &x, 2) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn asum_basic() {
        let x = [1.0f64, -2.0, 3.0, -4.0];
        assert_eq!(asum(4, &x, 1), 10.0);
    }
}
