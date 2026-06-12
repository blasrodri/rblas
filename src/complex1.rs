//! Complex Level-1 BLAS (`c`/`z`): vector–vector operations on [`Complex<T>`].

use crate::complex::Complex;
use crate::Float;

/// `y := alpha · x + y` (CAXPY/ZAXPY).
#[inline]
pub fn axpy<T: Float>(
    n: usize,
    alpha: Complex<T>,
    x: &[Complex<T>],
    incx: usize,
    y: &mut [Complex<T>],
    incy: usize,
) {
    if n == 0 || alpha.is_zero() {
        return;
    }
    assert!(incx > 0 && incy > 0, "axpy: stride must be positive");
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        y[iy] = alpha * x[ix] + y[iy];
        ix += incx;
        iy += incy;
    }
}

/// Unconjugated dot product `Σ xᵢ · yᵢ` (CDOTU/ZDOTU).
#[inline]
pub fn dotu<T: Float>(
    n: usize,
    x: &[Complex<T>],
    incx: usize,
    y: &[Complex<T>],
    incy: usize,
) -> Complex<T> {
    let mut acc = Complex::zero();
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        acc = acc + x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    acc
}

/// Conjugated dot product `Σ conj(xᵢ) · yᵢ` (CDOTC/ZDOTC).
#[inline]
pub fn dotc<T: Float>(
    n: usize,
    x: &[Complex<T>],
    incx: usize,
    y: &[Complex<T>],
    incy: usize,
) -> Complex<T> {
    let mut acc = Complex::zero();
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        acc = acc + x[ix].conj() * y[iy];
        ix += incx;
        iy += incy;
    }
    acc
}

/// `x := alpha · x` (CSCAL/ZSCAL).
#[inline]
pub fn scal<T: Float>(n: usize, alpha: Complex<T>, x: &mut [Complex<T>], incx: usize) {
    if n == 0 {
        return;
    }
    assert!(incx > 0, "scal: stride must be positive");
    let mut ix = 0;
    for _ in 0..n {
        x[ix] = alpha * x[ix];
        ix += incx;
    }
}

/// Euclidean norm `‖x‖₂` over complex elements (SCNRM2/DZNRM2), scaled to avoid
/// overflow (same algorithm as the real [`crate::level1::nrm2`]).
#[inline]
pub fn nrm2<T: Float>(n: usize, x: &[Complex<T>], incx: usize) -> T {
    if n == 0 {
        return T::ZERO;
    }
    assert!(incx > 0, "nrm2: stride must be positive");
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    let mut ix = 0;
    // Treat each complex element as its two real components.
    for _ in 0..n {
        for comp in [x[ix].re, x[ix].im] {
            if comp != T::ZERO {
                let a = comp.abs();
                if scale.lt(a) {
                    let r = scale.div(a);
                    ssq = T::ONE.add(ssq.mul(r.mul(r)));
                    scale = a;
                } else {
                    let r = a.div(scale);
                    ssq = ssq.add(r.mul(r));
                }
            }
        }
        ix += incx;
    }
    scale.mul(ssq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    type C = Complex<f64>;

    #[test]
    fn axpy_and_scal() {
        let x = [C::new(1.0, 1.0), C::new(2.0, -1.0)];
        let mut y = [C::new(0.0, 0.0), C::new(1.0, 1.0)];
        axpy(2, C::new(2.0, 0.0), &x, 1, &mut y, 1);
        assert_eq!(y, [C::new(2.0, 2.0), C::new(5.0, -1.0)]);

        let mut z = [C::new(1.0, 0.0)];
        scal(1, C::new(0.0, 1.0), &mut z, 1); // multiply by i
        assert_eq!(z, [C::new(0.0, 1.0)]);
    }

    #[test]
    fn dotu_vs_dotc() {
        let x = [C::new(1.0, 2.0)];
        let y = [C::new(3.0, 4.0)];
        // dotu = (1+2i)(3+4i) = -5 + 10i
        assert_eq!(dotu(1, &x, 1, &y, 1), C::new(-5.0, 10.0));
        // dotc = conj(1+2i)(3+4i) = (1-2i)(3+4i) = 11 - 2i
        assert_eq!(dotc(1, &x, 1, &y, 1), C::new(11.0, -2.0));
    }

    #[test]
    fn nrm2_basic() {
        let x = [C::new(3.0, 4.0)]; // |3+4i| = 5
        assert!((nrm2(1, &x, 1) - 5.0).abs() < 1e-12);
    }
}
