//! Complex Level-2 BLAS (`c`/`z`): matrix–vector operations.

use crate::complex::Complex;
use crate::{Float, Layout, Transpose};

/// General complex matrix–vector multiply: `y := alpha · op(A) · x + beta · y`.
///
/// `op(A)` applies `trans`: none, transpose, or **conjugate** transpose
/// ([`Transpose::ConjTrans`], which conjugates each referenced `A` entry).
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: Float>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    x: &[Complex<T>],
    incx: usize,
    beta: Complex<T>,
    y: &mut [Complex<T>],
    incy: usize,
) {
    let (rs, cs) = layout.strides(lda);
    let conj = trans == Transpose::ConjTrans;
    let leny = if trans.is_transposed() { n } else { m };

    // y := beta·y
    if !beta.is_zero() || beta != Complex::one() {
        let mut iy = 0;
        for _ in 0..leny {
            y[iy] = if beta.is_zero() {
                Complex::zero()
            } else {
                beta * y[iy]
            };
            iy += incy;
        }
    }
    if alpha.is_zero() {
        return;
    }

    let aval = |i: usize, j: usize| -> Complex<T> {
        let v = a[i * rs + j * cs];
        if conj {
            v.conj()
        } else {
            v
        }
    };

    if !trans.is_transposed() {
        for i in 0..m {
            let mut acc = Complex::zero();
            let mut jx = 0;
            for j in 0..n {
                acc = acc + aval(i, j) * x[jx];
                jx += incx;
            }
            y[i * incy] = y[i * incy] + alpha * acc;
        }
    } else {
        // (op(A)x)_j = sum_i A_eff[i,j] x_i, where A_eff applies the conjugate.
        for j in 0..n {
            let mut acc = Complex::zero();
            let mut ix = 0;
            for i in 0..m {
                acc = acc + aval(i, j) * x[ix];
                ix += incx;
            }
            y[j * incy] = y[j * incy] + alpha * acc;
        }
    }
}

/// Rank-1 update, **unconjugated**: `A := alpha · x · yᵀ + A` (CGERU/ZGERU).
#[allow(clippy::too_many_arguments)]
pub fn geru<T: Float>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    x: &[Complex<T>],
    incx: usize,
    y: &[Complex<T>],
    incy: usize,
    a: &mut [Complex<T>],
    lda: usize,
) {
    rank1(layout, m, n, alpha, x, incx, y, incy, a, lda, false);
}

/// Rank-1 update, **conjugated**: `A := alpha · x · yᴴ + A` (CGERC/ZGERC).
#[allow(clippy::too_many_arguments)]
pub fn gerc<T: Float>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    x: &[Complex<T>],
    incx: usize,
    y: &[Complex<T>],
    incy: usize,
    a: &mut [Complex<T>],
    lda: usize,
) {
    rank1(layout, m, n, alpha, x, incx, y, incy, a, lda, true);
}

#[allow(clippy::too_many_arguments)]
fn rank1<T: Float>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    x: &[Complex<T>],
    incx: usize,
    y: &[Complex<T>],
    incy: usize,
    a: &mut [Complex<T>],
    lda: usize,
    conj: bool,
) {
    if alpha.is_zero() {
        return;
    }
    let (rs, cs) = layout.strides(lda);
    for i in 0..m {
        let axi = alpha * x[i * incx];
        for j in 0..n {
            let yj = if conj {
                y[j * incy].conj()
            } else {
                y[j * incy]
            };
            let idx = i * rs + j * cs;
            a[idx] = a[idx] + axi * yj;
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    type C = Complex<f64>;

    #[test]
    fn gemv_notrans() {
        // A = [[1+0i, 0+1i]] (1×2 col-major: a[0]=1, a[1*lda]=i), x=(2,3i? )
        let a = [C::new(1.0, 0.0), C::new(0.0, 1.0)]; // 1x2, lda=1
        let x = [C::new(1.0, 0.0), C::new(1.0, 0.0)];
        let mut y = [C::new(0.0, 0.0)];
        gemv(
            Layout::ColMajor,
            Transpose::None,
            1,
            2,
            C::one(),
            &a,
            1,
            &x,
            1,
            C::zero(),
            &mut y,
            1,
        );
        // y0 = 1*1 + i*1 = 1 + i
        assert_eq!(y, [C::new(1.0, 1.0)]);
    }

    #[test]
    fn gerc_conjugates_y() {
        let x = [C::new(1.0, 0.0)];
        let y = [C::new(0.0, 1.0)]; // conj -> -i
        let mut a = [C::new(0.0, 0.0)];
        gerc(Layout::ColMajor, 1, 1, C::one(), &x, 1, &y, 1, &mut a, 1);
        // A += x * conj(y) = 1 * (-i) = -i
        assert_eq!(a, [C::new(0.0, -1.0)]);
    }
}
