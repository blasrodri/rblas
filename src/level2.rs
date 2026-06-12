//! Level-2 BLAS: matrix–vector operations.

use crate::{Float, Layout, Transpose};

/// General matrix–vector multiply: `y := alpha * op(A) * x + beta * y`.
///
/// `A` is stored `m × n` in `layout` with leading dimension `lda`. When
/// `trans` is a transpose, the operation uses `Aᵀ`, so `x` has length `m` and
/// `y` has length `n`; otherwise `x` has length `n` and `y` has length `m`.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: Float>(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    x: &[T],
    incx: usize,
    beta: T,
    y: &mut [T],
    incy: usize,
) {
    let (rs, cs) = layout.strides(lda);
    // (len_y, len_x): output rows, input cols after the transpose.
    let (leny, lenx) = if trans.is_transposed() {
        (n, m)
    } else {
        (m, n)
    };

    // y := beta * y
    scale_y(leny, beta, y, incy);
    if alpha == T::ZERO {
        return;
    }

    if !trans.is_transposed() {
        // y_i += alpha * sum_j A[i,j] x_j
        for i in 0..m {
            let mut acc = T::ZERO;
            let mut jx = 0;
            for j in 0..n {
                acc = a[i * rs + j * cs].mul_add(x[jx], acc);
                jx += incx;
            }
            y[i * incy] = alpha.mul_add(acc, y[i * incy]);
        }
    } else {
        // y_j += alpha * sum_i A[i,j] x_i   (= (Aᵀx)_j)
        let _ = lenx;
        for i in 0..m {
            let xi = alpha.mul(x[i * incx]);
            for j in 0..n {
                let yj = j * incy;
                y[yj] = a[i * rs + j * cs].mul_add(xi, y[yj]);
            }
        }
    }
}

/// Rank-1 update: `A := alpha * x * yᵀ + A`.
#[allow(clippy::too_many_arguments)]
pub fn ger<T: Float>(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: T,
    x: &[T],
    incx: usize,
    y: &[T],
    incy: usize,
    a: &mut [T],
    lda: usize,
) {
    if alpha == T::ZERO {
        return;
    }
    let (rs, cs) = layout.strides(lda);
    for i in 0..m {
        let axi = alpha.mul(x[i * incx]);
        for j in 0..n {
            let idx = i * rs + j * cs;
            a[idx] = axi.mul_add(y[j * incy], a[idx]);
        }
    }
}

#[inline]
fn scale_y<T: Float>(len: usize, beta: T, y: &mut [T], incy: usize) {
    if beta == T::ONE {
        return;
    }
    let mut iy = 0;
    for _ in 0..len {
        y[iy] = if beta == T::ZERO {
            T::ZERO
        } else {
            beta.mul(y[iy])
        };
        iy += incy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_notrans_colmajor() {
        // A = [[1,2,3],[4,5,6]] stored col-major (m=2,n=3): cols [1,4],[2,5],[3,6]
        let a = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = [1.0f64, 1.0, 1.0];
        let mut y = [0.0f64, 0.0];
        gemv(
            Layout::ColMajor,
            Transpose::None,
            2,
            3,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, [6.0, 15.0]); // row sums
    }

    #[test]
    fn gemv_trans_colmajor() {
        let a = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 col-major
        let x = [1.0f64, 1.0];
        let mut y = [0.0f64; 3];
        gemv(
            Layout::ColMajor,
            Transpose::Trans,
            2,
            3,
            1.0,
            &a,
            2,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, [5.0, 7.0, 9.0]); // column sums
    }

    #[test]
    fn gemv_rowmajor() {
        // Same logical A row-major (m=2,n=3): [1,2,3,4,5,6], lda=3
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f64, 0.0, 1.0];
        let mut y = [0.0f64, 0.0];
        gemv(
            Layout::RowMajor,
            Transpose::None,
            2,
            3,
            1.0,
            &a,
            3,
            &x,
            1,
            0.0,
            &mut y,
            1,
        );
        assert_eq!(y, [4.0, 10.0]); // 1+3, 4+6
    }

    #[test]
    fn ger_basic() {
        // x*yᵀ for x=[1,2], y=[3,4]; col-major 2x2
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];
        let mut a = [0.0f64; 4];
        ger(Layout::ColMajor, 2, 2, 1.0, &x, 1, &y, 1, &mut a, 2);
        // A[0,0]=3 A[1,0]=6 A[0,1]=4 A[1,1]=8 -> col-major [3,6,4,8]
        assert_eq!(a, [3.0, 6.0, 4.0, 8.0]);
    }
}
