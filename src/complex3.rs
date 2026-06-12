//! Complex Level-3 BLAS (`c`/`z`): matrix–matrix operations.
//!
//! [`gemm`] reuses the fast real packed kernel via the **4M decomposition**:
//! splitting each complex operand into real/imaginary planes and computing the
//! product with four real GEMMs
//!
//! ```text
//! P_re = Ar·Br − Ai·Bi
//! P_im = Ar·Bi + Ai·Br
//! ```
//!
//! then applying the complex `alpha`/`beta` while re-interleaving into `C`.
//! `symm`/`hemm`/`trsm` are direct reference implementations.

use alloc::vec;
use alloc::vec::Vec;

use crate::complex::Complex;
use crate::{level3, Diag, Float, Side, Transpose, Uplo};

/// General complex matrix–matrix multiply:
/// `C := alpha · op(A) · op(B) + beta · C` (column-major).
///
/// `op` applies `trans` per operand, including [`Transpose::ConjTrans`]
/// (conjugate transpose). Implemented via four real GEMMs (4M) on split planes.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Float>(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    b: &[Complex<T>],
    ldb: usize,
    beta: Complex<T>,
    c: &mut [Complex<T>],
    ldc: usize,
) {
    // Build contiguous column-major real/imag planes of op(A) (m×k) and op(B)
    // (k×n), folding in any transpose and conjugation.
    let (ar, ai) = split_op(transa, m, k, a, lda);
    let (br, bi) = split_op(transb, k, n, b, ldb);

    // Real GEMMs (column-major, leading dims = row counts m / k).
    // P_re = Ar·Br − Ai·Bi ; P_im = Ar·Bi + Ai·Br.
    let mut pr = vec![T::ZERO; m * n];
    let mut pi = vec![T::ZERO; m * n];
    // pr = Ar·Br
    rgemm(m, n, k, T::ONE, &ar, &br, T::ZERO, &mut pr);
    // pr += (−1)·Ai·Bi
    rgemm(m, n, k, T::ZERO.sub(T::ONE), &ai, &bi, T::ONE, &mut pr);
    // pi = Ar·Bi
    rgemm(m, n, k, T::ONE, &ar, &bi, T::ZERO, &mut pi);
    // pi += Ai·Br
    rgemm(m, n, k, T::ONE, &ai, &br, T::ONE, &mut pi);

    // C := alpha·P + beta·C, element-wise complex.
    for j in 0..n {
        for i in 0..m {
            let p = Complex::new(pr[i + j * m], pi[i + j * m]);
            let cij = i + j * ldc;
            let cur = c[cij];
            let scaled = if beta.is_zero() {
                Complex::zero()
            } else {
                beta * cur
            };
            c[cij] = alpha * p + scaled;
        }
    }
}

/// Real wrapper: column-major real GEMM `C := alpha·A·B + beta·C` with
/// leading dims equal to row counts (the planes are tightly packed).
#[inline]
#[allow(clippy::too_many_arguments)]
fn rgemm<T: Float>(m: usize, n: usize, k: usize, alpha: T, a: &[T], b: &[T], beta: T, c: &mut [T]) {
    level3::gemm(
        crate::Layout::ColMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        alpha,
        a,
        m,
        b,
        k,
        beta,
        c,
        m,
    );
}

/// Split `op(A)` (logical `rows × cols` after `trans`) into contiguous
/// column-major real and imaginary planes, applying transpose/conjugation.
fn split_op<T: Float>(
    trans: Transpose,
    rows: usize,
    cols: usize,
    a: &[Complex<T>],
    lda: usize,
) -> (Vec<T>, Vec<T>) {
    let mut re = vec![T::ZERO; rows * cols];
    let mut im = vec![T::ZERO; rows * cols];
    let conj = trans == Transpose::ConjTrans;
    for j in 0..cols {
        for i in 0..rows {
            // Source index in stored (pre-transpose) A.
            let v = if trans.is_transposed() {
                a[j + i * lda] // stored cols×rows: A[j, i]
            } else {
                a[i + j * lda]
            };
            let v = if conj { v.conj() } else { v };
            re[i + j * rows] = v.re;
            im[i + j * rows] = v.im;
        }
    }
    (re, im)
}

/// Complex symmetric matrix–matrix multiply (CSYMM/ZSYMM): `A` symmetric
/// (`A = Aᵀ`, **not** Hermitian), only `uplo` triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn symm<T: Float>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    b: &[Complex<T>],
    ldb: usize,
    beta: Complex<T>,
    c: &mut [Complex<T>],
    ldc: usize,
) {
    he_or_sy(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, false);
}

/// Complex Hermitian matrix–matrix multiply (CHEMM/ZHEMM): `A` Hermitian
/// (`A = Aᴴ`), only `uplo` triangle stored; the mirrored half is conjugated and
/// the diagonal's imaginary part is taken as zero.
#[allow(clippy::too_many_arguments)]
pub fn hemm<T: Float>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    b: &[Complex<T>],
    ldb: usize,
    beta: Complex<T>,
    c: &mut [Complex<T>],
    ldc: usize,
) {
    he_or_sy(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc, true);
}

#[allow(clippy::too_many_arguments)]
fn he_or_sy<T: Float>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    b: &[Complex<T>],
    ldb: usize,
    beta: Complex<T>,
    c: &mut [Complex<T>],
    ldc: usize,
    hermitian: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    // Materialize the full q×q symmetric/Hermitian A from its stored triangle
    // (Hermitian conjugates the mirrored half and takes a real diagonal), then
    // route through the 4M complex GEMM — reusing the fast packed real kernel.
    let q = if side == Side::Left { m } else { n };
    let mut full = vec![Complex::zero(); q * q];
    for j in 0..q {
        for i in 0..q {
            let stored_here = match uplo {
                Uplo::Upper => i <= j,
                Uplo::Lower => i >= j,
            };
            full[i + j * q] = if stored_here {
                let v = a[i + j * lda];
                if hermitian && i == j {
                    Complex::new(v.re, T::ZERO)
                } else {
                    v
                }
            } else {
                let v = a[j + i * lda];
                if hermitian {
                    v.conj()
                } else {
                    v
                }
            };
        }
    }
    match side {
        Side::Left => gemm(
            Transpose::None,
            Transpose::None,
            m,
            n,
            m,
            alpha,
            &full,
            q,
            b,
            ldb,
            beta,
            c,
            ldc,
        ),
        Side::Right => gemm(
            Transpose::None,
            Transpose::None,
            m,
            n,
            n,
            alpha,
            b,
            ldb,
            &full,
            q,
            beta,
            c,
            ldc,
        ),
    }
}

/// Complex triangular solve (CTRSM/ZTRSM): `op(A)·X = alpha·B` (Left) or
/// `X·op(A) = alpha·B` (Right), in place. `op` applies `trans` incl. conjugate.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: Float>(
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: Complex<T>,
    a: &[Complex<T>],
    lda: usize,
    b: &mut [Complex<T>],
    ldb: usize,
) {
    if alpha != Complex::one() {
        for j in 0..n {
            for i in 0..m {
                b[i + j * ldb] = alpha * b[i + j * ldb];
            }
        }
    }
    let conj = trans == Transpose::ConjTrans;
    let aop = |r: usize, c: usize| -> Complex<T> {
        let v = if trans.is_transposed() {
            a[c + r * lda]
        } else {
            a[r + c * lda]
        };
        if conj {
            v.conj()
        } else {
            v
        }
    };
    let eff_upper = match (uplo, trans.is_transposed()) {
        (Uplo::Upper, false) | (Uplo::Lower, true) => true,
        (Uplo::Lower, false) | (Uplo::Upper, true) => false,
    };
    let finish = |s: Complex<T>, a_ii: Complex<T>| match diag {
        Diag::Unit => s,
        Diag::NonUnit => cdiv(s, a_ii),
    };

    match side {
        Side::Left => {
            for j in 0..n {
                let col = j * ldb;
                if eff_upper {
                    for i in (0..m).rev() {
                        let mut s = b[i + col];
                        for kk in (i + 1)..m {
                            s = s - aop(i, kk) * b[kk + col];
                        }
                        b[i + col] = finish(s, aop(i, i));
                    }
                } else {
                    for i in 0..m {
                        let mut s = b[i + col];
                        for kk in 0..i {
                            s = s - aop(i, kk) * b[kk + col];
                        }
                        b[i + col] = finish(s, aop(i, i));
                    }
                }
            }
        }
        Side::Right => {
            let order: Vec<usize> = if eff_upper {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            };
            for i in 0..m {
                for &j in &order {
                    let mut s = b[i + j * ldb];
                    if eff_upper {
                        for kk in 0..j {
                            s = s - b[i + kk * ldb] * aop(kk, j);
                        }
                    } else {
                        for kk in (j + 1)..n {
                            s = s - b[i + kk * ldb] * aop(kk, j);
                        }
                    }
                    b[i + j * ldb] = finish(s, aop(j, j));
                }
            }
        }
    }
}

/// Complex division `a / b`.
#[inline(always)]
fn cdiv<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let d = b.norm_sqr();
    // a·conj(b) / |b|²
    let num = a * b.conj();
    Complex::new(num.re.div(d), num.im.div(d))
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    type C = Complex<f64>;

    /// Naive complex GEMM reference (col-major, no trans).
    fn naive(m: usize, n: usize, k: usize, a: &[C], b: &[C]) -> Vec<C> {
        let mut c = vec![C::zero(); m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = C::zero();
                for p in 0..k {
                    acc = acc + a[i + p * m] * b[p + j * k];
                }
                c[i + j * m] = acc;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_naive() {
        let (m, n, k) = (5, 4, 6);
        let a: Vec<C> = (0..m * k)
            .map(|i| C::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let b: Vec<C> = (0..k * n)
            .map(|i| C::new((i as f64 * 0.2).cos(), (i as f64 * 0.5).sin()))
            .collect();
        let mut c = vec![C::zero(); m * n];
        gemm(
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            C::one(),
            &a,
            m,
            &b,
            k,
            C::zero(),
            &mut c,
            m,
        );
        let want = naive(m, n, k, &a, &b);
        for (g, w) in c.iter().zip(&want) {
            assert!((g.re - w.re).abs() < 1e-9 && (g.im - w.im).abs() < 1e-9);
        }
    }

    #[test]
    fn gemm_conjtrans_a() {
        // C = Aᴴ·B. A stored k×m (so op(A) is m×k).
        let (m, n, k) = (3, 2, 4);
        let a: Vec<C> = (0..k * m)
            .map(|i| C::new(i as f64, (i as f64) * 0.5))
            .collect();
        let b: Vec<C> = (0..k * n).map(|i| C::new(1.0, i as f64)).collect();
        let mut c = vec![C::zero(); m * n];
        gemm(
            Transpose::ConjTrans,
            Transpose::None,
            m,
            n,
            k,
            C::one(),
            &a,
            k,
            &b,
            k,
            C::zero(),
            &mut c,
            m,
        );
        // Reference: conj(A[p,i]) * B[p,j].
        for j in 0..n {
            for i in 0..m {
                let mut acc = C::zero();
                for p in 0..k {
                    acc = acc + a[p + i * k].conj() * b[p + j * k];
                }
                assert!((c[i + j * m] - acc).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn trsm_left_lower_roundtrip() {
        // Solve L·X = B then check L·X == B.
        let m = 3usize;
        let l = vec![
            C::new(2.0, 1.0),
            C::new(1.0, 0.0),
            C::new(0.5, -1.0),
            C::zero(),
            C::new(3.0, -1.0),
            C::new(1.0, 1.0),
            C::zero(),
            C::zero(),
            C::new(4.0, 0.5),
        ];
        let b0 = vec![C::new(1.0, 1.0), C::new(2.0, 0.0), C::new(0.0, 3.0)];
        let mut b = b0.clone();
        trsm(
            Side::Left,
            Uplo::Lower,
            Transpose::None,
            Diag::NonUnit,
            m,
            1,
            C::one(),
            &l,
            m,
            &mut b,
            m,
        );
        let lat = |i: usize, j: usize| l[i + j * m];
        for i in 0..m {
            let mut s = C::zero();
            for j in 0..=i {
                s = s + lat(i, j) * b[j];
            }
            assert!((s - b0[i]).abs() < 1e-9, "row {i}");
        }
    }

    #[test]
    fn hemm_left_upper() {
        // Hermitian A (2×2), upper stored; off-diag mirrored as conjugate.
        // A = [[2, 1+i],[1-i, 3]]; store upper: a[0,0]=2, a[0,1]=1+i, a[1,1]=3.
        let a = vec![
            C::new(2.0, 0.0),
            C::new(9.0, 9.0), // col0 (lower entry unused)
            C::new(1.0, 1.0),
            C::new(3.0, 0.0), // col1
        ];
        let b = vec![C::new(1.0, 0.0), C::new(0.0, 1.0)]; // 2×1
        let mut c = vec![C::zero(); 2];
        hemm(
            Side::Left,
            Uplo::Upper,
            2,
            1,
            C::one(),
            &a,
            2,
            &b,
            2,
            C::zero(),
            &mut c,
            2,
        );
        // c0 = 2*(1) + (1+i)*(i) = 2 + (i + i²) = 2 + (i -1) = 1 + i
        // c1 = conj(1+i)*(1) + 3*(i) = (1-i) + 3i = 1 + 2i
        assert!((c[0] - C::new(1.0, 1.0)).abs() < 1e-12);
        assert!((c[1] - C::new(1.0, 2.0)).abs() < 1e-12);
    }

    #[test]
    fn hemm_left_lower_larger() {
        // Hermitian A (5×5) lower-stored, B (5×3). Verify vs explicit mirror.
        let q = 5usize;
        let n = 3usize;
        let a: Vec<C> = (0..q * q)
            .map(|i| C::new((i as f64 * 0.3).sin(), (i as f64 * 0.4).cos()))
            .collect();
        let b: Vec<C> = (0..q * n)
            .map(|i| C::new((i as f64) * 0.1, (i as f64) * 0.05))
            .collect();
        let mut c = vec![C::zero(); q * n];
        hemm(
            Side::Left,
            Uplo::Lower,
            q,
            n,
            C::new(1.5, -0.5),
            &a,
            q,
            &b,
            q,
            C::zero(),
            &mut c,
            q,
        );
        // Reference Hermitian access from lower triangle.
        let her = |i: usize, j: usize| -> C {
            if i > j {
                a[i + j * q]
            } else if i < j {
                a[j + i * q].conj()
            } else {
                C::new(a[i + j * q].re, 0.0)
            }
        };
        let alpha = C::new(1.5, -0.5);
        for j in 0..n {
            for i in 0..q {
                let mut acc = C::zero();
                for p in 0..q {
                    acc = acc + her(i, p) * b[p + j * q];
                }
                let want = alpha * acc;
                assert!((c[i + j * q] - want).abs() < 1e-9, "({i},{j})");
            }
        }
    }
}
