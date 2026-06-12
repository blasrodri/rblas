//! Level-3 BLAS: matrix–matrix operations.
//!
//! The flagship routine is [`gemm`]. It accepts an arbitrary [`Layout`] and
//! [`Transpose`] for each operand and reduces them to the single column-major,
//! no-transpose shape the packed kernel ([`crate::kernel::gemm_contig`])
//! expects — by reinterpreting strides rather than physically transposing.

use crate::kernel;
use crate::{Diag, Float, Layout, Side, Transpose, Uplo};

/// Number of worker threads the parallel GEMM path will use.
///
/// Available only with the `threads` feature; reflects the active rayon pool.
#[cfg(feature = "threads")]
pub fn num_threads() -> usize {
    rayon::current_num_threads()
}

/// General matrix–matrix multiply: `C := alpha * op(A) * op(B) + beta * C`.
///
/// `op(A)` is `A` (or `Aᵀ` when `transa` is a transpose). Logical dimensions
/// after the transpose are: `op(A)` is `m × k`, `op(B)` is `k × n`, `C` is
/// `m × n`.
///
/// - `lda`/`ldb`/`ldc` are the leading dimensions in the matrices' *stored*
///   layout (before transposition).
///
/// # Panics
/// Panics if any operand slice is too small for its stated dimensions/leading
/// dimension.
#[allow(clippy::too_many_arguments)]
pub fn gemm<T: Float>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // Normalize everything to column-major by swapping the row/col problem when
    // the caller is row-major: a row-major C(m×n) is a column-major Cᵀ(n×m), and
    // (A·B)ᵀ = Bᵀ·Aᵀ. So row-major gemm(A,B) == col-major gemm(opb(Bᵀ)? ...).
    // We implement this by flipping to a col-major call with A/B swapped.
    match layout {
        Layout::ColMajor => {
            gemm_colmajor(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        Layout::RowMajor => {
            // Cᵀ = (alpha·op(A)op(B) + beta·C)ᵀ = alpha·op(B)ᵀ op(A)ᵀ + beta·Cᵀ.
            // In col-major terms with dimensions (n,m,k): A'←B, B'←A, swap trans.
            gemm_colmajor(transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc)
        }
    }
}

/// Checked [`gemm`]: validates dimensions and buffer sizes up front, returning
/// [`BlasError`] instead of panicking. Use when the dimensions come from
/// untrusted input. On success the result is identical to [`gemm`].
#[allow(clippy::too_many_arguments)]
pub fn try_gemm<T: Float>(
    layout: Layout,
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) -> crate::Result<()> {
    // op(A) is m×k, op(B) is k×n, C is m×n. The *stored* shape swaps rows/cols
    // when transposed; the leading dim and buffer length then follow the layout.
    let (a_rows, a_cols) = if transa.is_transposed() {
        (k, m)
    } else {
        (m, k)
    };
    let (b_rows, b_cols) = if transb.is_transposed() {
        (n, k)
    } else {
        (k, n)
    };
    check_matrix(layout, "a", a_rows, a_cols, lda, a.len())?;
    check_matrix(layout, "b", b_rows, b_cols, ldb, b.len())?;
    check_matrix(layout, "c", m, n, ldc, c.len())?;
    gemm(
        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
    Ok(())
}

/// Validate one stored `rows × cols` matrix against its `lead`ing dimension and
/// backing slice `len`, per `layout`.
fn check_matrix(
    layout: Layout,
    which: &'static str,
    rows: usize,
    cols: usize,
    lead: usize,
    len: usize,
) -> crate::Result<()> {
    use crate::BlasError;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    // Minimum leading dim: the contiguous extent (RowMajor → cols, ColMajor →
    // rows), and the slice must span the last element of the last line.
    let (min_lead, need) = match layout {
        Layout::RowMajor => (cols, (rows - 1) * lead + cols),
        Layout::ColMajor => (rows, (cols - 1) * lead + rows),
    };
    if lead < min_lead {
        return Err(BlasError::InvalidLeadingDim {
            which,
            got: lead,
            min: min_lead,
        });
    }
    if len < need {
        return Err(BlasError::BufferTooSmall {
            which,
            got: len,
            need,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_colmajor<T: Float>(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // The packed kernel currently consumes no-transpose column-major panels.
    // For transposed inputs we fall back to a strided gather during packing via
    // a thin adapter: materialize op(A)/op(B) only when transposed. For the
    // common NN case this is zero-copy.
    match (transa.is_transposed(), transb.is_transposed()) {
        (false, false) => {
            assert!(a.len() >= (k - 1) * lda + m || k == 0, "gemm: A too small");
            assert!(b.len() >= (n - 1) * ldb + k || n == 0, "gemm: B too small");
            assert!(c.len() >= (n - 1) * ldc + m || n == 0, "gemm: C too small");
            kernel::gemm_contig(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        _ => {
            // Transpose path: build contiguous column-major op(A), op(B).
            let (am, bm) = materialize(transa, transb, m, n, k, a, lda, b, ldb);
            kernel::gemm_contig(m, n, k, alpha, &am, m, &bm, k, beta, c, ldc);
        }
    }
}

/// Produce contiguous column-major `op(A)` (`m×k`) and `op(B)` (`k×n`).
#[allow(clippy::too_many_arguments)]
fn materialize<T: Float>(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
) -> (alloc::vec::Vec<T>, alloc::vec::Vec<T>) {
    use alloc::vec;
    let mut am = vec![T::ZERO; m * k];
    let mut bm = vec![T::ZERO; k * n];
    if transa.is_transposed() {
        // Stored A is k×m col-major; op(A)=Aᵀ is m×k. a[(p) + i*lda] -> am[i + p*m].
        for p in 0..k {
            for i in 0..m {
                am[i + p * m] = a[p + i * lda];
            }
        }
    } else {
        for p in 0..k {
            for i in 0..m {
                am[i + p * m] = a[i + p * lda];
            }
        }
    }
    if transb.is_transposed() {
        // Stored B is n×k col-major; op(B)=Bᵀ is k×n. b[(j) + p*ldb] -> bm[p + j*k].
        for j in 0..n {
            for p in 0..k {
                bm[p + j * k] = b[j + p * ldb];
            }
        }
    } else {
        for j in 0..n {
            for p in 0..k {
                bm[p + j * k] = b[p + j * ldb];
            }
        }
    }
    (am, bm)
}

// ============================================================================
// syrk, symm, trsm — column-major. These reference implementations are correct
// for all uplo/trans/side/diag combinations; the compute-bound inner products
// reuse the SIMD dot kernel where the access pattern is contiguous.
// ============================================================================

/// Symmetric rank-`k` update: `C := alpha · op(A) · op(A)ᵀ + beta · C`,
/// updating only the `uplo` triangle of the symmetric `n × n` matrix `C`.
///
/// `op(A)` is `n × k`: `A` itself when `trans` is [`Transpose::None`] (stored
/// `n × k`), or `Aᵀ` otherwise (stored `k × n`). Column-major.
#[allow(clippy::too_many_arguments)]
pub fn syrk<T: Float>(
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    if n == 0 {
        return;
    }
    // Compute the full product P = alpha·op(A)·op(A)ᵀ via the packed GEMM
    // kernel, then fold only the `uplo` triangle into C with beta. op(A) is
    // n×k; the second factor is its transpose, so the GEMM transpose flags are
    // (trans, opposite(trans)).
    let (ta, tb) = if trans.is_transposed() {
        // op(A)=Aᵀ (stored k×n). C = Aᵀ·A ⇒ first Trans, second None.
        (Transpose::Trans, Transpose::None)
    } else {
        // op(A)=A (stored n×k). C = A·Aᵀ ⇒ first None, second Trans.
        (Transpose::None, Transpose::Trans)
    };
    let mut p = alloc::vec![T::ZERO; n * n];
    gemm(
        Layout::ColMajor,
        ta,
        tb,
        n,
        n,
        k,
        alpha,
        a,
        lda,
        a,
        lda,
        T::ZERO,
        &mut p,
        n,
    );
    for j in 0..n {
        let (i_lo, i_hi) = match uplo {
            Uplo::Upper => (0, j + 1),
            Uplo::Lower => (j, n),
        };
        for i in i_lo..i_hi {
            let cij = i + j * ldc;
            let scaled = if beta == T::ZERO {
                T::ZERO
            } else {
                beta.mul(c[cij])
            };
            c[cij] = p[i + j * n].add(scaled);
        }
    }
}

/// Symmetric matrix–matrix multiply: `C := alpha · A · B + beta · C` (side
/// [`Side::Left`]) or `C := alpha · B · A + beta · C` ([`Side::Right`]), where
/// `A` is symmetric with only its `uplo` triangle stored. `C`/`B` are `m × n`,
/// column-major.
#[allow(clippy::too_many_arguments)]
pub fn symm<T: Float>(
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: T,
    c: &mut [T],
    ldc: usize,
) {
    // A is q×q (q = m for Left, n for Right). Materialize the full symmetric
    // matrix from its stored `uplo` triangle, then route through the packed
    // GEMM kernel — far faster than the triple loop for non-trivial sizes.
    let q = if side == Side::Left { m } else { n };
    let full = full_symmetric(uplo, q, a, lda, false);
    match side {
        Side::Left => gemm(
            Layout::ColMajor,
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
            Layout::ColMajor,
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

/// Expand a `q × q` matrix stored only in its `uplo` triangle into a full dense
/// column-major buffer. With `conj`, the mirrored half is conjugated (Hermitian)
/// — for the real path `conj` is `false` and it's a plain symmetric mirror.
fn full_symmetric<T: Float>(
    uplo: Uplo,
    q: usize,
    a: &[T],
    lda: usize,
    _conj: bool,
) -> alloc::vec::Vec<T> {
    let mut full = alloc::vec![T::ZERO; q * q];
    for j in 0..q {
        for i in 0..q {
            let stored_here = match uplo {
                Uplo::Upper => i <= j,
                Uplo::Lower => i >= j,
            };
            full[i + j * q] = if stored_here {
                a[i + j * lda]
            } else {
                a[j + i * lda]
            };
        }
    }
    full
}

/// Triangular solve with multiple right-hand sides:
/// `op(A) · X = alpha · B` ([`Side::Left`]) or `X · op(A) = alpha · B`
/// ([`Side::Right`]), solved in place (`B` is overwritten by `X`).
///
/// `A` is triangular (`uplo`, optionally [`Diag::Unit`]); `op(A)` applies
/// `trans`. `B`/`X` are `m × n`, column-major. Solved by substitution.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: Float>(
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &mut [T],
    ldb: usize,
) {
    if alpha != T::ONE {
        for j in 0..n {
            for i in 0..m {
                b[i + j * ldb] = alpha.mul(b[i + j * ldb]);
            }
        }
    }
    // op(A)[r, c]: the effective triangular entry after applying `trans`.
    let aop = |r: usize, c: usize| -> T {
        if trans.is_transposed() {
            a[c + r * lda]
        } else {
            a[r + c * lda]
        }
    };
    // Effective upper/lower after transpose.
    let eff_upper = match (uplo, trans.is_transposed()) {
        (Uplo::Upper, false) | (Uplo::Lower, true) => true,
        (Uplo::Lower, false) | (Uplo::Upper, true) => false,
    };

    match side {
        Side::Left => {
            // Solve op(A) (m×m) · X = B, one B column at a time.
            for j in 0..n {
                solve_left(eff_upper, diag, m, &aop, b, ldb, j);
            }
        }
        Side::Right => {
            // Solve X · op(A) (n×n) = B. Equivalent to row-wise solves; do it
            // by columns of op(A): forward/back over j.
            solve_right(eff_upper, diag, m, n, &aop, b, ldb);
        }
    }
}

/// Solve `op(A)·x = b_col` in place for one column (Left side).
fn solve_left<T: Float>(
    upper: bool,
    diag: Diag,
    m: usize,
    aop: &impl Fn(usize, usize) -> T,
    b: &mut [T],
    ldb: usize,
    j: usize,
) {
    let col = j * ldb;
    if upper {
        // Back substitution: i from m-1 down to 0.
        for i in (0..m).rev() {
            let mut s = b[i + col];
            for kk in (i + 1)..m {
                s = neg_mul_add(aop(i, kk), b[kk + col], s);
            }
            b[i + col] = finish(diag, s, aop(i, i));
        }
    } else {
        // Forward substitution: i from 0 up.
        for i in 0..m {
            let mut s = b[i + col];
            for kk in 0..i {
                s = neg_mul_add(aop(i, kk), b[kk + col], s);
            }
            b[i + col] = finish(diag, s, aop(i, i));
        }
    }
}

/// Solve `X · op(A) = B` in place (Right side), `op(A)` is `n×n`.
#[allow(clippy::too_many_arguments)]
fn solve_right<T: Float>(
    upper: bool,
    diag: Diag,
    m: usize,
    n: usize,
    aop: &impl Fn(usize, usize) -> T,
    b: &mut [T],
    ldb: usize,
) {
    // For each row of X independently: x_row · op(A) = b_row.
    // Column order: if op(A) upper, solve j ascending; else descending.
    let order: alloc::vec::Vec<usize> = if upper {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };
    for i in 0..m {
        for &j in &order {
            let mut s = b[i + j * ldb];
            // Subtract already-solved columns kk that contribute to column j:
            // op(A)[kk, j] for kk before j in solve order.
            if upper {
                for kk in 0..j {
                    s = neg_mul_add(b[i + kk * ldb], aop(kk, j), s);
                }
            } else {
                for kk in (j + 1)..n {
                    s = neg_mul_add(b[i + kk * ldb], aop(kk, j), s);
                }
            }
            b[i + j * ldb] = finish(diag, s, aop(j, j));
        }
    }
}

/// `acc - a*x`, the substitution update step.
#[inline(always)]
fn neg_mul_add<T: Float>(a: T, x: T, acc: T) -> T {
    acc.sub(a.mul(x))
}

#[inline(always)]
fn finish<T: Float>(diag: Diag, s: T, a_ii: T) -> T {
    match diag {
        Diag::Unit => s,
        Diag::NonUnit => s.div(a_ii),
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops mirror the math in reference checks
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn naive(
        ta: Transpose,
        tb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
    ) -> alloc::vec::Vec<f64> {
        let mut c = alloc::vec![0.0f64; m * n];
        let at = |i: usize, p: usize| {
            if ta.is_transposed() {
                a[p + i * lda]
            } else {
                a[i + p * lda]
            }
        };
        let bt = |p: usize, j: usize| {
            if tb.is_transposed() {
                b[j + p * ldb]
            } else {
                b[p + j * ldb]
            }
        };
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += at(i, p) * bt(p, j);
                }
                c[i + j * m] = acc;
            }
        }
        c
    }

    #[test]
    fn try_gemm_ok_and_errors() {
        use crate::BlasError;
        let (m, n, k) = (4, 3, 5);
        let a = vec![1.0f64; m * k];
        let b = vec![1.0f64; k * n];
        let mut c = vec![0.0f64; m * n];

        // Valid call succeeds.
        assert!(try_gemm(
            Layout::ColMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c,
            m,
        )
        .is_ok());

        // lda too small (m=4 but lda=3).
        let e = try_gemm(
            Layout::ColMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            3,
            &b,
            k,
            0.0,
            &mut c,
            m,
        )
        .unwrap_err();
        assert_eq!(
            e,
            BlasError::InvalidLeadingDim {
                which: "a",
                got: 3,
                min: 4
            }
        );

        // C buffer too small.
        let mut small = vec![0.0f64; m * n - 1];
        let e = try_gemm(
            Layout::ColMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut small,
            m,
        )
        .unwrap_err();
        assert!(matches!(e, BlasError::BufferTooSmall { which: "c", .. }));
    }

    #[test]
    fn gemm_nn_colmajor() {
        let (m, n, k) = (6, 5, 7);
        let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i * 2) as f64).collect();
        let mut c = vec![0.0; m * n];
        gemm(
            Layout::ColMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c,
            m,
        );
        let want = naive(Transpose::None, Transpose::None, m, n, k, &a, m, &b, k);
        for (g, w) in c.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9);
        }
    }

    #[test]
    fn gemm_tn_colmajor() {
        // A transposed: stored as k×m.
        let (m, n, k) = (4, 5, 6);
        let a: Vec<f64> = (0..k * m).map(|i| (i as f64).sqrt()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.1).collect();
        let mut c = vec![0.0; m * n];
        gemm(
            Layout::ColMajor,
            Transpose::Trans,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            k,
            0.0,
            &mut c,
            m,
        );
        let want = naive(Transpose::Trans, Transpose::None, m, n, k, &a, k, &b, k);
        for (g, w) in c.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9);
        }
    }

    #[test]
    fn gemm_rowmajor_nn() {
        let (m, n, k) = (5, 4, 3);
        // Row-major A (m×k, lda=k), B (k×n, ldb=n), C (m×n, ldc=n).
        let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i + 1) as f64).collect();
        let mut c = vec![0.0; m * n];
        gemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c,
            n,
        );
        // Reference row-major multiply.
        let mut want = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                want[i * n + j] = acc;
            }
        }
        for (g, w) in c.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9);
        }
    }

    #[test]
    fn syrk_lower_notrans() {
        // A is 3×2 col-major; C = A·Aᵀ (3×3), lower triangle.
        let (n, k) = (3usize, 2usize);
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // cols [1,2,3],[4,5,6]
        let mut c = vec![0.0f64; n * n];
        syrk(
            Uplo::Lower,
            Transpose::None,
            n,
            k,
            1.0,
            &a,
            n,
            0.0,
            &mut c,
            n,
        );
        // Reference full A·Aᵀ then check lower triangle.
        let aem = |i: usize, p: usize| a[i + p * n];
        for j in 0..n {
            for i in j..n {
                let mut want = 0.0;
                for p in 0..k {
                    want += aem(i, p) * aem(j, p);
                }
                assert!((c[i + j * n] - want).abs() < 1e-12, "({i},{j})");
            }
        }
    }

    #[test]
    fn symm_left_upper() {
        // Symmetric A (2×2, upper stored), B (2×2). C = A·B.
        let a = vec![2.0f64, 0.0 /*unused lower*/, 1.0, 3.0]; // A=[[2,1],[1,3]]
        let b = vec![1.0f64, 2.0, 3.0, 4.0]; // cols [1,2],[3,4]
        let mut c = vec![0.0f64; 4];
        symm(
            Side::Left,
            Uplo::Upper,
            2,
            2,
            1.0,
            &a,
            2,
            &b,
            2,
            0.0,
            &mut c,
            2,
        );
        // A=[[2,1],[1,3]] ; B cols (1,2),(3,4)
        // C col0 = A·(1,2)=(2*1+1*2, 1*1+3*2)=(4,7); col1=A·(3,4)=(6+4,3+12)=(10,15)
        assert_eq!(c, vec![4.0, 7.0, 10.0, 15.0]);
    }

    #[test]
    fn syrk_trans_upper_larger() {
        // op(A)=Aᵀ: A stored k×n. C = Aᵀ·A (n×n), upper, with alpha/beta.
        let (n, k) = (9usize, 11usize);
        let a: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.31).sin()).collect();
        let mut c: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.1).collect();
        let c0 = c.clone();
        syrk(
            Uplo::Upper,
            Transpose::Trans,
            n,
            k,
            2.0,
            &a,
            k,
            3.0,
            &mut c,
            n,
        );
        // Reference: 2·sum_p A[p,i]A[p,j] + 3·C0, upper triangle.
        for j in 0..n {
            for i in 0..=j {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[p + i * k] * a[p + j * k];
                }
                let want = 2.0 * acc + 3.0 * c0[i + j * n];
                assert!((c[i + j * n] - want).abs() < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn symm_right_lower_larger() {
        // C = alpha·B·A + beta·C, A symmetric n×n lower-stored.
        let (m, n) = (7usize, 6usize);
        let a_lower: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.17).cos()).collect();
        let b: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.05).collect();
        let mut c = vec![0.0f64; m * n];
        symm(
            Side::Right,
            Uplo::Lower,
            m,
            n,
            1.0,
            &a_lower,
            n,
            &b,
            m,
            0.0,
            &mut c,
            m,
        );
        // Reference with full symmetric A from lower triangle.
        let sym = |i: usize, j: usize| {
            if i >= j {
                a_lower[i + j * n]
            } else {
                a_lower[j + i * n]
            }
        };
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += b[i + p * m] * sym(p, j);
                }
                assert!((c[i + j * m] - acc).abs() < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn trsm_left_lower_notrans() {
        // Solve L·X = B for lower-triangular L (3×3), one RHS.
        let l = vec![
            2.0f64, 1.0, 1.0, // col0: L[0,0]=2,L[1,0]=1,L[2,0]=1
            0.0, 3.0, 2.0, //   col1: L[1,1]=3,L[2,1]=2
            0.0, 0.0, 4.0, //   col2: L[2,2]=4
        ];
        // Pick X=(1,2,3); B = L·X.
        let xtrue = [1.0f64, 2.0, 3.0];
        let lat = |i: usize, j: usize| l[i + j * 3];
        let mut b = vec![0.0f64; 3];
        for i in 0..3 {
            let mut s = 0.0;
            for j in 0..=i {
                s += lat(i, j) * xtrue[j];
            }
            b[i] = s;
        }
        trsm(
            Side::Left,
            Uplo::Lower,
            Transpose::None,
            Diag::NonUnit,
            3,
            1,
            1.0,
            &l,
            3,
            &mut b,
            3,
        );
        for (got, want) in b.iter().zip(&xtrue) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn trsm_right_upper_roundtrip() {
        // Solve X·U = B, then verify X·U == B by recomputing.
        let (m, n) = (2usize, 3usize);
        let u = vec![2.0f64, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]; // upper 3×3 col-major
        let b0 = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2×3
        let mut b = b0.clone();
        trsm(
            Side::Right,
            Uplo::Upper,
            Transpose::None,
            Diag::NonUnit,
            m,
            n,
            1.0,
            &u,
            n,
            &mut b,
            m,
        );
        // Recompute X·U and compare to b0.
        let uat = |i: usize, j: usize| u[i + j * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..n {
                    s += b[i + kk * m] * uat(kk, j);
                }
                assert!((s - b0[i + j * m]).abs() < 1e-10, "({i},{j}) {s}");
            }
        }
    }
}
