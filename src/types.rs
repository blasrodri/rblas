//! Shared descriptors: matrix memory [`Layout`] and the BLAS [`Transpose`] op.

/// Memory layout of a dense matrix.
///
/// BLAS traditionally fixes column-major (Fortran) order, but rblas accepts
/// either so Rust callers holding row-major buffers don't have to transpose
/// first. Kernels translate a layout into the appropriate row/column strides.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Elements of a row are contiguous; `a[i][j]` lives at `i * lda + j`.
    RowMajor,
    /// Elements of a column are contiguous; `a[i][j]` lives at `i + j * lda`.
    ColMajor,
}

impl Layout {
    /// Return `(row_stride, col_stride)` in elements for the given leading
    /// dimension `lda`.
    #[inline(always)]
    pub fn strides(self, lda: usize) -> (usize, usize) {
        match self {
            Layout::RowMajor => (lda, 1),
            Layout::ColMajor => (1, lda),
        }
    }
}

/// Whether and how an input matrix is transposed before the operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transpose {
    /// Use the matrix as stored.
    None,
    /// Use the transpose `Aᵀ`.
    Trans,
    /// Use the conjugate transpose `Aᴴ` (identical to `Trans` for real types).
    ConjTrans,
}

impl Transpose {
    /// Does this op swap the row and column indices of the matrix?
    #[inline(always)]
    pub fn is_transposed(self) -> bool {
        matches!(self, Transpose::Trans | Transpose::ConjTrans)
    }
}

/// Which triangle of a symmetric/triangular matrix is referenced.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Uplo {
    /// The upper triangle (`i ≤ j`).
    Upper,
    /// The lower triangle (`i ≥ j`).
    Lower,
}

/// Which side a triangular matrix multiplies from (in `trsm`/`trmm`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    /// `op(A) · X` — `A` on the left.
    Left,
    /// `X · op(A)` — `A` on the right.
    Right,
}

/// Whether a triangular matrix has a unit (implicit-1) diagonal.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Diag {
    /// Diagonal entries are stored and used.
    NonUnit,
    /// Diagonal is treated as all ones (entries not referenced).
    Unit,
}
