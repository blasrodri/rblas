//! Structured errors for the checked (`try_*`) entry points.
//!
//! The primary routines panic on violated preconditions — the idiomatic Rust
//! response to a programming error, and zero-cost on the hot path. The `try_*`
//! variants instead validate up front and return [`BlasError`] so callers that
//! receive dimensions from untrusted input can handle them gracefully.

use core::fmt;

/// Result alias for checked rblas routines.
pub type Result<T> = core::result::Result<T, BlasError>;

/// A precondition violation detected by a checked (`try_*`) routine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlasError {
    /// A leading dimension was smaller than the minimum its layout/size requires.
    InvalidLeadingDim {
        /// Which operand (`"a"`, `"b"`, `"c"`).
        which: &'static str,
        /// The leading dimension supplied.
        got: usize,
        /// The minimum acceptable value.
        min: usize,
    },
    /// An operand slice was shorter than its dimensions/leading dimension imply.
    BufferTooSmall {
        /// Which operand (`"a"`, `"b"`, `"c"`).
        which: &'static str,
        /// The slice length supplied.
        got: usize,
        /// The minimum required length.
        need: usize,
    },
    /// A stride/increment was zero where a positive value is required.
    ZeroStride {
        /// Which vector (`"x"`, `"y"`).
        which: &'static str,
    },
}

impl fmt::Display for BlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlasError::InvalidLeadingDim { which, got, min } => write!(
                f,
                "invalid leading dimension for `{which}`: got {got}, need ≥ {min}"
            ),
            BlasError::BufferTooSmall { which, got, need } => write!(
                f,
                "buffer `{which}` too small: got {got} elements, need ≥ {need}"
            ),
            BlasError::ZeroStride { which } => {
                write!(f, "stride for `{which}` must be positive (got 0)")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BlasError {}
