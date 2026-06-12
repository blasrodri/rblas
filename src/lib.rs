//! # rblas — a pure-Rust BLAS
//!
//! A from-scratch implementation of the Basic Linear Algebra Subprograms in
//! safe-by-default Rust, with architecture-specific SIMD microkernels guarded
//! behind runtime feature detection.
//!
//! ## Design
//!
//! The crate is organized in the classic three levels:
//! - **Level 1** ([`level1`]): vector–vector ops (`axpy`, `dot`, `scal`, `nrm2`, …)
//! - **Level 2** ([`level2`]): matrix–vector ops (`gemv`, `ger`, …)
//! - **Level 3** ([`level3`]): matrix–matrix ops (`gemm`, `trsm`, `syrk`, …)
//!
//! Performance-critical kernels live in [`kernel`], dispatched at runtime to the
//! best implementation the host CPU supports (AVX2 on x86-64, NEON on aarch64,
//! scalar everywhere else). All `unsafe` SIMD is confined to that module behind
//! `is_x86_feature_detected!` / `cfg(target_arch)` gates.
//!
//! ## Numeric scope
//!
//! Generic over the [`Float`] trait (`f32` / `f64`). Matrices are described by
//! explicit dimensions and strides ([`Layout`]) so the same code serves
//! row-major and column-major callers without copying.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

// Level-3 packing buffers need a heap allocator. We pull in `alloc`
// unconditionally; a future no-alloc path would take caller-provided scratch.
extern crate alloc;

pub mod complex;
pub mod complex1;
pub mod complex2;
pub mod complex3;
mod error;
pub mod kernel;
pub mod level1;
pub mod level2;
pub mod level3;
mod types;

pub use complex::{Complex, C32, C64};
pub use error::{BlasError, Result};
pub use types::{Diag, Layout, Side, Transpose, Uplo};

/// Arch-specific kernel bundle folded into [`Float`] as a supertrait.
///
/// On aarch64/x86-64 it carries the SIMD GEMM microkernel; elsewhere it is
/// empty. This keeps the public [`Float`] trait body identical across targets.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[doc(hidden)]
pub trait ArchKernel: kernel::gemm::MicroKernel {}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl<T: kernel::gemm::MicroKernel> ArchKernel for T {}

/// Arch-specific kernel bundle (no extra requirements on other targets).
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[doc(hidden)]
pub trait ArchKernel {}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl<T> ArchKernel for T {}

/// Numeric element type supported by rblas routines.
///
/// Implemented for [`f32`] and [`f64`]. The trait abstracts the handful of
/// primitive operations the generic (non-SIMD) reference paths need; SIMD
/// kernels are written per-type and per-arch in [`kernel`].
pub trait Float:
    Copy + PartialEq + Send + Sync + 'static + kernel::Element + kernel::gemm::Scratch + ArchKernel
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    /// `self + other`.
    fn add(self, other: Self) -> Self;
    /// `self * other`.
    fn mul(self, other: Self) -> Self;
    /// `self / other`.
    fn div(self, other: Self) -> Self;
    /// `self - other`.
    fn sub(self, other: Self) -> Self;
    /// Fused multiply-add: `self * a + b` (a true hardware FMA under `std`).
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// `sqrt(self)`.
    fn sqrt(self) -> Self;
    /// `self < other`.
    fn lt(self, other: Self) -> bool;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        self / other
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self - other
    }
    #[inline(always)]
    fn lt(self, other: Self) -> bool {
        self < other
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(feature = "std")]
        {
            f32::mul_add(self, a, b)
        }
        #[cfg(not(feature = "std"))]
        {
            self * a + b
        }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::from_bits(self.to_bits() & 0x7fff_ffff)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            f32::sqrt(self)
        }
        #[cfg(not(feature = "std"))]
        {
            sqrt_newton_f32(self)
        }
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        self / other
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self - other
    }
    #[inline(always)]
    fn lt(self, other: Self) -> bool {
        self < other
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(feature = "std")]
        {
            f64::mul_add(self, a, b)
        }
        #[cfg(not(feature = "std"))]
        {
            self * a + b
        }
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::from_bits(self.to_bits() & 0x7fff_ffff_ffff_ffff)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            f64::sqrt(self)
        }
        #[cfg(not(feature = "std"))]
        {
            sqrt_newton_f64(self)
        }
    }
}

#[cfg(not(feature = "std"))]
#[inline]
fn sqrt_newton_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f32::from_bits((x.to_bits() >> 1) + 0x1fc0_0000);
    for _ in 0..4 {
        g = 0.5 * (g + x / g);
    }
    g
}

#[cfg(not(feature = "std"))]
#[inline]
fn sqrt_newton_f64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f64::from_bits((x.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..5 {
        g = 0.5 * (g + x / g);
    }
    g
}
