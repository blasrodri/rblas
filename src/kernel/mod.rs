//! Performance kernels with runtime SIMD dispatch.
//!
//! Public functions here take **contiguous** slices (stride 1); the Level-1/2/3
//! modules peel off the strided/edge cases and only reach the kernels on the
//! hot, contiguous path. Each kernel chooses an implementation once, based on
//! runtime CPU feature detection, and falls back to a portable scalar version.
//!
//! All `unsafe` in the crate is contained in this module's arch submodules and
//! is justified by a successful `is_x86_feature_detected!` / `cfg(target_arch)`
//! guard before any intrinsic is reached.

// Index-based loops are deliberate in the SIMD/tile kernels (fixed unrolled
// bounds the optimizer can prove), and `needless_return` fires on the cfg-gated
// early returns that pick an arch path.
#![allow(clippy::needless_range_loop, clippy::needless_return)]

use crate::Float;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(target_arch = "aarch64")]
mod gemm_neon;

#[cfg(target_arch = "x86_64")]
mod gemm_avx2;

pub mod gemm;

pub use gemm::gemm_contig;

/// `y := alpha * x + y` over contiguous slices of equal length.
#[inline]
pub fn axpy_contig<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    debug_assert_eq!(x.len(), y.len());
    dispatch::axpy(alpha, x, y);
}

/// Dot product of two contiguous slices of equal length.
#[inline]
pub fn dot_contig<T: Float>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    dispatch::dot(x, y)
}

/// `x := alpha * x` over a contiguous slice.
#[inline]
pub fn scal_contig<T: Float>(alpha: T, x: &mut [T]) {
    dispatch::scal(alpha, x);
}

/// Type-erased dispatch layer.
///
/// Because Rust lacks specialization on stable, we route through a small
/// `Element` trait that each concrete float implements with its best kernel.
/// The generic `Float`-bounded callers above forward to it.
mod dispatch {
    use crate::Float;

    #[inline(always)]
    pub fn axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
        T::axpy(alpha, x, y);
    }
    #[inline(always)]
    pub fn dot<T: Float>(x: &[T], y: &[T]) -> T {
        T::dot(x, y)
    }
    #[inline(always)]
    pub fn scal<T: Float>(alpha: T, x: &mut [T]) {
        T::scal(alpha, x);
    }
}

/// Per-element kernel selection. Implemented for `f32` and `f64`; each method
/// performs runtime feature detection and dispatches accordingly.
///
/// This is a supertrait of [`crate::Float`], so any `T: Float` can reach these
/// kernels. It is `pub` only to satisfy the bound leak; it is `#[doc(hidden)]`
/// and not part of the stable surface.
#[doc(hidden)]
pub trait Element: Copy {
    fn axpy(alpha: Self, x: &[Self], y: &mut [Self]);
    fn dot(x: &[Self], y: &[Self]) -> Self;
    fn scal(alpha: Self, x: &mut [Self]);
}

macro_rules! impl_element {
    ($ty:ty, $axpy:ident, $dot:ident, $scal:ident) => {
        impl Element for $ty {
            #[inline]
            fn axpy(alpha: Self, x: &[Self], y: &mut [Self]) {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
                    {
                        // SAFETY: guarded by runtime detection of avx2+fma.
                        return unsafe { crate::kernel::avx2::$axpy(alpha, x, y) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is baseline-mandatory on aarch64.
                    return unsafe { crate::kernel::neon::$axpy(alpha, x, y) };
                }
                #[allow(unreachable_code)]
                crate::kernel::scalar::axpy(alpha, x, y)
            }

            #[inline]
            fn dot(x: &[Self], y: &[Self]) -> Self {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
                    {
                        // SAFETY: guarded by runtime detection of avx2+fma.
                        return unsafe { crate::kernel::avx2::$dot(x, y) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is baseline-mandatory on aarch64.
                    return unsafe { crate::kernel::neon::$dot(x, y) };
                }
                #[allow(unreachable_code)]
                crate::kernel::scalar::dot(x, y)
            }

            #[inline]
            fn scal(alpha: Self, x: &mut [Self]) {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: guarded by runtime detection of avx2.
                        return unsafe { crate::kernel::avx2::$scal(alpha, x) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is baseline-mandatory on aarch64.
                    return unsafe { crate::kernel::neon::$scal(alpha, x) };
                }
                #[allow(unreachable_code)]
                crate::kernel::scalar::scal(alpha, x)
            }
        }
    };
}

impl_element!(f32, axpy_f32, dot_f32, scal_f32);
impl_element!(f64, axpy_f64, dot_f64, scal_f64);
