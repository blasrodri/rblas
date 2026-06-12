//! Complex element type and the arithmetic the complex routines need.
//!
//! BLAS keeps real (S/D) and complex (C/Z) routines separate, and so do we:
//! [`Complex<T>`] is its own element type with its own Level-1/2/3 entry points
//! ([`crate::complex1`], [`crate::complex2`], [`crate::complex3`]). The heavy
//! Level-3 path ([`crate::complex3::gemm`]) reuses the fast real packed kernel
//! by splitting operands into real/imaginary planes (the "4M" decomposition).

use crate::Float;
use core::ops::{Add, Mul, Sub};

/// A complex number `re + i·im` over a real [`Float`] (`f32`/`f64`).
///
/// Memory layout is `#[repr(C)]` with `re` first, so a `&[Complex<T>]` is
/// exactly the interleaved `[re0, im0, re1, im1, …]` layout BLAS uses for its
/// `c`/`z` arrays — letting callers transmute between the two views if needed.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Float> Complex<T> {
    /// `re + i·im`.
    #[inline(always)]
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// `0 + 0i`.
    #[inline(always)]
    pub fn zero() -> Self {
        Complex {
            re: T::ZERO,
            im: T::ZERO,
        }
    }

    /// `1 + 0i`.
    #[inline(always)]
    pub fn one() -> Self {
        Complex {
            re: T::ONE,
            im: T::ZERO,
        }
    }

    /// Is this exactly `0 + 0i`?
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.re == T::ZERO && self.im == T::ZERO
    }

    /// Complex conjugate `re − i·im`.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: T::ZERO.sub(self.im),
        }
    }

    /// Squared magnitude `re² + im²`.
    #[inline(always)]
    pub fn norm_sqr(self) -> T {
        self.re.mul_add(self.re, self.im.mul(self.im))
    }

    /// Magnitude `sqrt(re² + im²)`.
    #[inline(always)]
    pub fn abs(self) -> T {
        self.norm_sqr().sqrt()
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Complex {
            re: self.re.add(o.re),
            im: self.im.add(o.im),
        }
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Complex {
            re: self.re.sub(o.re),
            im: self.im.sub(o.im),
        }
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        // (a+bi)(c+di) = (ac − bd) + (ad + bc)i
        Complex {
            re: self.re.mul_add(o.re, T::ZERO.sub(self.im.mul(o.im))),
            im: self.re.mul_add(o.im, self.im.mul(o.re)),
        }
    }
}

/// Single-precision complex (`c` in BLAS).
pub type C32 = Complex<f32>;
/// Double-precision complex (`z` in BLAS).
pub type C64 = Complex<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_and_conj() {
        let a = Complex::new(1.0f64, 2.0);
        let b = Complex::new(3.0f64, -1.0);
        let p = a * b; // (3+2) + (-1+6)i = 5 + 5i
        assert_eq!(p, Complex::new(5.0, 5.0));
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        assert!((a.abs() - 5.0f64.sqrt()).abs() < 1e-12);
    }
}
