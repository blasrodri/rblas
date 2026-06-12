//! Portable scalar kernels. Always correct, used as the fallback when no SIMD
//! target feature is available. Written to auto-vectorize reasonably under LLVM
//! but never relied upon to.

use crate::Float;

#[inline]
pub fn axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = alpha.mul_add(xi, *yi);
    }
}

#[inline]
pub fn dot<T: Float>(x: &[T], y: &[T]) -> T {
    // Four independent accumulators to expose ILP to the scalar backend.
    let mut a0 = T::ZERO;
    let mut a1 = T::ZERO;
    let mut a2 = T::ZERO;
    let mut a3 = T::ZERO;
    let chunks = x.len() / 4;
    for k in 0..chunks {
        let i = k * 4;
        a0 = x[i].mul_add(y[i], a0);
        a1 = x[i + 1].mul_add(y[i + 1], a1);
        a2 = x[i + 2].mul_add(y[i + 2], a2);
        a3 = x[i + 3].mul_add(y[i + 3], a3);
    }
    let mut acc = a0.add(a1).add(a2.add(a3));
    for i in (chunks * 4)..x.len() {
        acc = x[i].mul_add(y[i], acc);
    }
    acc
}

#[inline]
pub fn scal<T: Float>(alpha: T, x: &mut [T]) {
    for xi in x.iter_mut() {
        *xi = alpha.mul(*xi);
    }
}
