//! Hand-written x86-64 AVX2+FMA GEMM microkernels (column-major C).
//!
//! Same layout convention as the NEON path ([`super::gemm_neon`]): the `MR` row
//! dimension lives in vector lanes (rows are contiguous down a C column) and the
//! `NR` column dimension is the broadcast scalar. Writeback is a contiguous
//! vector load/FMA/store per C column.
//!
//! AVX2 vectors are 256-bit: 8×f32 or 4×f64.
//! - f32, `MR=8`: one `__m256` per C column → 8 accumulators for `NR=8`.
//! - f64, `MR=8`: two `__m256d` per C column → 16 accumulators for `NR=8`.
//!
//! Both fit the 16-register YMM file with room for the A column and B broadcast.
//! Packed layout (matches [`super::gemm`]): `ap[p*MR + r]`, `bp[p*NR + s]`.

use core::arch::x86_64::*;

/// f32 microkernel, fixed `MR=8 × NR=8` tile.
///
/// # Safety
/// Reached only after `avx2`+`fma` runtime detection. `ap` ≥ `kc*8`, `bp` ≥
/// `kc*8`; the C tile `[ci..ci+mr, cj..cj+nr]` in bounds; `mr,nr ≤ 8`.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn micro_8x8_f32(
    mr: usize,
    nr: usize,
    kc: usize,
    alpha: f32,
    ap: *const f32,
    bp: *const f32,
    c: *mut f32,
    ldc: usize,
    ci: usize,
    cj: usize,
) {
    // One __m256 accumulator (8 rows) per C column.
    let mut acc = [_mm256_setzero_ps(); 8];

    let mut p = 0;
    while p < kc {
        let a = ap.add(p * 8);
        let b = bp.add(p * 8);
        let av = _mm256_loadu_ps(a); // rows 0..8 of A column p
        macro_rules! col {
            ($s:literal) => {{
                let bs = _mm256_broadcast_ss(&*b.add($s));
                acc[$s] = _mm256_fmadd_ps(av, bs, acc[$s]);
            }};
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
        p += 1;
    }

    let va = _mm256_set1_ps(alpha);
    if mr == 8 && nr == 8 {
        for s in 0..8 {
            let cptr = c.add(ci + (cj + s) * ldc);
            let cv = _mm256_loadu_ps(cptr);
            _mm256_storeu_ps(cptr, _mm256_fmadd_ps(va, acc[s], cv));
        }
    } else {
        let mut buf = [0.0f32; 8];
        for s in 0..nr {
            _mm256_storeu_ps(buf.as_mut_ptr(), acc[s]);
            let col = c.add(ci + (cj + s) * ldc);
            for r in 0..mr {
                *col.add(r) += alpha * buf[r];
            }
        }
    }
}

/// f64 microkernel, fixed `MR=8 × NR=8` tile (two `__m256d` per C column).
///
/// # Safety
/// As [`micro_8x8_f32`], with `f64` element type.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn micro_8x8_f64(
    mr: usize,
    nr: usize,
    kc: usize,
    alpha: f64,
    ap: *const f64,
    bp: *const f64,
    c: *mut f64,
    ldc: usize,
    ci: usize,
    cj: usize,
) {
    // Two __m256d halves (rows 0..4, 4..8) per C column.
    let mut lo = [_mm256_setzero_pd(); 8];
    let mut hi = [_mm256_setzero_pd(); 8];

    let mut p = 0;
    while p < kc {
        let a = ap.add(p * 8);
        let b = bp.add(p * 8);
        let a_lo = _mm256_loadu_pd(a); // rows 0..4
        let a_hi = _mm256_loadu_pd(a.add(4)); // rows 4..8
        macro_rules! col {
            ($s:literal) => {{
                let bs = _mm256_broadcast_sd(&*b.add($s));
                lo[$s] = _mm256_fmadd_pd(a_lo, bs, lo[$s]);
                hi[$s] = _mm256_fmadd_pd(a_hi, bs, hi[$s]);
            }};
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
        p += 1;
    }

    let va = _mm256_set1_pd(alpha);
    if mr == 8 && nr == 8 {
        for s in 0..8 {
            let cptr = c.add(ci + (cj + s) * ldc);
            let c_lo = _mm256_loadu_pd(cptr);
            let c_hi = _mm256_loadu_pd(cptr.add(4));
            _mm256_storeu_pd(cptr, _mm256_fmadd_pd(va, lo[s], c_lo));
            _mm256_storeu_pd(cptr.add(4), _mm256_fmadd_pd(va, hi[s], c_hi));
        }
    } else {
        let mut buf = [0.0f64; 8];
        for s in 0..nr {
            _mm256_storeu_pd(buf.as_mut_ptr(), lo[s]);
            _mm256_storeu_pd(buf.as_mut_ptr().add(4), hi[s]);
            let col = c.add(ci + (cj + s) * ldc);
            for r in 0..mr {
                *col.add(r) += alpha * buf[r];
            }
        }
    }
}
