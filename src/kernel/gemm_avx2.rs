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
    let va = _mm256_set1_pd(alpha);
    if mr == 8 && nr == 8 {
        // Process the 8 columns in two groups of 4: only 8 __m256d accumulators
        // (lo[4]+hi[4]) live at once instead of 16, leaving YMM registers free
        // for the A-column and B-broadcast operands. The 16-accumulator version
        // spilled the whole register file every K iteration (the f32 tile is
        // fine — it needs only 8). Mirrors the NEON DGEMM fix.
        col_group_8x4(kc, va, ap, bp, c, ci, cj, ldc, 0);
        col_group_8x4(kc, va, ap, bp, c, ci, cj, ldc, 4);
    } else {
        // Edge tile: compute the live mr×nr corner with scalar spill.
        let mut lo = [_mm256_setzero_pd(); 8];
        let mut hi = [_mm256_setzero_pd(); 8];
        let mut p = 0;
        while p < kc {
            let a = ap.add(p * 8);
            let b = bp.add(p * 8);
            let a_lo = _mm256_loadu_pd(a);
            let a_hi = _mm256_loadu_pd(a.add(4));
            for s in 0..nr {
                let bs = _mm256_broadcast_sd(&*b.add(s));
                lo[s] = _mm256_fmadd_pd(a_lo, bs, lo[s]);
                hi[s] = _mm256_fmadd_pd(a_hi, bs, hi[s]);
            }
            p += 1;
        }
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

/// Accumulate and write back an `8×4` slab of the C tile (rows 0..8, columns
/// `cj+col0 .. cj+col0+4`). 8 live __m256d accumulators — no register spill.
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn col_group_8x4(
    kc: usize,
    va: __m256d,
    ap: *const f64,
    bp: *const f64,
    c: *mut f64,
    ci: usize,
    cj: usize,
    ldc: usize,
    col0: usize,
) {
    let mut lo = [_mm256_setzero_pd(); 4];
    let mut hi = [_mm256_setzero_pd(); 4];
    let mut p = 0;
    while p < kc {
        let a = ap.add(p * 8);
        let b = bp.add(p * 8 + col0);
        let a_lo = _mm256_loadu_pd(a);
        let a_hi = _mm256_loadu_pd(a.add(4));
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
        p += 1;
    }
    for s in 0..4 {
        let cptr = c.add(ci + (cj + col0 + s) * ldc);
        let c_lo = _mm256_loadu_pd(cptr);
        let c_hi = _mm256_loadu_pd(cptr.add(4));
        _mm256_storeu_pd(cptr, _mm256_fmadd_pd(va, lo[s], c_lo));
        _mm256_storeu_pd(cptr.add(4), _mm256_fmadd_pd(va, hi[s], c_hi));
    }
}
