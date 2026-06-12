//! Hand-written AArch64 NEON GEMM microkernel (column-major C).
//!
//! Computes one `MR×NR` register tile `C += alpha · (Ap · Bp)` from packed
//! panels, keeping the C accumulator entirely in NEON registers.
//!
//! Layout choice for **column-major C**: the `MR` row dimension goes in vector
//! lanes (rows are contiguous down a C column), and the `NR` column dimension
//! is the broadcast scalar. With `MR=8` that's two `float32x4_t` per C column;
//! `NR=8` columns give 16 accumulator vectors. Each rank-1 update loads the A
//! column once (2 vectors) and broadcasts each of the 8 B values. Writeback is a
//! contiguous vector store per C column — no lane extraction.
//!
//! Packed layout (matches [`super::gemm`]): `ap[p*MR + r]`, `bp[p*NR + s]`.

use core::arch::aarch64::*;

/// f32 microkernel, fixed `MR=8 × NR=8` tile.
///
/// # Safety
/// `ap` ≥ `kc*8` elems, `bp` ≥ `kc*8` elems; the C tile `[ci..ci+mr, cj..cj+nr]`
/// in bounds; `mr,nr ≤ 8`. Reached only on aarch64 (NEON baseline).
#[target_feature(enable = "neon")]
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
    // c_lo[col], c_hi[col]: rows 0..4 and 4..8 of C column `col`.
    let z = vdupq_n_f32(0.0);
    let mut lo = [z; 8];
    let mut hi = [z; 8];

    let mut p = 0;
    while p < kc {
        let a = ap.add(p * 8);
        let b = bp.add(p * 8);
        let a_lo = vld1q_f32(a); // rows 0..4 of A column p
        let a_hi = vld1q_f32(a.add(4)); // rows 4..8

        // For each of the 8 C columns, broadcast B[p,col] and FMA.
        macro_rules! col {
            ($s:literal) => {{
                let bs = vld1q_dup_f32(b.add($s));
                lo[$s] = vfmaq_f32(lo[$s], a_lo, bs);
                hi[$s] = vfmaq_f32(hi[$s], a_hi, bs);
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

    let va = vdupq_n_f32(alpha);
    if mr == 8 && nr == 8 {
        // Full tile: read-modify-write each C column with two vector ops.
        for s in 0..8 {
            let cptr = c.add(ci + (cj + s) * ldc);
            let c_lo = vld1q_f32(cptr);
            let c_hi = vld1q_f32(cptr.add(4));
            vst1q_f32(cptr, vfmaq_f32(c_lo, va, lo[s]));
            vst1q_f32(cptr.add(4), vfmaq_f32(c_hi, va, hi[s]));
        }
    } else {
        // Edge tile: spill to scratch and write only the live mr×nr corner.
        let mut buf = [0.0f32; 8];
        for s in 0..nr {
            vst1q_f32(buf.as_mut_ptr(), lo[s]);
            vst1q_f32(buf.as_mut_ptr().add(4), hi[s]);
            let col = c.add(ci + (cj + s) * ldc);
            for r in 0..mr {
                *col.add(r) += alpha * buf[r];
            }
        }
    }
}

/// f64 microkernel, fixed `MR=8 × NR=8` tile (four `float64x2_t` per column).
///
/// # Safety
/// As [`micro_8x8_f32`], with `f64` element type.
#[target_feature(enable = "neon")]
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
    let va = vdupq_n_f64(alpha);
    if mr == 8 && nr == 8 {
        // Full tile: process the 8 C columns in two groups of 4 so only 16
        // accumulator vectors (4 row-pairs × 4 cols) are live at once instead of
        // 32. That keeps the whole tile in the register file with headroom for
        // the A/B/C operands — the previous 32-accumulator version spilled.
        col_group_8x4(kc, va, ap, bp, c, ci, cj, ldc, 0);
        col_group_8x4(kc, va, ap, bp, c, ci, cj, ldc, 4);
    } else {
        // Edge tile: accumulate the live mr×nr corner with scalar spill.
        let z = vdupq_n_f64(0.0);
        let mut q0 = [z; 8];
        let mut q1 = [z; 8];
        let mut q2 = [z; 8];
        let mut q3 = [z; 8];
        let mut p = 0;
        while p < kc {
            let a = ap.add(p * 8);
            let b = bp.add(p * 8);
            let a0 = vld1q_f64(a);
            let a1 = vld1q_f64(a.add(2));
            let a2 = vld1q_f64(a.add(4));
            let a3 = vld1q_f64(a.add(6));
            for s in 0..nr {
                let bs = vld1q_dup_f64(b.add(s));
                q0[s] = vfmaq_f64(q0[s], a0, bs);
                q1[s] = vfmaq_f64(q1[s], a1, bs);
                q2[s] = vfmaq_f64(q2[s], a2, bs);
                q3[s] = vfmaq_f64(q3[s], a3, bs);
            }
            p += 1;
        }
        let mut buf = [0.0f64; 8];
        for s in 0..nr {
            vst1q_f64(buf.as_mut_ptr(), q0[s]);
            vst1q_f64(buf.as_mut_ptr().add(2), q1[s]);
            vst1q_f64(buf.as_mut_ptr().add(4), q2[s]);
            vst1q_f64(buf.as_mut_ptr().add(6), q3[s]);
            let col = c.add(ci + (cj + s) * ldc);
            for r in 0..mr {
                *col.add(r) += alpha * buf[r];
            }
        }
    }
}

/// Accumulate and write back an `8×4` slab of the C tile (rows 0..8, columns
/// `cj+col0 .. cj+col0+4`). 16 live accumulators.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn col_group_8x4(
    kc: usize,
    va: float64x2_t,
    ap: *const f64,
    bp: *const f64,
    c: *mut f64,
    ci: usize,
    cj: usize,
    ldc: usize,
    col0: usize,
) {
    let z = vdupq_n_f64(0.0);
    // q[col][rowpair]
    let mut q0 = [z; 4];
    let mut q1 = [z; 4];
    let mut q2 = [z; 4];
    let mut q3 = [z; 4];
    let mut p = 0;
    while p < kc {
        let a = ap.add(p * 8);
        let b = bp.add(p * 8 + col0);
        let a0 = vld1q_f64(a);
        let a1 = vld1q_f64(a.add(2));
        let a2 = vld1q_f64(a.add(4));
        let a3 = vld1q_f64(a.add(6));
        macro_rules! col {
            ($s:literal) => {{
                let bs = vld1q_dup_f64(b.add($s));
                q0[$s] = vfmaq_f64(q0[$s], a0, bs);
                q1[$s] = vfmaq_f64(q1[$s], a1, bs);
                q2[$s] = vfmaq_f64(q2[$s], a2, bs);
                q3[$s] = vfmaq_f64(q3[$s], a3, bs);
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
        let c0 = vld1q_f64(cptr);
        let c1 = vld1q_f64(cptr.add(2));
        let c2 = vld1q_f64(cptr.add(4));
        let c3 = vld1q_f64(cptr.add(6));
        vst1q_f64(cptr, vfmaq_f64(c0, va, q0[s]));
        vst1q_f64(cptr.add(2), vfmaq_f64(c1, va, q1[s]));
        vst1q_f64(cptr.add(4), vfmaq_f64(c2, va, q2[s]));
        vst1q_f64(cptr.add(6), vfmaq_f64(c3, va, q3[s]));
    }
}
