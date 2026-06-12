//! Cache-blocked, packed GEMM following the GotoBLAS / BLIS structure.
//!
//! Computes `C := alpha * A * B + beta * C` for column-major operands. The
//! Level-3 module reduces all transpose/layout combinations to this one shape
//! by choosing strides, so the hot loop here only ever sees column-major
//! no-transpose panels.
//!
//! ## Structure
//!
//! ```text
//! for jc in 0..N step NC          // B columns / C columns  (L3 cache)
//!   for pc in 0..K step KC        // shared dimension        (L2 cache)
//!     pack B[pc:pc+KC, jc:jc+NC] -> Bp      (KC×NC, panel-major)
//!     for ic in 0..M step MC      // A rows / C rows         (L2 cache)
//!       pack A[ic:ic+MC, pc:pc+KC] -> Ap    (MC×KC, panel-major)
//!       macrokernel(Ap, Bp, C[ic.., jc..])  // register-blocked microkernel
//! ```
//!
//! Packing makes both operands streamed contiguously in the microkernel, which
//! is the single biggest win over a naive triple loop.

use crate::Float;
#[cfg(feature = "std")]
use alloc::vec::Vec;

/// Reusable GEMM packing scratch.
///
/// `with_scratch` hands the driver two zero-initialized buffers of at least the
/// requested lengths and runs `f` with them. Under `std` the buffers are kept in
/// thread-local storage and grown on demand, so repeated GEMM calls on a thread
/// reuse the same allocation — eliminating the per-call heap traffic that
/// dominates small-matrix runtime. Without `std` it allocates fresh each call.
///
/// `#[doc(hidden)]`: implementation detail, not stable surface.
#[doc(hidden)]
pub trait Scratch: Sized {
    /// Run `f` with packing buffers of length ≥ `a_len` / `b_len`.
    fn with_scratch<R>(
        a_len: usize,
        b_len: usize,
        f: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R;
}

#[cfg(feature = "std")]
macro_rules! impl_scratch {
    ($ty:ty, $key:ident) => {
        thread_local! {
            static $key: core::cell::RefCell<(Vec<$ty>, Vec<$ty>)> =
                const { core::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        impl Scratch for $ty {
            #[inline]
            fn with_scratch<R>(
                a_len: usize,
                b_len: usize,
                f: impl FnOnce(&mut [Self], &mut [Self]) -> R,
            ) -> R {
                $key.with(|cell| {
                    let mut bufs = cell.borrow_mut();
                    let (pa, pb) = &mut *bufs;
                    grow_zeroed(pa, a_len);
                    grow_zeroed(pb, b_len);
                    f(&mut pa[..a_len], &mut pb[..b_len])
                })
            }
        }
    };
}

#[cfg(feature = "std")]
impl_scratch!(f32, SCRATCH_F32);
#[cfg(feature = "std")]
impl_scratch!(f64, SCRATCH_F64);

/// Ensure `v` has length ≥ `len`, zero-filling any reused/new region so the
/// MR/NR zero-padding the packers rely on is always present.
#[cfg(feature = "std")]
#[inline]
fn grow_zeroed<T: Float>(v: &mut Vec<T>, len: usize) {
    if v.len() < len {
        v.resize(len, T::ZERO);
    } else {
        for x in &mut v[..len] {
            *x = T::ZERO;
        }
    }
}

// no_std: no thread-locals, so just allocate (still correct, just not reused).
#[cfg(not(feature = "std"))]
impl<T: Float> Scratch for T {
    #[inline]
    fn with_scratch<R>(
        a_len: usize,
        b_len: usize,
        f: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R {
        let mut pa = alloc::vec![T::ZERO; a_len];
        let mut pb = alloc::vec![T::ZERO; b_len];
        f(&mut pa, &mut pb)
    }
}

/// Register-block dimensions of the microkernel (rows × cols of C per call).
/// Chosen to fit the accumulator tile in vector registers. Tunable per arch;
/// these are conservative values that work for NEON (32× 128-bit) and AVX2.
const MR: usize = 8;
const NR: usize = 8;

/// Cache-blocking parameters. Sized so that an `MC×KC` packed-A panel fits L2
/// and a `KC×NC` packed-B panel fits L3, with a `KC×NR` slice of B in L1.
///
/// These suit a large-L2 core (e.g. Apple M-series, 128 KB L1d / 16 MB L2);
/// empirically GEMM throughput here is insensitive to block size over a wide
/// range (the kernel is compute-bound), so a fixed generous default beats the
/// complexity of full runtime autotuning. `MC`/`KC` are clamped to the problem.
const MC: usize = 384;
const KC: usize = 512;
const NC: usize = 4096;

/// Top-level GEMM on column-major slices.
///
/// - `a`: `m × k`, column-major, leading dimension `lda`
/// - `b`: `k × n`, column-major, leading dimension `ldb`
/// - `c`: `m × n`, column-major, leading dimension `ldc`
///
/// Computes `c := alpha * a * b + beta * c`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_contig<T: Float>(
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
    if m == 0 || n == 0 {
        return;
    }
    // Apply beta to C up front; the macrokernel then only accumulates.
    apply_beta(m, n, beta, c, ldc);
    if k == 0 || alpha == T::ZERO {
        return;
    }

    // Each NC-wide column band of C is an independent unit of work: it reads all
    // of A and its own slice of B, and writes a disjoint set of C columns. That
    // makes the `jc` loop embarrassingly parallel — the `threads` feature farms
    // the bands out to rayon, the serial build just iterates them.
    #[cfg(feature = "threads")]
    {
        // Parallelize once the problem is big enough that thread overhead pays
        // off: enough columns to fill the pool, and enough total work that the
        // packing/spawn cost is amortized.
        let big_enough = n >= 2 * NR && (m * n * k) >= 1 << 18;
        if big_enough && rayon::current_num_threads() > 1 {
            return gemm_parallel(m, n, k, alpha, a, lda, b, ldb, c, ldc);
        }
    }

    let mut jc = 0;
    while jc < n {
        let nc = NC.min(n - jc);
        // C columns [jc, jc+nc) start at offset jc*ldc.
        let cband = &mut c[jc * ldc..];
        gemm_band(m, nc, k, alpha, a, lda, b, ldb, jc, cband, ldc);
        jc += NC;
    }
}

/// Compute one column band of C: `C[:, j0..j0+nc] += alpha · A · B[:, j0..j0+nc]`.
///
/// `cband` points at the first column of the band (`c + j0*ldc`); columns within
/// it are addressed `0..nc` relative to that. `b`/`ldb` and `j0` select the
/// band's B columns. Beta has already been applied by the caller.
#[allow(clippy::too_many_arguments)]
fn gemm_band<T: Float>(
    m: usize,
    nc: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    j0: usize,
    cband: &mut [T],
    ldc: usize,
) {
    // Per-band (hence per-thread) packing scratch, right-sized to the blocks.
    let a_cap = round_up(MC.min(m), MR) * KC.min(k);
    let b_cap = KC.min(k) * round_up(nc, NR);

    T::with_scratch(a_cap, b_cap, |pack_a, pack_b| {
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            // Band-local B starts at column j0; pack reads B[pc.., j0..j0+nc].
            pack_b_panel(kc, nc, b, ldb, pc, j0, pack_b);
            let mut ic = 0;
            while ic < m {
                let mc = MC.min(m - ic);
                pack_a_panel(mc, kc, a, lda, ic, pc, pack_a);
                // Within the band, C columns are 0-based, so cj = 0.
                macrokernel(mc, nc, kc, alpha, pack_a, pack_b, cband, ldc, ic, 0);
                ic += MC;
            }
            pc += KC;
        }
    });
}

/// Parallel driver: split C into NC-wide column bands and run them on rayon.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn gemm_parallel<T: Float>(
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    c: &mut [T],
    ldc: usize,
) {
    use rayon::prelude::*;

    // Pick a column-chunk width that yields several tasks per worker (for load
    // balance) while staying NR-aligned and no wider than NC (cache blocking).
    // Aiming for ~4 tasks/thread keeps the rayon work-stealing queues full even
    // when chunk costs vary.
    let threads = rayon::current_num_threads();
    let target_tasks = (threads * 4).max(1);
    let mut chunk_cols = n.div_ceil(target_tasks);
    chunk_cols = round_up(chunk_cols.max(NR), NR).min(NC);

    // Column-major: a `chunk_cols`-column slab is `chunk_cols*ldc` consecutive
    // elements, so `chunks_mut` carves disjoint, ordered slabs.
    let slab = chunk_cols * ldc;
    let chunks: alloc::vec::Vec<(usize, &mut [T])> = c
        .chunks_mut(slab)
        .enumerate()
        .map(|(ci, chunk)| (ci * chunk_cols, chunk))
        .collect();

    chunks.into_par_iter().for_each(|(j0, cband)| {
        let nc = chunk_cols.min(n - j0);
        gemm_band(m, nc, k, alpha, a, lda, b, ldb, j0, cband, ldc);
    });
}

/// Round `x` up to the next multiple of `m` (`m` a power-of-two block size).
#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// `c := beta * c` over an `m×n` column-major block.
fn apply_beta<T: Float>(m: usize, n: usize, beta: T, c: &mut [T], ldc: usize) {
    if beta == T::ONE {
        return;
    }
    for j in 0..n {
        let col = &mut c[j * ldc..j * ldc + m];
        if beta == T::ZERO {
            for v in col.iter_mut() {
                *v = T::ZERO;
            }
        } else {
            for v in col.iter_mut() {
                *v = beta.mul(*v);
            }
        }
    }
}

/// Pack `A[ic:ic+mc, pc:pc+kc]` (column-major source) into `MR`-row panels:
/// the buffer holds, for each row-panel, all `kc` columns contiguously so the
/// microkernel reads A as a flat stream.
fn pack_a_panel<T: Float>(
    mc: usize,
    kc: usize,
    a: &[T],
    lda: usize,
    ic: usize,
    pc: usize,
    pack: &mut [T],
) {
    let mut dst = 0;
    let mut i = 0;
    while i < mc {
        let mr = MR.min(mc - i);
        for p in 0..kc {
            let acol = pc + p;
            for r in 0..mr {
                pack[dst] = a[(ic + i + r) + acol * lda];
                dst += 1;
            }
            // Zero-pad the partial row-panel up to MR so the microkernel can
            // always assume a full MR stride.
            for _ in mr..MR {
                pack[dst] = T::ZERO;
                dst += 1;
            }
        }
        i += MR;
    }
}

/// Pack `B[pc:pc+kc, jc:jc+nc]` (column-major source) into `NR`-column panels.
fn pack_b_panel<T: Float>(
    kc: usize,
    nc: usize,
    b: &[T],
    ldb: usize,
    pc: usize,
    jc: usize,
    pack: &mut [T],
) {
    let mut dst = 0;
    let mut j = 0;
    while j < nc {
        let nr = NR.min(nc - j);
        for p in 0..kc {
            let brow = pc + p;
            for c in 0..nr {
                pack[dst] = b[brow + (jc + j + c) * ldb];
                dst += 1;
            }
            for _ in nr..NR {
                pack[dst] = T::ZERO;
                dst += 1;
            }
        }
        j += NR;
    }
}

/// Iterate the packed panels in `MR×NR` register tiles and call the microkernel.
#[allow(clippy::too_many_arguments)]
fn macrokernel<T: Float>(
    mc: usize,
    nc: usize,
    kc: usize,
    alpha: T,
    pack_a: &[T],
    pack_b: &[T],
    c: &mut [T],
    ldc: usize,
    ic: usize,
    jc: usize,
) {
    let mut j = 0;
    while j < nc {
        let nr = NR.min(nc - j);
        let bp = &pack_b[(j / NR) * (kc * NR)..];
        let mut i = 0;
        while i < mc {
            let mr = MR.min(mc - i);
            let ap = &pack_a[(i / MR) * (kc * MR)..];
            micro_tile(mr, nr, kc, alpha, ap, bp, c, ldc, ic + i, jc + j);
            i += MR;
        }
        j += NR;
    }
}

/// Compute one `MR×NR` tile of C from packed panels.
///
/// Accumulates into a small stack array, then writes back with `alpha`. The
/// tile dims are compile-time-bounded (`MR`,`NR`) so LLVM keeps the accumulator
/// in registers and vectorizes the rank-1 update over the `kc` loop.
#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_tile<T: Float>(
    mr: usize,
    nr: usize,
    kc: usize,
    alpha: T,
    ap: &[T],
    bp: &[T],
    c: &mut [T],
    ldc: usize,
    ci: usize,
    cj: usize,
) {
    // On aarch64/x86_64 the type carries a hand-written SIMD microkernel via the
    // MicroKernel trait (NEON always; AVX2 when detected, else scalar inside).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        // SAFETY: panels are sized kc*MR / kc*NR by the packer; the C tile
        // (ci..ci+mr, cj..cj+nr) is in bounds by construction; mr,nr ≤ MR,NR.
        unsafe {
            T::micro(
                mr,
                nr,
                kc,
                alpha,
                ap.as_ptr(),
                bp.as_ptr(),
                c.as_mut_ptr(),
                ldc,
                ci,
                cj,
            );
        }
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    micro_tile_scalar(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
}

/// Portable register-blocked microkernel: the no-SIMD fallback, and the runtime
/// fallback x86-64 uses when AVX2 isn't present. Dead only on aarch64 (NEON is
/// unconditional there).
#[allow(clippy::too_many_arguments)]
#[inline]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn micro_tile_scalar<T: Float>(
    mr: usize,
    nr: usize,
    kc: usize,
    alpha: T,
    ap: &[T],
    bp: &[T],
    c: &mut [T],
    ldc: usize,
    ci: usize,
    cj: usize,
) {
    let mut acc = [[T::ZERO; NR]; MR];
    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;
        for r in 0..MR {
            let ar = ap[a_off + r];
            let accr = &mut acc[r];
            for s in 0..NR {
                accr[s] = ar.mul_add(bp[b_off + s], accr[s]);
            }
        }
    }
    for s in 0..nr {
        let col = (cj + s) * ldc;
        for r in 0..mr {
            let idx = ci + r + col;
            c[idx] = alpha.mul_add(acc[r][s], c[idx]);
        }
    }
}

/// Per-type GEMM microkernel selection (NEON on aarch64, AVX2 on x86-64).
///
/// `#[doc(hidden)]`, not stable surface — exists only so the generic driver can
/// reach the type-specific SIMD kernel without specialization. On x86-64 the
/// `micro` impl does a runtime `avx2`+`fma` check and falls back to the scalar
/// tile when absent.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[doc(hidden)]
pub trait MicroKernel: Copy {
    /// Compute `C[tile] += alpha * (Ap·Bp)` for the fixed `MR×NR` tile.
    ///
    /// # Safety
    /// `ap` ≥ `kc*MR`, `bp` ≥ `kc*NR`; C tile in bounds; `mr,nr ≤ MR,NR`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn micro(
        mr: usize,
        nr: usize,
        kc: usize,
        alpha: Self,
        ap: *const Self,
        bp: *const Self,
        c: *mut Self,
        ldc: usize,
        ci: usize,
        cj: usize,
    );
}

#[cfg(target_arch = "aarch64")]
impl MicroKernel for f32 {
    #[inline]
    unsafe fn micro(
        mr: usize,
        nr: usize,
        kc: usize,
        alpha: Self,
        ap: *const Self,
        bp: *const Self,
        c: *mut Self,
        ldc: usize,
        ci: usize,
        cj: usize,
    ) {
        super::gemm_neon::micro_8x8_f32(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
    }
}

#[cfg(target_arch = "aarch64")]
impl MicroKernel for f64 {
    #[inline]
    unsafe fn micro(
        mr: usize,
        nr: usize,
        kc: usize,
        alpha: Self,
        ap: *const Self,
        bp: *const Self,
        c: *mut Self,
        ldc: usize,
        ci: usize,
        cj: usize,
    ) {
        super::gemm_neon::micro_8x8_f64(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
    }
}

#[cfg(target_arch = "x86_64")]
impl MicroKernel for f32 {
    #[inline]
    unsafe fn micro(
        mr: usize,
        nr: usize,
        kc: usize,
        alpha: Self,
        ap: *const Self,
        bp: *const Self,
        c: *mut Self,
        ldc: usize,
        ci: usize,
        cj: usize,
    ) {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            super::gemm_avx2::micro_8x8_f32(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
        } else {
            micro_raw_scalar(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl MicroKernel for f64 {
    #[inline]
    unsafe fn micro(
        mr: usize,
        nr: usize,
        kc: usize,
        alpha: Self,
        ap: *const Self,
        bp: *const Self,
        c: *mut Self,
        ldc: usize,
        ci: usize,
        cj: usize,
    ) {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            super::gemm_avx2::micro_8x8_f64(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
        } else {
            micro_raw_scalar(mr, nr, kc, alpha, ap, bp, c, ldc, ci, cj);
        }
    }
}

/// Raw-pointer scalar microkernel — the x86-64 fallback when AVX2 is absent.
///
/// # Safety
/// Same contract as [`MicroKernel::micro`].
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn micro_raw_scalar<T: Float>(
    mr: usize,
    nr: usize,
    kc: usize,
    alpha: T,
    ap: *const T,
    bp: *const T,
    c: *mut T,
    ldc: usize,
    ci: usize,
    cj: usize,
) {
    let mut acc = [[T::ZERO; NR]; MR];
    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;
        for r in 0..MR {
            let ar = *ap.add(a_off + r);
            for s in 0..NR {
                acc[r][s] = ar.mul_add(*bp.add(b_off + s), acc[r][s]);
            }
        }
    }
    for s in 0..nr {
        for r in 0..mr {
            let cp = c.add(ci + r + (cj + s) * ldc);
            *cp = alpha.mul_add(acc[r][s], *cp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference column-major GEMM for cross-checking.
    fn ref_gemm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i + p * m] * b[p + j * k];
                }
                c[i + j * m] = acc;
            }
        }
    }

    #[test]
    fn gemm_matches_reference() {
        let (m, n, k) = (17, 13, 19); // deliberately non-multiples of MR/NR
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.5).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.3).cos()).collect();
        let mut c = vec![0.0f64; m * n];
        let mut expected = vec![0.0f64; m * n];
        ref_gemm(m, n, k, &a, &b, &mut expected);
        gemm_contig(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m);
        for (got, want) in c.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn gemm_alpha_beta() {
        let (m, n, k) = (8, 8, 8);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let mut c = vec![1.0f32; m * n];
        // C = 2*A*B + 3*C
        gemm_contig(m, n, k, 2.0, &a, m, &b, k, 3.0, &mut c, m);
        // Spot check (0,0): 2*sum_p a[p*m]*b[p] + 3*1
        let mut acc = 0.0f32;
        for p in 0..k {
            acc += a[p * m] * b[p];
        }
        assert!((c[0] - (2.0 * acc + 3.0)).abs() < 1e-3);
    }

    /// Directly exercise the AVX2 microkernel intrinsics against a scalar tile.
    /// Self-skips when the host lacks AVX2/FMA (so it's a no-op on CI runners
    /// without the feature, but real coverage where it exists).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_microkernel_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            eprintln!("skipping avx2_microkernel_matches_scalar: no AVX2/FMA");
            return;
        }
        let kc = 11usize;
        // Packed panels: ap[p*MR + r], bp[p*NR + s].
        let ap: Vec<f32> = (0..kc * MR).map(|i| (i as f32 * 0.123).sin()).collect();
        let bp: Vec<f32> = (0..kc * NR).map(|i| (i as f32 * 0.077).cos()).collect();
        let ldc = MR;
        let alpha = 1.5f32;

        let mut c_avx = vec![0.5f32; MR * NR];
        let mut c_ref = c_avx.clone();
        unsafe {
            super::gemm_avx2::micro_8x8_f32(
                MR,
                NR,
                kc,
                alpha,
                ap.as_ptr(),
                bp.as_ptr(),
                c_avx.as_mut_ptr(),
                ldc,
                0,
                0,
            );
            micro_raw_scalar(
                MR,
                NR,
                kc,
                alpha,
                ap.as_ptr(),
                bp.as_ptr(),
                c_ref.as_mut_ptr(),
                ldc,
                0,
                0,
            );
        }
        for (g, w) in c_avx.iter().zip(&c_ref) {
            assert!((g - w).abs() < 1e-4, "avx2 {g} vs scalar {w}");
        }
    }
}
