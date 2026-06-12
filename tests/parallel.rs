//! Verifies the multi-threaded GEMM path (feature = "threads") produces the
//! same result as a single-thread reference. Only meaningful when the problem
//! crosses the NC column-band boundary, so `n` is chosen large enough to split
//! into several rayon tasks.
#![cfg(feature = "threads")]

use rblas::{level3, Layout, Transpose};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

#[test]
fn parallel_matches_serial() {
    // n > NC (4096) so the band partition yields multiple parallel tasks.
    let (m, n, k) = (64, 5000, 48);
    let mut s = 0xC0FFEEu64;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut s)).collect();

    let mut c_par = vec![0.0f32; m * n];
    level3::gemm(
        Layout::ColMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        0.0,
        &mut c_par,
        m,
    );

    // Independent reference: plain triple loop.
    let mut c_ref = vec![0.0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i + p * m] * b[p + j * k];
            }
            c_ref[i + j * m] = acc;
        }
    }

    let mut max_rel = 0.0f32;
    for (g, w) in c_par.iter().zip(&c_ref) {
        let denom = w.abs().max(1e-3);
        max_rel = max_rel.max((g - w).abs() / denom);
    }
    assert!(
        max_rel < 1e-2,
        "parallel GEMM diverged: max rel err {max_rel}"
    );
}
